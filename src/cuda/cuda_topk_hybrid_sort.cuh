// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "cuda_topk.h"
#include "cuda_topk_warp_sort_helper.cuh"
#include "cuda_topk_common.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace hybrid_sort {

/**
 * @brief A portable, multi-kernel, host-planned hybrid Top-K algorithm.
 *
 * Algorithm Overview:
 * This algorithm is designed for portability and does not require cooperative launch.
 * It uses a host-side planner to orchestrate a multi-stage reduction process.
 *
 * 1.  **Host-Side Planning (`GetReductionPlan`)**: Before launching any kernels, the host
 * determines an optimal multi-step reduction plan. It greedily decides how many partitions
 * to merge in each step, aiming to minimize the number of kernel launches. It also selects
 * the best internal sorting algorithm for each step based on cached benchmark data.
 *
 * 2.  **Stage 1 (Partition Top-K)**: A standard kernel (`Stage1_FindPartitionsTopK`) is launched to
 * find the top `K_PADDED` candidates from each vocabulary partition. The internal sorting
 * algorithm is chosen based on cached benchmark data.
 *
 * 3.  **Stage 2 (Planned Reduction)**: A series of reduction kernels (`BlockReduceTopK`)
 * are launched according to the plan from step 1.
 *
 * Performance Characteristics:
 * -   **Strengths**: Its main advantage is portability, as it runs on GPUs that do not support
 * cooperative kernels. The host-side planner can make intelligent decisions to create an
 * efficient reduction strategy based on live benchmark data.
 * -   **Weaknesses**: The use of multiple kernel launches can introduce higher overhead compared to
 * single-kernel cooperative approaches like `iterative_sort` or `cascaded_sort`.
 */

// The internal sorting algorithm to be used inside the reduction kernel, chosen by the host-side planner.
enum class ReductionAlgorithm {
  WARP_BITONIC,
  WARP_MERGE_SORT,
  CUB_BLOCK_MERGE,
  CUB_RADIX_SORT
};

// Contains the parameters for a single reduction kernel launch.
struct ReductionStep {
  int partitions_per_block;      // How many candidate sets to merge in this step.
  int block_size;                // The thread block size for the kernel launch.
  ReductionAlgorithm algorithm;  // The internal sort algorithm to use.
};

using ReductionPlan = std::vector<ReductionStep>;
constexpr int kMaxPartitions = 256;
constexpr int kMaxItemsToSortPerBlock = 4096;  // Budget based on ~64KB of shared memory.

// A set of well-spaced candidate partition sizes.
constexpr std::array<int, 4> kCandidatePartitionSizes = {2816, 3328, 4096, 4864};

// --- Host-Side Planning Logic ---

/**
 * @brief Estimates the best partition size for the given vocabulary size.
 * The goal is to select a size that results in a total number of partitions
 * close to a power of two, which is ideal for reduction stages.
 */
inline int EstimateBestPartitionSize(int vocab_size) {
  constexpr std::array<int, 8> kPowerOfTwoTargets = {2, 4, 8, 16, 32, 64, 128, 256};

  int best_partition_size = 0;
  double min_waste_ratio = std::numeric_limits<double>::infinity();

  for (int p_size : kCandidatePartitionSizes) {
    int num_partitions = CeilDiv(vocab_size, p_size);

    if (num_partitions > kMaxPartitions) {
      continue;
    }

    // Find the smallest power-of-two target that is >= num_partitions
    auto it = std::lower_bound(kPowerOfTwoTargets.begin(), kPowerOfTwoTargets.end(), num_partitions);
    if (it != kPowerOfTwoTargets.end()) {
      int target_partitions = *it;
      // The "waste" is how many "empty" partitions we'd need to add to the target.
      // A lower waste ratio is better.
      double waste_ratio = static_cast<double>(target_partitions - num_partitions) / target_partitions;

      if (waste_ratio < min_waste_ratio) {
        min_waste_ratio = waste_ratio;
        best_partition_size = p_size;
      }
    }
  }

  // Fallback in case no suitable partition size was found.
  return (best_partition_size == 0) ? kCandidatePartitionSizes[0] : best_partition_size;
}

/**
 * @brief Creates a deterministic, multi-step reduction plan.
 * This function determines how many partitions to merge in each step to minimize
 * kernel launches, and which internal sort algorithm is best for each step based on benchmark data.
 */
inline ReductionPlan GetReductionPlan(int num_partitions, int k_padded, cudaStream_t stream) {
  ReductionPlan plan;
  int current_partitions = num_partitions;

  const auto& benchmarks = GetOrRunSortBenchmark(stream);

  while (current_partitions > 1) {
    // Determine the max number of partitions we can merge in one block based on shared memory budget.
    int max_p_per_block = (k_padded > 0) ? (kMaxItemsToSortPerBlock / k_padded) : 0;
    if (max_p_per_block <= 1) break;  // Cannot reduce further

    int p_per_block = 1;
    // Greedily select the largest reduction factor possible from our available kernels.
    constexpr std::array<int, 4> kReductionFactors = {16, 8, 4, 2};
    for (int factor : kReductionFactors) {
      if (factor <= current_partitions && factor <= max_p_per_block) {
        p_per_block = factor;
        break;
      }
    }

    if (p_per_block <= 1) {
      break;  // Stuck, cannot make progress
    }

    ReductionStep step;
    step.partitions_per_block = p_per_block;
    step.block_size = (k_padded <= 16) ? 128 : 256;

    int sort_size = k_padded * step.partitions_per_block;

    // Use benchmark results to pick the fastest algorithm for this sort size.
    SortAlgo best_algo = benchmarks.GetBestAlgo(sort_size);
    switch (best_algo) {
      case SortAlgo::WARP_BITONIC:
        step.algorithm = ReductionAlgorithm::WARP_BITONIC;
        break;
      case SortAlgo::CUB_WARP_MERGE:
        step.algorithm = ReductionAlgorithm::WARP_MERGE_SORT;
        break;
      case SortAlgo::CUB_BLOCK_MERGE:
        step.algorithm = ReductionAlgorithm::CUB_BLOCK_MERGE;
        break;
      case SortAlgo::CUB_BLOCK_RADIX:
        step.algorithm = ReductionAlgorithm::CUB_RADIX_SORT;
        break;
      default:
        // Fallback to a robust choice
        step.algorithm = ReductionAlgorithm::CUB_BLOCK_MERGE;
        break;
    }

    plan.push_back(step);
    current_partitions = CeilDiv(current_partitions, step.partitions_per_block);
  }

  return plan;
}

// --- Stage 1 Kernel ---
// Finds the Top-K within each partition of the vocabulary.
template <int kBlockSize, int kPartitionSize, int K, bool UseMergeSort>
__global__ void Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                          int* __restrict__ intermediate_indices,
                                          float* __restrict__ intermediate_scores,
                                          int vocab_size, int num_partitions) {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  __shared__ Stage1TempStorageType smem;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K, UseMergeSort>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

// A simple greater-than comparator for descending sort
struct DescendingOp {
  template <typename T>
  __device__ __host__ bool operator()(const T& a, const T& b) const {
    return a > b;
  }
};

/**
 * @brief The unified reduction kernel. This single kernel can perform any of the reduction
 * algorithms by using compile-time template parameters. `if constexpr` is used to ensure
 * only the code for the selected algorithm is instantiated.
 */
template <int kBlockSize, int K_PADDED, int PartitionsPerBlock, ReductionAlgorithm Algorithm>
__global__ void BlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                float* __restrict__ scores_out, int* __restrict__ indices_out, int num_partitions_in) {
  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  if (block_start_partition >= num_partitions_in) {
    return;
  }
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);
  const int num_elements_to_sort = K_PADDED * num_partitions_to_process;

  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions_in * K_PADDED;
  const size_t out_base_offset = (static_cast<size_t>(batch_idx) * gridDim.x + blockIdx.x) * K_PADDED;

  constexpr int kSortSize = K_PADDED * PartitionsPerBlock;

  if constexpr (Algorithm == ReductionAlgorithm::CUB_RADIX_SORT) {
#ifdef STABLE_TOPK
    using SortKeyT = uint64_t;
    using CubTempStorage = typename cub::BlockRadixSort<SortKeyT, kBlockSize, CeilDiv(kSortSize, kBlockSize)>::TempStorage;
#else
    using SortKeyT = float;
    using SortValueT = int;
    using CubTempStorage = typename cub::BlockRadixSort<SortKeyT, kBlockSize, CeilDiv(kSortSize, kBlockSize), SortValueT>::TempStorage;
#endif
    __shared__ CubTempStorage cub_radix_storage;

    constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
#ifdef STABLE_TOPK
    SortKeyT thread_keys[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = topk_common::PackStableSortKey(scores_in[offset], indices_in[offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread>(cub_radix_storage).SortDescendingBlockedToStriped(thread_keys);
    if (threadIdx.x < K_PADDED) {
      scores_out[out_base_offset + threadIdx.x] = topk_common::UnpackStableSortScore(thread_keys[0]);
      indices_out[out_base_offset + threadIdx.x] = topk_common::UnpackStableSortIndex(thread_keys[0]);
    }
#else
    SortKeyT thread_keys[kItemsPerThread];
    SortValueT thread_values[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = scores_in[offset];
        thread_values[i] = indices_in[offset];
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = INT_MAX;
      }
    }
    cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(cub_radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      scores_out[out_base_offset + threadIdx.x] = thread_keys[0];
      indices_out[out_base_offset + threadIdx.x] = thread_values[0];
    }
#endif
  } else if constexpr (Algorithm == ReductionAlgorithm::CUB_BLOCK_MERGE) {
#ifdef STABLE_TOPK
    using SortKeyT = uint64_t;
    using SortValueT = cub::NullType;
    using CubTempStorage = typename cub::BlockMergeSort<SortKeyT, kBlockSize, CeilDiv(kSortSize, kBlockSize), SortValueT>::TempStorage;
    __shared__ CubTempStorage cub_merge_storage;
    constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
    SortKeyT thread_keys[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; i++) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = topk_common::PackStableSortKey(scores_in[offset], indices_in[offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(cub_merge_storage).Sort(thread_keys, DescendingOp());

    float thread_scores_out[kItemsPerThread];
    int thread_indices_out[kItemsPerThread];
    for (int i = 0; i < kItemsPerThread; ++i) {
      thread_scores_out[i] = topk_common::UnpackStableSortScore(thread_keys[i]);
      thread_indices_out[i] = topk_common::UnpackStableSortIndex(thread_keys[i]);
    }
    cub::StoreDirectBlocked(threadIdx.x, scores_out + out_base_offset, thread_scores_out, K_PADDED);
    cub::StoreDirectBlocked(threadIdx.x, indices_out + out_base_offset, thread_indices_out, K_PADDED);
#else
    using SortKeyT = float;
    using SortValueT = int;
    using CubTempStorage = typename cub::BlockMergeSort<SortKeyT, kBlockSize, CeilDiv(kSortSize, kBlockSize), SortValueT>::TempStorage;
    __shared__ CubTempStorage cub_merge_storage;
    constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
    SortKeyT thread_keys[kItemsPerThread];
    SortValueT thread_values[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = scores_in[offset];
        thread_values[i] = indices_in[offset];
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = INT_MAX;
      }
    }
    cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(cub_merge_storage).Sort(thread_keys, thread_values, DescendingOp());
    cub::StoreDirectBlocked(threadIdx.x, scores_out + out_base_offset, thread_keys, K_PADDED);
    cub::StoreDirectBlocked(threadIdx.x, indices_out + out_base_offset, thread_values, K_PADDED);
#endif
  } else {
    constexpr int kSortSizePo2 = topk_common::NextPowerOfTwo(kSortSize);

    union SharedStorage {
      struct {
        __align__(128) float scores[kSortSizePo2];
        __align__(128) int indices[kSortSizePo2];
      } smem_sort;
      typename cub::WarpMergeSort<uint64_t, (kSortSizePo2 + 31) / 32, 32>::TempStorage cub_warp_storage;
    };
    __shared__ SharedStorage smem_union;
    auto& smem_sort = smem_union.smem_sort;

    for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
      if (i < num_elements_to_sort) {
        int partition_idx = i / K_PADDED;
        int element_idx = i % K_PADDED;
        size_t global_offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        smem_sort.scores[i] = scores_in[global_offset];
        smem_sort.indices[i] = indices_in[global_offset];
      } else {
        smem_sort.scores[i] = -FLT_MAX;
        smem_sort.indices[i] = INT_MAX;
      }
    }

    if constexpr (Algorithm == ReductionAlgorithm::WARP_BITONIC) {
      for (int i = kSortSize + threadIdx.x; i < kSortSizePo2; i += kBlockSize) {
        smem_sort.scores[i] = -FLT_MAX;
        smem_sort.indices[i] = INT_MAX;
      }
    }
    __syncthreads();

    if constexpr (Algorithm == ReductionAlgorithm::WARP_BITONIC) {
      if (threadIdx.x < warpSize) {
        float my_score = (threadIdx.x < kSortSize) ? smem_sort.scores[threadIdx.x] : -FLT_MAX;
        int my_index = (threadIdx.x < kSortSize) ? smem_sort.indices[threadIdx.x] : INT_MAX;
        topk_common::WarpBitonicSort(my_score, my_index);
        if (threadIdx.x < K_PADDED) {
          smem_sort.scores[threadIdx.x] = my_score;
          smem_sort.indices[threadIdx.x] = my_index;
        }
      }
    } else if constexpr (Algorithm == ReductionAlgorithm::WARP_MERGE_SORT) {
      topk_common::WarpMergeSort<kSortSizePo2>(smem_sort.scores, smem_sort.indices, &smem_union.cub_warp_storage, num_elements_to_sort);
    }

    __syncthreads();

    if (threadIdx.x < K_PADDED) {
      indices_out[out_base_offset + threadIdx.x] = smem_sort.indices[threadIdx.x];
      scores_out[out_base_offset + threadIdx.x] = smem_sort.scores[threadIdx.x];
    }
  }
}

// --- Kernel Launcher ---
// Helper to launch the reduction kernel with the correct template parameters from the plan.
template <int K_PADDED>
void LaunchReductionStep(const ReductionStep& step, cudaStream_t stream,
                         const float* scores_in, const int* indices_in,
                         float* scores_out, int* indices_out,
                         int num_partitions_in, int batch_size) {
  dim3 grid(CeilDiv(num_partitions_in, step.partitions_per_block), batch_size);
  dim3 block(step.block_size);

  auto dispatch_and_launch = [&](auto block_size_const, auto p_per_block_const) {
    constexpr int B_SIZE = block_size_const.value;
    constexpr int P_PER_BLOCK = p_per_block_const.value;
    constexpr int kSortSize = K_PADDED * P_PER_BLOCK;

    switch (step.algorithm) {
      case ReductionAlgorithm::CUB_RADIX_SORT:
        BlockReduceTopK<B_SIZE, K_PADDED, P_PER_BLOCK, ReductionAlgorithm::CUB_RADIX_SORT><<<grid, block, 0, stream>>>(
            scores_in, indices_in, scores_out, indices_out, num_partitions_in);
        break;
      case ReductionAlgorithm::CUB_BLOCK_MERGE:
        BlockReduceTopK<B_SIZE, K_PADDED, P_PER_BLOCK, ReductionAlgorithm::CUB_BLOCK_MERGE><<<grid, block, 0, stream>>>(
            scores_in, indices_in, scores_out, indices_out, num_partitions_in);
        break;
      case ReductionAlgorithm::WARP_MERGE_SORT:
        if constexpr (kSortSize > 32 && kSortSize <= 128) {
          BlockReduceTopK<B_SIZE, K_PADDED, P_PER_BLOCK, ReductionAlgorithm::WARP_MERGE_SORT><<<grid, block, 0, stream>>>(
              scores_in, indices_in, scores_out, indices_out, num_partitions_in);
        }
        break;
      case ReductionAlgorithm::WARP_BITONIC:
        if constexpr (kSortSize <= 32) {
          BlockReduceTopK<B_SIZE, K_PADDED, P_PER_BLOCK, ReductionAlgorithm::WARP_BITONIC><<<grid, block, 0, stream>>>(
              scores_in, indices_in, scores_out, indices_out, num_partitions_in);
        }
        break;
    }
  };

  auto dispatch_partitions = [&](auto block_size_const) {
    switch (step.partitions_per_block) {
      case 16:
        dispatch_and_launch(block_size_const, std::integral_constant<int, 16>());
        break;
      case 8:
        dispatch_and_launch(block_size_const, std::integral_constant<int, 8>());
        break;
      case 4:
        dispatch_and_launch(block_size_const, std::integral_constant<int, 4>());
        break;
      case 2:
        dispatch_and_launch(block_size_const, std::integral_constant<int, 2>());
        break;
      default:
        break;
    }
  };

  if (step.block_size == 128) {
    dispatch_partitions(std::integral_constant<int, 128>());
  } else {
    dispatch_partitions(std::integral_constant<int, 256>());
  }
}

// Helper to launch Stage 1 with the correct K_TEMPLATE compile-time constant.
template <int K_TEMPLATE>
void LaunchStage1(cudaStream_t stream, int partition_size, int num_partitions, int batch_size,
                  const float* scores_in, TopkData* data, int vocab_size) {
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(256);

  const auto& benchmarks = GetSortBenchmarkResults();
  bool use_merge_sort = benchmarks.GetBestAlgo(partition_size) == SortAlgo::CUB_BLOCK_MERGE;

#define LAUNCH_STAGE1_KERNEL(P_SIZE, USE_MERGE)                                                            \
  Stage1_FindPartitionsTopK<256, P_SIZE, K_TEMPLATE, USE_MERGE><<<grid_stage1, block_stage1, 0, stream>>>( \
      scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions)

  if (partition_size == kCandidatePartitionSizes[0]) {
    if (use_merge_sort)
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[0], true);
    else
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[0], false);
  } else if (partition_size == kCandidatePartitionSizes[1]) {
    if (use_merge_sort)
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[1], true);
    else
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[1], false);
  } else if (partition_size == kCandidatePartitionSizes[2]) {
    if (use_merge_sort)
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[2], true);
    else
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[2], false);
  } else if (partition_size == kCandidatePartitionSizes[3]) {
    if (use_merge_sort)
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[3], true);
    else
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[3], false);
  } else {
    // This path should not be taken due to the checks in IsSupported.
    assert(false);
  }

#undef LAUNCH_STAGE1_KERNEL
}

// --- Main Entry Point ---

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));  // caller shall check IsSupported before calling this function.
  if (data->hybrid_sort_partition_size == 0) {
    data->hybrid_sort_partition_size = EstimateBestPartitionSize(vocab_size);
  }

  // --- Stage 0: Planning ---
  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // Pad k to a supported value for template instantiation.
  int k_padded;
  if (k <= 4)
    k_padded = 4;
  else if (k <= 8)
    k_padded = 8;
  else if (k <= 16)
    k_padded = 16;
  else if (k <= 32)
    k_padded = 32;
  else if (k <= 64)
    k_padded = 64;
  else if (k <= 128)
    k_padded = 128;
  else
    k_padded = 256;

  // Create the reduction plan based on the padded k value.
  const ReductionPlan plan = GetReductionPlan(num_partitions, k_padded, stream);

  // This lambda captures the launch logic to be called by the k-dispatch below.
  auto launch_kernels = [&](auto k_padded_const) {
    constexpr int K_PADDED = k_padded_const.value;
    // --- Stage 1: Find Partition Top-K ---
    LaunchStage1<K_PADDED>(stream, partition_size, num_partitions, batch_size, scores_in, data, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    // --- Stage 2: Planned Reduction ---
    if (num_partitions > 1 && !plan.empty()) {
      int current_num_partitions = num_partitions;
      // Setup ping-pong pointers for reduction inputs/outputs.
      float* scores_in_ptr = data->intermediate_scores_1;
      int* indices_in_ptr = data->intermediate_indices_1;
      float* scores_out_ptr = data->intermediate_scores_2;
      int* indices_out_ptr = data->intermediate_indices_2;

      // Execute each step in the reduction plan.
      for (const auto& step : plan) {
        LaunchReductionStep<K_PADDED>(step, stream, scores_in_ptr, indices_in_ptr, scores_out_ptr, indices_out_ptr, current_num_partitions, batch_size);
        CUDA_CHECK(cudaGetLastError());
        current_num_partitions = CeilDiv(current_num_partitions, step.partitions_per_block);
        std::swap(scores_in_ptr, scores_out_ptr);
        std::swap(indices_in_ptr, indices_out_ptr);
      }
      // The final result is in the last "in" pointer after the swaps.
      data->topk_scores = scores_in_ptr;
      data->topk_indices = indices_in_ptr;
    } else {
      // No reduction was needed.
      data->topk_scores = data->intermediate_scores_1;
      data->topk_indices = data->intermediate_indices_1;
    }
    data->topk_stride = K_PADDED;
  };

  if (k_padded == 4)
    launch_kernels(std::integral_constant<int, 4>());
  else if (k_padded == 8)
    launch_kernels(std::integral_constant<int, 8>());
  else if (k_padded == 16)
    launch_kernels(std::integral_constant<int, 16>());
  else if (k_padded == 32)
    launch_kernels(std::integral_constant<int, 32>());
  else if (k_padded == 64)
    launch_kernels(std::integral_constant<int, 64>());
  else if (k_padded == 128)
    launch_kernels(std::integral_constant<int, 128>());
  else if (k_padded == 256)
    launch_kernels(std::integral_constant<int, 256>());
}

bool IsSupported(int /*batch_size*/, int vocab_size, int k) {
  if (k > kHybridSortMaxK) {
    return false;
  }

  const int partition_size = EstimateBestPartitionSize(vocab_size);
  // Ensure the estimated partition size is one of the valid, non-zero candidates
  // before proceeding. This prevents invalid template instantiations.
  bool is_valid_psize = false;
  for (int p_size : kCandidatePartitionSizes) {
    if (partition_size == p_size) {
      is_valid_psize = true;
      break;
    }
  }
  if (!is_valid_psize) {
    return false;
  }

  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) {
    return false;
  }

  return true;
}

}  // namespace hybrid_sort
}  // namespace cuda
}  // namespace Generators
