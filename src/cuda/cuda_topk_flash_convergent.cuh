// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_warp_sort_helper.cuh"
#include <cooperative_groups.h>
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace flash_convergent {

/**
 * @brief A two-stage cooperative algorithm with a single-step reduction phase.
 *
 * Algorithm Overview:
 * This algorithm uses cooperative groups to perform the entire Top-K operation in
 * a single kernel launch, but with a different reduction strategy than `iterative_sort`.
 *
 * 1.  **Stage 1 (Partition Top-K)**: All thread blocks work in parallel to find the
 * top `K_PADDED` candidates from their assigned vocabulary partitions using
 * `topk_common::FindPartitionTopK`. The results are written to a global
 * intermediate buffer.
 *
 * 2.  **Grid-Wide Sync**: A `cg::grid_group::sync()` ensures all partitions are processed.
 *
 * 3.  **Stage 2 (Single-Step Reduction)**: A single, specialized thread block (`blockIdx.x == 0`)
 * is responsible for the final merge. It loads all candidates from the intermediate
 * buffer and performs a final, large sort to find the global Top-K.
 *
 * Performance Characteristics:
 * -   **Strengths**: By performing the final reduction in a single step, it avoids the overhead of
 * iterative loops and multiple grid-wide synchronizations found in other cooperative methods.
 * It intelligently switches its internal sorting method based on the total number of candidates,
 * leveraging benchmark data to choose the fastest algorithm.
 * -   **Weaknesses**: Requires cooperative launch support. Its primary limitation is the total
 * number of partitions (`kMaxPartitions`), as a single block must be able to load and sort
 * all candidates. This makes it less scalable for extremely large vocabularies that would
 * require many partitions.
 */
namespace cg = cooperative_groups;

// The limit on partitions is due to cooperative group residency requirements and the
// fact that a single block must sort all `k * num_partitions` candidates in Stage 2.
constexpr int kMaxPartitions = 64;

// This partition sizes select as {11, 13, 16, 19} * 256.
constexpr std::array<int, 4> kPartitionSizes = {2816, 3328, 4096, 4864};

// The internal sorting algorithm to be used inside the reduction stage, chosen by the host-side launcher.
enum class ReductionAlgorithm {
  WARP_MERGE_SORT,
  CUB_BLOCK_MERGE,
  CUB_RADIX_SORT
};

// --- Metaprogramming to select the correct SharedStorage type ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel, ReductionAlgorithm Algorithm>
struct SharedStorageSelector;

// Specialization for WARP_MERGE_SORT
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
struct SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::WARP_MERGE_SORT> {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  static constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  static constexpr int kSortSizePo2 = topk_common::NextPowerOfTwo(kSortSize);

  union type {
    Stage1TempStorageType stage1_storage;
    struct {
      float scores[kSortSizePo2];
      int indices[kSortSizePo2];
    } warp_sort_storage;
    typename cub::WarpMergeSort<uint64_t, (kSortSizePo2 + 31) / 32, 32>::TempStorage cub_warp_storage;
  };
};

// Specialization for CUB_BLOCK_MERGE
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
struct SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::CUB_BLOCK_MERGE> {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  static constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  static constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  union type {
    Stage1TempStorageType stage1_storage;
#ifdef STABLE_TOPK
    typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage merge_storage;
#else
    typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage merge_storage;
#endif
  };
};

// Specialization for CUB_RADIX_SORT
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
struct SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::CUB_RADIX_SORT> {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  static constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  static constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  union type {
    Stage1TempStorageType stage1_storage;
#ifdef STABLE_TOPK
    typename cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThread>::TempStorage radix_storage;
#else
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>::TempStorage radix_storage;
#endif
  };
};

// --- Unified Convergent Kernel ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel, ReductionAlgorithm Algorithm, bool UseMergeSortForStage1>
__global__ void FlashConvergentKernel(const float* __restrict__ scores_in,
                                      float* __restrict__ intermediate_scores,
                                      int* __restrict__ intermediate_indices,
                                      float* __restrict__ scores_out,
                                      int* __restrict__ indices_out,
                                      int vocab_size,
                                      int num_partitions,
                                      int k_actual) {
  cg::grid_group grid = cg::this_grid();
  constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);

  using SharedStorage = typename SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, Algorithm>::type;
  __shared__ SharedStorage smem;

  // --- Stage 1: Parallel Partition Sort ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, UseMergeSortForStage1>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2: One block performs the final merge ---
  if (blockIdx.x == 0) {
    const int batch_idx = blockIdx.y;
    const int num_elements_to_sort = num_partitions * K_PADDED;

    if constexpr (Algorithm == ReductionAlgorithm::WARP_MERGE_SORT) {
      constexpr int kSortSizePo2 = topk_common::NextPowerOfTwo(kSortSize);
      for (int i = threadIdx.x; i < kSortSizePo2; i += kBlockSize) {
        if (i < num_elements_to_sort) {
          smem.warp_sort_storage.scores[i] = intermediate_scores[(size_t)batch_idx * num_partitions * K_PADDED + i];
          smem.warp_sort_storage.indices[i] = intermediate_indices[(size_t)batch_idx * num_partitions * K_PADDED + i];
        } else {
          smem.warp_sort_storage.scores[i] = -FLT_MAX;
          smem.warp_sort_storage.indices[i] = INT_MAX;
        }
      }
      __syncthreads();

      topk_common::WarpMergeSort<kSortSizePo2>(smem.warp_sort_storage.scores, smem.warp_sort_storage.indices, &smem.cub_warp_storage, num_elements_to_sort);
      __syncthreads();

      if (threadIdx.x < k_actual) {
        size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
        scores_out[out_offset] = smem.warp_sort_storage.scores[threadIdx.x];
        indices_out[out_offset] = smem.warp_sort_storage.indices[threadIdx.x];
      }
    } else {  // Handle CUB Block-level Sorts
#ifdef STABLE_TOPK
      using SortKeyT = uint64_t;
      SortKeyT thread_keys[kItemsPerThread];
      for (int i = 0; i < kItemsPerThread; ++i) {
        int load_idx = threadIdx.x * kItemsPerThread + i;  // Blocked load
        if (load_idx < num_elements_to_sort) {
          size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + load_idx;
          thread_keys[i] = topk_common::PackStableSortKey(intermediate_scores[offset], intermediate_indices[offset]);
        } else {
          thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
        }
      }
      if constexpr (Algorithm == ReductionAlgorithm::CUB_RADIX_SORT) {
        cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread>(smem.radix_storage).SortDescendingBlockedToStriped(thread_keys);
        if (threadIdx.x < k_actual) {
          size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
          scores_out[out_offset] = topk_common::UnpackStableSortScore(thread_keys[0]);
          indices_out[out_offset] = topk_common::UnpackStableSortIndex(thread_keys[0]);
        }
      } else {  // CUB_BLOCK_MERGE
        cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, cub::NullType>(smem.merge_storage).Sort(thread_keys, topk_common::DescendingOp());
        float thread_scores_out[kItemsPerThread];
        int thread_indices_out[kItemsPerThread];
        for (int i = 0; i < kItemsPerThread; ++i) {
          thread_scores_out[i] = topk_common::UnpackStableSortScore(thread_keys[i]);
          thread_indices_out[i] = topk_common::UnpackStableSortIndex(thread_keys[i]);
        }
        cub::StoreDirectBlocked(threadIdx.x, scores_out + (size_t)batch_idx * k_actual, thread_scores_out, k_actual);
        cub::StoreDirectBlocked(threadIdx.x, indices_out + (size_t)batch_idx * k_actual, thread_indices_out, k_actual);
      }
#else
      using SortKeyT = float;
      using SortValueT = int;
      SortKeyT thread_keys[kItemsPerThread];
      SortValueT thread_values[kItemsPerThread];
      for (int i = 0; i < kItemsPerThread; ++i) {
        int load_idx = threadIdx.x * kItemsPerThread + i;  // Blocked load
        if (load_idx < num_elements_to_sort) {
          size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + load_idx;
          thread_keys[i] = intermediate_scores[offset];
          thread_values[i] = intermediate_indices[offset];
        } else {
          thread_keys[i] = -FLT_MAX;
          thread_values[i] = INT_MAX;
        }
      }
      if constexpr (Algorithm == ReductionAlgorithm::CUB_RADIX_SORT) {
        cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
        if (threadIdx.x < k_actual) {
          size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
          scores_out[out_offset] = thread_keys[0];
          indices_out[out_offset] = thread_values[0];
        }
      } else {  // CUB_BLOCK_MERGE
        cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.merge_storage).Sort(thread_keys, thread_values, topk_common::DescendingOp());
        cub::StoreDirectBlocked(threadIdx.x, scores_out + (size_t)batch_idx * k_actual, thread_keys, k_actual);
        cub::StoreDirectBlocked(threadIdx.x, indices_out + (size_t)batch_idx * k_actual, thread_values, k_actual);
      }
#endif
    }
  }
}

// --- Host-side Launcher ---

inline int EstimateBestPartitionSize(int vocab_size, int k) {
  const auto& benchmarks = GetSortBenchmarkResults();

  double min_total_latency = std::numeric_limits<double>::max();
  int best_partition_size = 0;

  for (int p_size : kPartitionSizes) {
    const int num_partitions = CeilDiv(vocab_size, p_size);
    if (num_partitions > kMaxPartitions) {
      continue;
    }

    // Estimate Stage 1 latency
    SortAlgo best_algo_s1 = benchmarks.GetBestAlgo(p_size);
    float latency_s1 = benchmarks.GetLatency(best_algo_s1, p_size);

    // Estimate Stage 2 latency
    int sort_size_s2 = kConvergentSortMaxK * num_partitions;
    SortAlgo best_algo_s2 = benchmarks.GetBestAlgo(sort_size_s2);
    float latency_s2 = benchmarks.GetLatency(best_algo_s2, sort_size_s2);

    float total_latency = latency_s1 + latency_s2;

    if (total_latency < min_total_latency) {
      min_total_latency = total_latency;
      best_partition_size = p_size;
    }
  }

  // Fallback if no suitable size found (e.g., for very small vocab_size)
  if (best_partition_size == 0) {
    best_partition_size = kPartitionSizes[0];
  }

  return best_partition_size;
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kConvergentSortMaxK;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel, bool UseMergeS1>
void* GetKernel() {
  const auto& benchmarks = GetSortBenchmarkResults();
  constexpr int sort_size_s2 = K_PADDED * kMaxPartitionsForKernel;
  SortAlgo best_algo_s2 = benchmarks.GetBestAlgo(sort_size_s2);

  if (best_algo_s2 == SortAlgo::CUB_WARP_MERGE || best_algo_s2 == SortAlgo::WARP_BITONIC) {
    if constexpr (sort_size_s2 <= 256) {
      return (void*)FlashConvergentKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::WARP_MERGE_SORT, UseMergeS1>;
    }
  } else if (best_algo_s2 == SortAlgo::CUB_BLOCK_MERGE) {
    return (void*)FlashConvergentKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::CUB_BLOCK_MERGE, UseMergeS1>;
  }

  // CUB_BLOCK_RADIX
  return (void*)FlashConvergentKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, ReductionAlgorithm::CUB_RADIX_SORT, UseMergeS1>;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, bool UseMergeS1>
void* GetKernelForNumPartitions(int num_partitions) {
  if (num_partitions <= 8) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 8, UseMergeS1>();
  if (num_partitions <= 16) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 16, UseMergeS1>();
  if (num_partitions <= 32) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 32, UseMergeS1>();
  return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 64, UseMergeS1>();
}

template <int P_SIZE, int K_PADDED, bool UseMergeS1>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  int num_partitions = CeilDiv(vocab_size, P_SIZE);

  dim3 grid(num_partitions, batch_size);
  dim3 block(kBlockSize);

  void* kernel = GetKernelForNumPartitions<kBlockSize, P_SIZE, K_PADDED, UseMergeS1>(num_partitions);
  void* kernel_args[] = {(void*)&scores_in, (void*)&data->intermediate_scores_1, (void*)&data->intermediate_indices_1,
                         (void*)&data->intermediate_scores_2, (void*)&data->intermediate_indices_2,
                         (void*)&vocab_size, (void*)&num_partitions, (void*)&k};
  CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, kernel_args, 0, stream));
}

template <int K_PADDED>
void LaunchKernelByPartitionSize(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size) {
  const auto& benchmarks = GetSortBenchmarkResults();
  bool use_merge_s1 = benchmarks.GetBestAlgo(partition_size) == SortAlgo::CUB_BLOCK_MERGE;

#define LAUNCH_KERNEL_WITH_MERGE_FLAG(P_SIZE)                                                    \
  {                                                                                              \
    if (use_merge_s1) {                                                                          \
      LaunchKernel<P_SIZE, K_PADDED, true>(data, stream, scores_in, vocab_size, batch_size, k);  \
    } else {                                                                                     \
      LaunchKernel<P_SIZE, K_PADDED, false>(data, stream, scores_in, vocab_size, batch_size, k); \
    }                                                                                            \
  }

  if (partition_size == kPartitionSizes[0])
    LAUNCH_KERNEL_WITH_MERGE_FLAG(kPartitionSizes[0])
  else if (partition_size == kPartitionSizes[1])
    LAUNCH_KERNEL_WITH_MERGE_FLAG(kPartitionSizes[1])
  else if (partition_size == kPartitionSizes[2])
    LAUNCH_KERNEL_WITH_MERGE_FLAG(kPartitionSizes[2])
  else
    LAUNCH_KERNEL_WITH_MERGE_FLAG(kPartitionSizes[3])
#undef LAUNCH_KERNEL_WITH_MERGE_FLAG
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  if (data->flash_convergent_partition_size_k != k) {
    data->flash_convergent_partition_size_k = k;
    data->flash_convergent_partition_size = EstimateBestPartitionSize(vocab_size, k);
  }

  const int partition_size = data->flash_convergent_partition_size;

  if (k <= 4)
    LaunchKernelByPartitionSize<4>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 8)
    LaunchKernelByPartitionSize<8>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 16)
    LaunchKernelByPartitionSize<16>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 32)
    LaunchKernelByPartitionSize<32>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 52)
    LaunchKernelByPartitionSize<52>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else
    LaunchKernelByPartitionSize<64>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  CUDA_CHECK_LAUNCH();

  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k;
}

// --- The following implements IsSupported ---
template <int K_PADDED, int kMaxPartitionsForKernel>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;

  // When sort size is larger than 1024, CUB's block radix sort is better than block merge sort (See cuda_topk_sort_benchmark_cache.h).
  constexpr bool kUseMergeSortS1 = false;

  void* kernel;
  if (partition_size == kPartitionSizes[0])
    kernel = GetKernel<kBlockSize, kPartitionSizes[0], K_PADDED, kMaxPartitionsForKernel, kUseMergeSortS1>();
  else if (partition_size == kPartitionSizes[1])
    kernel = GetKernel<kBlockSize, kPartitionSizes[1], K_PADDED, kMaxPartitionsForKernel, kUseMergeSortS1>();
  else if (partition_size == kPartitionSizes[2])
    kernel = GetKernel<kBlockSize, kPartitionSizes[2], K_PADDED, kMaxPartitionsForKernel, kUseMergeSortS1>();
  else
    kernel = GetKernel<kBlockSize, kPartitionSizes[3], K_PADDED, kMaxPartitionsForKernel, kUseMergeSortS1>();

  return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

template <int K_PADDED>
bool IsSupportedDispatch(int batch_size, int partition_size, int num_partitions) {
  if (num_partitions <= 8) return CheckSupport<K_PADDED, 8>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 16) return CheckSupport<K_PADDED, 16>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 32) return CheckSupport<K_PADDED, 32>(batch_size, num_partitions, partition_size);
  return CheckSupport<K_PADDED, 64>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kConvergentSortMaxK) return false;

  const int partition_size = EstimateBestPartitionSize(vocab_size, k);
  if (partition_size == 0) return false;
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) return false;

  if (k <= 4) return IsSupportedDispatch<4>(batch_size, partition_size, num_partitions);
  if (k <= 8) return IsSupportedDispatch<8>(batch_size, partition_size, num_partitions);
  if (k <= 16) return IsSupportedDispatch<16>(batch_size, partition_size, num_partitions);
  if (k <= 32) return IsSupportedDispatch<32>(batch_size, partition_size, num_partitions);
  if (k <= 52) return IsSupportedDispatch<52>(batch_size, partition_size, num_partitions);
  return IsSupportedDispatch<64>(batch_size, partition_size, num_partitions);
}

}  // namespace flash_convergent
}  // namespace cuda
}  // namespace Generators
