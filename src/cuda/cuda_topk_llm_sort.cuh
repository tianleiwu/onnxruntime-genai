// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <type_traits>  // For std::integral_constant
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace llm_sort {

namespace cg = cooperative_groups;

// A helper struct to hold the reduction factors.
struct ReductionFactors {
  int factor1 = 1;
  int factor2 = 1;
  int factor3 = 1;
  int num_reduction_steps = 0;
};

// Computes the optimal reduction factors based on partition count and k.
// For large k, it will favor a 3-step reduction with smaller factors.
constexpr ReductionFactors GetReductionFactors(int num_partitions, int k) {
  constexpr int k_large_threshold = 48;

  // For large k, use a 3-step strategy with smaller, more efficient factors.
  if (k > k_large_threshold) {
    if (num_partitions > 32) {  // 33-64 partitions
      return {4, 4, 4, 3};
    }
    if (num_partitions > 16) {  // 17-32 partitions
      return {4, 4, 2, 3};
    }
  }

  // Otherwise, use the original high-performance 1 or 2-step logic for smaller k.
  if (num_partitions <= 1) {
    return {1, 1, 1, 0};
  }
  if (num_partitions <= 8) {
    int f1 = (num_partitions <= 2) ? 2 : ((num_partitions <= 4) ? 4 : 8);
    return {f1, 1, 1, 1};
  }
  if (num_partitions <= 16) {
    return {4, 4, 1, 2};
  }
  if (num_partitions <= 32) {
    return {8, 4, 1, 2};
  }
  // 33-64 partitions
  return {8, 8, 1, 2};
}

// Multi-Step Reduction kernel supporting up to 3 reduction steps.
template <int K_PADDED, int kBlockSize, int kPartitionSize, int Factor1, int Factor2, int Factor3>
__global__ void LlmSortKernel(const float* __restrict__ input_scores,
                              int* __restrict__ intermediate_indices_1,
                              float* __restrict__ intermediate_scores_1,
                              int* __restrict__ intermediate_indices_2,
                              float* __restrict__ intermediate_scores_2,
                              int vocab_size) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  // Use AoS for small K, SoA for large K. Threshold of 32 is a common heuristic.
  constexpr bool kUseAoS = (K_PADDED <= 32);

  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;

  // --- Shared Memory Union ---
  constexpr int kSortSize1 = K_PADDED * Factor1;
  constexpr int kSortSize2 = K_PADDED * Factor2;
  constexpr int kSortSize3 = K_PADDED * Factor3;

  union SharedStorage {
    typename BlockRadixSort::TempStorage stage1_storage;
    struct {
      __align__(128) float scores[kSortSize1];
      __align__(128) int indices[kSortSize1];
    } step1_storage_SoA;
    struct {
      __align__(128) bitonic_sort::KeyValue data[kSortSize1];
    } step1_storage_AoS;
    struct {
      __align__(128) float scores[kSortSize2];
      __align__(128) int indices[kSortSize2];
    } step2_storage_SoA;
    struct {
      __align__(128) bitonic_sort::KeyValue data[kSortSize2];
    } step2_storage_AoS;
    struct {
      __align__(128) float scores[kSortSize3];
      __align__(128) int indices[kSortSize3];
    } step3_storage_SoA;
    struct {
      __align__(128) bitonic_sort::KeyValue data[kSortSize3];
    } step3_storage_AoS;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Find Top-K within each partition ---
  {
    const float* batch_input_scores = input_scores + static_cast<size_t>(batch_idx) * vocab_size;
    const size_t batch_intermediate_offset_stage1 = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
    int* batch_intermediate_indices_1 = intermediate_indices_1 + batch_intermediate_offset_stage1;
    float* batch_intermediate_scores_1 = intermediate_scores_1 + batch_intermediate_offset_stage1;

    const int partition_start = partition_idx * kPartitionSize;
    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i) {
      int global_idx = partition_start + threadIdx.x + i * kBlockSize;
      if (global_idx < vocab_size) {
        thread_keys[i] = batch_input_scores[global_idx];
        thread_values[i] = global_idx;
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = -1;
      }
    }
    BlockRadixSort(smem.stage1_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      batch_intermediate_scores_1[offset] = thread_keys[0];
      batch_intermediate_indices_1[offset] = thread_values[0];
    }
  }
  grid.sync();

  // --- Stage 2, Step 1: First Reduction ---
  int partitions_after_step1 = num_partitions;
  if (Factor1 > 1) {
    partitions_after_step1 = CeilDiv(num_partitions, Factor1);
    if (partition_idx < partitions_after_step1) {
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;

      int first_child = partition_idx * Factor1;
      int num_to_process = min(Factor1, num_partitions - first_child);

      if constexpr (kUseAoS) {
        for (int i = threadIdx.x; i < kSortSize1; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step1_storage_AoS.data[i].score = scores_in_batch[offset];
            smem.step1_storage_AoS.data[i].index = indices_in_batch[offset];
          } else {
            smem.step1_storage_AoS.data[i].score = -FLT_MAX;
            smem.step1_storage_AoS.data[i].index = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_AoS<kBlockSize, kSortSize1>(smem.step1_storage_AoS.data);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step1_storage_AoS.data[threadIdx.x].score;
          indices_out_batch[out_offset] = smem.step1_storage_AoS.data[threadIdx.x].index;
        }
      } else {  // Use SoA
        for (int i = threadIdx.x; i < kSortSize1; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step1_storage_SoA.scores[i] = scores_in_batch[offset];
            smem.step1_storage_SoA.indices[i] = indices_in_batch[offset];
          } else {
            smem.step1_storage_SoA.scores[i] = -FLT_MAX;
            smem.step1_storage_SoA.indices[i] = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize1>(smem.step1_storage_SoA.scores, smem.step1_storage_SoA.indices);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step1_storage_SoA.scores[threadIdx.x];
          indices_out_batch[out_offset] = smem.step1_storage_SoA.indices[threadIdx.x];
        }
      }
    }
    grid.sync();
  }

  // --- Stage 2, Step 2: Second Reduction ---
  int partitions_after_step2 = partitions_after_step1;
  if (Factor2 > 1) {
    partitions_after_step2 = CeilDiv(partitions_after_step1, Factor2);
    if (partition_idx < partitions_after_step2) {
      const float* scores_in_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      float* scores_out_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      int* indices_out_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;

      int first_child = partition_idx * Factor2;
      int num_to_process = min(Factor2, partitions_after_step1 - first_child);

      if constexpr (kUseAoS) {
        for (int i = threadIdx.x; i < kSortSize2; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step2_storage_AoS.data[i].score = scores_in_batch[offset];
            smem.step2_storage_AoS.data[i].index = indices_in_batch[offset];
          } else {
            smem.step2_storage_AoS.data[i].score = -FLT_MAX;
            smem.step2_storage_AoS.data[i].index = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_AoS<kBlockSize, kSortSize2>(smem.step2_storage_AoS.data);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step2_storage_AoS.data[threadIdx.x].score;
          indices_out_batch[out_offset] = smem.step2_storage_AoS.data[threadIdx.x].index;
        }
      } else {  // Use SoA
        for (int i = threadIdx.x; i < kSortSize2; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step2_storage_SoA.scores[i] = scores_in_batch[offset];
            smem.step2_storage_SoA.indices[i] = indices_in_batch[offset];
          } else {
            smem.step2_storage_SoA.scores[i] = -FLT_MAX;
            smem.step2_storage_SoA.indices[i] = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize2>(smem.step2_storage_SoA.scores, smem.step2_storage_SoA.indices);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step2_storage_SoA.scores[threadIdx.x];
          indices_out_batch[out_offset] = smem.step2_storage_SoA.indices[threadIdx.x];
        }
      }
    }
    grid.sync();
  }

  // --- Stage 2, Step 3: Third Reduction ---
  if (Factor3 > 1) {
    int partitions_after_step3 = CeilDiv(partitions_after_step2, Factor3);
    if (partition_idx < partitions_after_step3) {
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;

      int first_child = partition_idx * Factor3;
      int num_to_process = min(Factor3, partitions_after_step2 - first_child);

      if constexpr (kUseAoS) {
        for (int i = threadIdx.x; i < kSortSize3; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step3_storage_AoS.data[i].score = scores_in_batch[offset];
            smem.step3_storage_AoS.data[i].index = indices_in_batch[offset];
          } else {
            smem.step3_storage_AoS.data[i].score = -FLT_MAX;
            smem.step3_storage_AoS.data[i].index = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_AoS<kBlockSize, kSortSize3>(smem.step3_storage_AoS.data);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step3_storage_AoS.data[threadIdx.x].score;
          indices_out_batch[out_offset] = smem.step3_storage_AoS.data[threadIdx.x].index;
        }
      } else {  // Use SoA
        for (int i = threadIdx.x; i < kSortSize3; i += kBlockSize) {
          if (i < K_PADDED * num_to_process) {
            size_t offset = (first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED);
            smem.step3_storage_SoA.scores[i] = scores_in_batch[offset];
            smem.step3_storage_SoA.indices[i] = indices_in_batch[offset];
          } else {
            smem.step3_storage_SoA.scores[i] = -FLT_MAX;
            smem.step3_storage_SoA.indices[i] = -1;
          }
        }
        __syncthreads();
        bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize3>(smem.step3_storage_SoA.scores, smem.step3_storage_SoA.indices);
        if (threadIdx.x < K_PADDED) {
          size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
          scores_out_batch[out_offset] = smem.step3_storage_SoA.scores[threadIdx.x];
          indices_out_batch[out_offset] = smem.step3_storage_SoA.indices[threadIdx.x];
        }
      }
    }
    // No grid.sync() needed after the final step.
  }
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kFlashSortMaxK;
}

// Parition sizes are optimized for common vocab_size (padded to multiple of 256) used in open source LLM:
//    32256, 32512, 128256, 128512, 152064, 152320, 200192, 200448, 201216, 201472, 262400, 262656.
// Constraints: partition_size are multiple of 256, partition_size <= 8192.
// Goal: mimize average waste ratio to get total partitions be one of 2, 4, 8, 16, 32 and 64.
// For example, vocab_size=201088, ideal partition size is 3142 for 64 partitions. The waste ratio is 1 - 3142/3328 = 0.055.
// The maximum vocab_size that this kernel can support is decided by below choices (i.e. 4864 * 64 = 311296).
constexpr std::array<int, 4> kAllowedPartitionSizes = {2048, 3328, 4352, 4864};

constexpr std::array<int, 7> kTargetPartitionCounts = {1, 2, 4, 8, 16, 32, 64};

inline int EstimateBestPartitionSize(int vocab_size) {
  double min_cost_ratio = std::numeric_limits<double>::infinity();
  int best_partition_size = 0;

  for (int partition_size : kAllowedPartitionSizes) {
    int partitions_needed = CeilDiv(vocab_size, partition_size);

    if (partitions_needed <= 64) {  // Max target count constraint
      // Find smallest target count >= partitions_needed
      auto target_it = std::lower_bound(kTargetPartitionCounts.begin(), kTargetPartitionCounts.end(), partitions_needed);
      if (target_it != kTargetPartitionCounts.end()) {
        int target = *target_it;
        double cost_ratio = static_cast<double>(partition_size * target - vocab_size) / vocab_size;
        if (cost_ratio < min_cost_ratio) {
          min_cost_ratio = cost_ratio;
          best_partition_size = partition_size;
        }
      }
    }
  }
  return best_partition_size;
}

// --- Unified Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  constexpr int kBlockSize = 256;
  const int partition_size = data->llm_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // --- Host-side lookup for reduction factors, now k-aware ---
  const auto factors = GetReductionFactors(num_partitions, k);
  const int factor1 = factors.factor1;
  const int factor2 = factors.factor2;
  const int factor3 = factors.factor3;
  const int num_reduction_steps = factors.num_reduction_steps;

  // Determine final output buffer based on number of steps
  if (num_reduction_steps % 2 == 1) {  // 1 or 3 steps
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }

  int num_partitions_out = num_partitions;
  if (num_reduction_steps > 0) num_partitions_out = CeilDiv(num_partitions, factor1);
  if (num_reduction_steps > 1) num_partitions_out = CeilDiv(num_partitions_out, factor2);
  if (num_reduction_steps > 2) num_partitions_out = CeilDiv(num_partitions_out, factor3);

  int k_padded_val;
  if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 56)
    k_padded_val = 56;
  else if (k <= 64)
    k_padded_val = 64;
  else
    k_padded_val = kFlashSortMaxK;
  data->topk_stride = k_padded_val * num_partitions_out;

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  auto launch_kernel = [&](auto k_padded, auto f1, auto f2, auto f3) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int F1 = decltype(f1)::value;
    constexpr int F2 = decltype(f2)::value;
    constexpr int F3 = decltype(f3)::value;
    dim3 grid(num_partitions, batch_size);
    dim3 block(kBlockSize);
    switch (partition_size) {
      case kAllowedPartitionSizes[0]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], F1, F2, F3>, grid, block, kernel_args, 0, stream));
        break;
      case kAllowedPartitionSizes[1]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], F1, F2, F3>, grid, block, kernel_args, 0, stream));
        break;
      case kAllowedPartitionSizes[2]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], F1, F2, F3>, grid, block, kernel_args, 0, stream));
        break;
      default:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], F1, F2, F3>, grid, block, kernel_args, 0, stream));
        break;
    }
  };

  auto dispatch_f3 = [&](auto k_padded, auto f1, auto f2) {
    if (factor3 == 1) launch_kernel(k_padded, f1, f2, std::integral_constant<int, 1>());
    if (factor3 == 2) launch_kernel(k_padded, f1, f2, std::integral_constant<int, 2>());
    if (factor3 == 4) launch_kernel(k_padded, f1, f2, std::integral_constant<int, 4>());
  };
  auto dispatch_f2 = [&](auto k_padded, auto f1) {
    if (factor2 == 1) dispatch_f3(k_padded, f1, std::integral_constant<int, 1>());
    if (factor2 == 4) dispatch_f3(k_padded, f1, std::integral_constant<int, 4>());
    if (factor2 == 8) dispatch_f3(k_padded, f1, std::integral_constant<int, 8>());
  };
  auto dispatch_f1 = [&](auto k_padded) {
    if (factor1 == 1) dispatch_f2(k_padded, std::integral_constant<int, 1>());
    if (factor1 == 2) dispatch_f2(k_padded, std::integral_constant<int, 2>());
    if (factor1 == 4) dispatch_f2(k_padded, std::integral_constant<int, 4>());
    if (factor1 == 8) dispatch_f2(k_padded, std::integral_constant<int, 8>());
  };

  if (k_padded_val == 4)
    dispatch_f1(std::integral_constant<int, 4>());
  else if (k_padded_val == 8)
    dispatch_f1(std::integral_constant<int, 8>());
  else if (k_padded_val == 16)
    dispatch_f1(std::integral_constant<int, 16>());
  else if (k_padded_val == 32)
    dispatch_f1(std::integral_constant<int, 32>());
  else if (k_padded_val == 56)
    dispatch_f1(std::integral_constant<int, 56>());
  else if (k_padded_val == 64)
    dispatch_f1(std::integral_constant<int, 64>());
  else
    dispatch_f1(std::integral_constant<int, kFlashSortMaxK>());

  CUDA_CHECK_LAUNCH();
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kFlashSortMaxK) {
    return false;
  }
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  if (partition_size == 0) {
    return false;
  }
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > 64) {
    return false;
  }

  int cooperative_launch_support = 0;
  cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, 0);
  if (!cooperative_launch_support) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;

  const auto factors = GetReductionFactors(num_partitions, k);
  const int factor1 = factors.factor1;
  const int factor2 = factors.factor2;
  const int factor3 = factors.factor3;

  void* kernel = nullptr;
  auto set_kernel_ptr = [&](auto k_padded, auto f1, auto f2, auto f3) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int F1 = decltype(f1)::value;
    constexpr int F2 = decltype(f2)::value;
    constexpr int F3 = decltype(f3)::value;
    switch (partition_size) {
      case kAllowedPartitionSizes[0]:
        kernel = (void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], F1, F2, F3>;
        break;
      case kAllowedPartitionSizes[1]:
        kernel = (void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], F1, F2, F3>;
        break;
      case kAllowedPartitionSizes[2]:
        kernel = (void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], F1, F2, F3>;
        break;
      default:
        kernel = (void*)LlmSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], F1, F2, F3>;
        break;
    }
  };

  auto dispatch_f3 = [&](auto k_padded, auto f1, auto f2) {
    if (factor3 == 1) set_kernel_ptr(k_padded, f1, f2, std::integral_constant<int, 1>());
    if (factor3 == 2) set_kernel_ptr(k_padded, f1, f2, std::integral_constant<int, 2>());
    if (factor3 == 4) set_kernel_ptr(k_padded, f1, f2, std::integral_constant<int, 4>());
  };
  auto dispatch_f2 = [&](auto k_padded, auto f1) {
    if (factor2 == 1) dispatch_f3(k_padded, f1, std::integral_constant<int, 1>());
    if (factor2 == 4) dispatch_f3(k_padded, f1, std::integral_constant<int, 4>());
    if (factor2 == 8) dispatch_f3(k_padded, f1, std::integral_constant<int, 8>());
  };
  auto dispatch_f1 = [&](auto k_padded) {
    if (factor1 == 1) dispatch_f2(k_padded, std::integral_constant<int, 1>());
    if (factor1 == 2) dispatch_f2(k_padded, std::integral_constant<int, 2>());
    if (factor1 == 4) dispatch_f2(k_padded, std::integral_constant<int, 4>());
    if (factor1 == 8) dispatch_f2(k_padded, std::integral_constant<int, 8>());
  };

  int k_padded_val;
  if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 56)
    k_padded_val = 56;
  else if (k <= 64)
    k_padded_val = 64;
  else
    k_padded_val = kFlashSortMaxK;

  if (k_padded_val == 4)
    dispatch_f1(std::integral_constant<int, 4>());
  else if (k_padded_val == 8)
    dispatch_f1(std::integral_constant<int, 8>());
  else if (k_padded_val == 16)
    dispatch_f1(std::integral_constant<int, 16>());
  else if (k_padded_val == 32)
    dispatch_f1(std::integral_constant<int, 32>());
  else if (k_padded_val == 56)
    dispatch_f1(std::integral_constant<int, 56>());
  else if (k_padded_val == 64)
    dispatch_f1(std::integral_constant<int, 64>());
  else
    dispatch_f1(std::integral_constant<int, kFlashSortMaxK>());

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kBlockSize, 0));
  int max_active_blocks = num_sm * max_blocks_per_sm;

  if (total_blocks > max_active_blocks) {
    return false;
  }

  return true;
}

}  // namespace llm_sort
}  // namespace cuda
}  // namespace Generators
