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

// --- Two-Step Reduction Kernel (Batch Size = 1) ---
// This specialized kernel is only for batch_size = 1 and a limited number of partitions (up to 64).
template <int K_PADDED, int kBlockSize, int kPartitionSize, int Factor1, int Factor2>
__global__ void FlashSortTwoStepBs1Kernel(const float* __restrict__ input_scores,
                                          int* __restrict__ intermediate_indices_1,
                                          float* __restrict__ intermediate_scores_1,
                                          int* __restrict__ intermediate_indices_2,
                                          float* __restrict__ intermediate_scores_2,
                                          int vocab_size) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int num_partitions = gridDim.x;

  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;

  // --- Shared Memory Union ---
  constexpr int kSortSize1 = K_PADDED * Factor1;
  constexpr int kSortSize2 = K_PADDED * Factor2;

  union SharedStorage {
    typename BlockRadixSort::TempStorage stage1_storage;
    struct {
      __align__(128) float scores[kSortSize1];
      __align__(128) int indices[kSortSize1];
    } step1_storage;
    struct {
      __align__(128) float scores[kSortSize2];
      __align__(128) int indices[kSortSize2];
    } step2_storage;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Find Top-K within each partition ---
  {
    const int partition_start = partition_idx * kPartitionSize;
    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i) {
      int global_idx = partition_start + threadIdx.x + i * kBlockSize;
      if (global_idx < vocab_size) {
        thread_keys[i] = input_scores[global_idx];
        thread_values[i] = global_idx;
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = -1;
      }
    }
    BlockRadixSort(smem.stage1_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      intermediate_scores_1[offset] = thread_keys[0];
      intermediate_indices_1[offset] = thread_values[0];
    }
  }
  grid.sync();

  // --- Stage 2, Step 1: First Reduction ---
  int partitions_after_step1 = num_partitions;
  if (Factor1 > 1) {
    partitions_after_step1 = CeilDiv(num_partitions, Factor1);
    if (partition_idx < partitions_after_step1) {
      int first_child = partition_idx * Factor1;
      int num_to_process = min(Factor1, num_partitions - first_child);
      for (int i = threadIdx.x; i < kSortSize1; i += kBlockSize) {
        if (i < K_PADDED * num_to_process) {
          smem.step1_storage.scores[i] = intermediate_scores_1[(first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED)];
          smem.step1_storage.indices[i] = intermediate_indices_1[(first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED)];
        } else {
          smem.step1_storage.scores[i] = -FLT_MAX;
          smem.step1_storage.indices[i] = -1;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize1>(smem.step1_storage.scores, smem.step1_storage.indices);
      if (threadIdx.x < K_PADDED) {
        intermediate_scores_2[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step1_storage.scores[threadIdx.x];
        intermediate_indices_2[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step1_storage.indices[threadIdx.x];
      }
    }
    grid.sync();
  }

  // --- Stage 2, Step 2: Second Reduction ---
  if (Factor2 > 1) {
    int partitions_after_step2 = CeilDiv(partitions_after_step1, Factor2);
    if (partition_idx < partitions_after_step2) {
      int first_child = partition_idx * Factor2;
      int num_to_process = min(Factor2, partitions_after_step1 - first_child);
      float* scores_in = (Factor1 > 1) ? intermediate_scores_2 : intermediate_scores_1;
      int* indices_in = (Factor1 > 1) ? intermediate_indices_2 : intermediate_indices_1;

      for (int i = threadIdx.x; i < kSortSize2; i += kBlockSize) {
        if (i < K_PADDED * num_to_process) {
          smem.step2_storage.scores[i] = scores_in[(first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED)];
          smem.step2_storage.indices[i] = indices_in[(first_child + i / K_PADDED) * K_PADDED + (i % K_PADDED)];
        } else {
          smem.step2_storage.scores[i] = -FLT_MAX;
          smem.step2_storage.indices[i] = -1;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize2>(smem.step2_storage.scores, smem.step2_storage.indices);
      if (threadIdx.x < K_PADDED) {
        // Final result goes to buffer 1, as it's the second step
        intermediate_scores_1[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step2_storage.scores[threadIdx.x];
        intermediate_indices_1[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step2_storage.indices[threadIdx.x];
      }
    }
  }
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kFlashSortMaxK;
}

// Below is to optimize parition size for common vocab_size (padded to multiple of 256) used in open source LLM:
//   32256, 32512, 128256, 128512, 152064, 152320, 200192, 200448, 201216, 201472, 262400, 262656.
// Constraints: partition_size are multiple of 256, partition_size <= 8192.
// Goal: mimize average waste ratio to get total partitions be one of 2, 4, 8, 16, 32 and 64.
// For example, vocab_size=201088, ideal partition size is 3142 to fit 64 partitions. The waste ratio is 1 - 3142/3328 = 0.055.
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
        int allocated_space = partition_size * target;
        int waste = allocated_space - vocab_size;
        double cost_ratio = static_cast<double>(waste) / vocab_size;

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

  // --- Host-side lookup for reduction factors ---
  int factor1 = 1, factor2 = 1;
  int num_reduction_steps = 0;

  if (num_partitions > 1) {
    if (num_partitions <= 8) {
      num_reduction_steps = 1;
      if (num_partitions <= 2)
        factor1 = 2;
      else if (num_partitions <= 4)
        factor1 = 4;
      else
        factor1 = 8;
    } else {
      num_reduction_steps = 2;
      if (num_partitions <= 16) {
        factor1 = 4;
        factor2 = 4;
      } else if (num_partitions <= 32) {  // 17-32 partitions
        factor1 = 8;
        factor2 = 4;
      } else {  // 33-64 partitions
        factor1 = 8;
        factor2 = 8;
      }
    }
  }

  // Determine final output buffer based on number of steps
  if (num_reduction_steps % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }

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
  data->topk_stride = k_padded_val;

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  auto launch_kernel = [&](auto k_padded, auto f1, auto f2) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int F1 = decltype(f1)::value;
    constexpr int F2 = decltype(f2)::value;
    dim3 grid(num_partitions);
    dim3 block(kBlockSize);
    switch (partition_size) {
      case kAllowedPartitionSizes[0]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortTwoStepBs1Kernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], F1, F2>, grid, block, kernel_args, 0, stream));
        break;
      case kAllowedPartitionSizes[1]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortTwoStepBs1Kernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], F1, F2>, grid, block, kernel_args, 0, stream));
        break;
      case kAllowedPartitionSizes[2]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortTwoStepBs1Kernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], F1, F2>, grid, block, kernel_args, 0, stream));
        break;
      case kAllowedPartitionSizes[3]:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortTwoStepBs1Kernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], F1, F2>, grid, block, kernel_args, 0, stream));
        break;
    }
  };

  auto dispatch_f2 = [&](auto k_padded, auto f1) {
    if (factor2 == 1) launch_kernel(k_padded, f1, std::integral_constant<int, 1>());
    if (factor2 == 4) launch_kernel(k_padded, f1, std::integral_constant<int, 4>());
    if (factor2 == 8) launch_kernel(k_padded, f1, std::integral_constant<int, 8>());
  };

  auto dispatch_f1 = [&](auto k_padded) {
    if (factor1 == 1) dispatch_f2(k_padded, std::integral_constant<int, 1>());
    if (factor1 == 2) dispatch_f2(k_padded, std::integral_constant<int, 2>());
    if (factor1 == 4) dispatch_f2(k_padded, std::integral_constant<int, 4>());
    if (factor1 == 8) dispatch_f2(k_padded, std::integral_constant<int, 8>());
  };

  if (k <= 4) {
    dispatch_f1(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    dispatch_f1(std::integral_constant<int, 8>());
  } else if (k <= 16) {
    dispatch_f1(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    dispatch_f1(std::integral_constant<int, 32>());
  } else if (k <= 56) {
    dispatch_f1(std::integral_constant<int, 56>());  // optimize common use case of k=50.
  } else if (k <= 64) {
    dispatch_f1(std::integral_constant<int, 64>());
  } else {
    dispatch_f1(std::integral_constant<int, kFlashSortMaxK>());
  }

  CUDA_CHECK_LAUNCH();
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kFlashSortMaxK || batch_size != 1) {
    return false;
  }
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  if (partition_size == 0) {  // No suitable partition size found
    return false;
  }
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > 64) {
    return false;
  }

  int cooperative_launch_support = 0;
  cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, 0);
  return cooperative_launch_support == 1;
}

}  // namespace llm_sort
}  // namespace cuda
}  // namespace Generators
