// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <type_traits>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace iterative_sort {

/**
 * @brief A single-kernel cooperative sort, specialized for **mid-to-large k**.
 *
 * Algorithm Overview:
 * This is an evolution of the original iterative sort, now featuring an adaptive reduction factor.
 *
 * 1.  **Host-Side Planning**: A simple host-side planner (`GetBestReductionFactor`) chooses a
 * single, fixed reduction factor (e.g., 6, 7, or 8) for the entire reduction process.
 * The factor is chosen to minimize workload imbalance for the specific number of partitions.
 *
 * 2.  **Stage 1 (Partition Top-K)**: All blocks find top candidates in parallel.
 *
 * 3.  **Stage 2 (Adaptive Iterative Reduction)**: The kernel enters a loop, repeatedly
 * merging candidates using the single, pre-calculated reduction factor. This maintains the
 * low overhead of a single kernel launch while being more efficient than a globally fixed
 * factor for partition counts that are not powers of 4.
 *
 * Performance Characteristics:
 * -   **Strengths**: Fast for mid-to-large `k` where a consistent reduction strategy is
 * beneficial and the overhead of more complex planning is not justified.
 * -   **Weaknesses**: Requires cooperative launch. May be less optimal than fully adaptive
 * (multi-factor) plans for certain partition counts.
 */

namespace cg = cooperative_groups;

__host__ __device__ inline void swap_ptr(float*& a, float*& b) { float* tmp = a; a = b; b = tmp; }
__host__ __device__ inline void swap_ptr(int*& a, int*& b) { int* tmp = a; a = b; b = tmp; }

template <int K_PADDED, int kBlockSize, int kPartitionSize, int kReductionFactor>
__global__ void AdaptiveIterativeSortKernel(const float* __restrict__ input_scores,
                                            int* __restrict__ intermediate_indices_1,
                                            float* __restrict__ intermediate_scores_1,
                                            int* __restrict__ intermediate_indices_2,
                                            float* __restrict__ intermediate_scores_2,
                                            int vocab_size) {
  constexpr bool UseCubMergeSort = (kPartitionSize <= 1024);
  cg::grid_group grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  constexpr int kSortSize = K_PADDED * kReductionFactor;
  constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
      Stage1TempStorageType stage1_storage;
      typename cub::WarpMergeSort<uint64_t, (kSortSize + 31) / 32, 32>::TempStorage cub_warp_storage;
#ifdef STABLE_TOPK
      typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage cub_block_merge_storage;
#else
      typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage cub_block_merge_storage;
#endif
      struct {
          __align__(128) float scores[kSortSize];
          __align__(128) int indices[kSortSize];
      } stage2_storage;
  };
  __shared__ SharedStorage smem;

  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, UseCubMergeSort>(
      input_scores, intermediate_indices_1, intermediate_scores_1, vocab_size, num_partitions, smem.stage1_storage);
  grid.sync();

  int* p_indices_in = intermediate_indices_1;
  float* p_scores_in = intermediate_scores_1;
  int* p_indices_out = intermediate_indices_2;
  float* p_scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = CeilDiv(partitions_remaining, kReductionFactor);
    if (partition_idx < num_active_blocks) {
      const size_t in_batch_offset = static_cast<size_t>(batch_idx) * partitions_remaining * K_PADDED;
      const size_t out_batch_offset = static_cast<size_t>(batch_idx) * num_active_blocks * K_PADDED;
      
      int first_child = partition_idx * kReductionFactor;
      int num_to_process = min(kReductionFactor, partitions_remaining - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize, K_PADDED, kItemsPerThread>(
          p_scores_in + in_batch_offset, p_indices_in + in_batch_offset,
          p_scores_out + out_batch_offset, p_indices_out + out_batch_offset,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(p_scores_in, p_scores_out);
    swap_ptr(p_indices_in, p_indices_out);
    grid.sync();
  }
}

inline int EstimateBestPartitionSize(int vocab_size) {
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  return 4096;
}

// Simple planner to find the best single reduction factor
inline int GetBestReductionFactor(int num_partitions, int k_padded) {
    int best_factor = 2;
    float min_waste = 1.0f;

    // Iterate through possible factors, favoring larger ones
    for (int factor = 8; factor >= 2; --factor) {
        if (k_padded * factor > 4096) continue; // Shared mem limit

        int num_blocks = CeilDiv(num_partitions, factor);
        int last_block_workload = num_partitions - (num_blocks - 1) * factor;
        float waste_ratio = 1.0f - (float)last_block_workload / factor;

        if (waste_ratio < min_waste) {
            min_waste = waste_ratio;
            best_factor = factor;
        }
    }
    return best_factor;
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));
  if (data->iterative_sort_partition_size == 0) {
    data->iterative_sort_partition_size = EstimateBestPartitionSize(vocab_size);
  }

  const int partition_size = data->iterative_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  
  int k_padded_val;
  if (k <= 4) k_padded_val = 4;
  else if (k <= 8) k_padded_val = 8;
  else if (k <= 16) k_padded_val = 16;
  else if (k <= 32) k_padded_val = 32;
  else k_padded_val = 64;

  const int reduction_factor = GetBestReductionFactor(num_partitions, k_padded_val);

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  auto launch_iterative_sort = [&](auto k_padded, auto r_factor) {
      constexpr int K_PADDED = decltype(k_padded)::value;
      constexpr int R_FACTOR = decltype(r_factor)::value;
      constexpr int kBlockSize = 256;
      dim3 grid(num_partitions, batch_size);
      dim3 block(kBlockSize);

      switch (partition_size) {
          case 1024:
              CUDA_CHECK((cudaLaunchCooperativeKernel((void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 1024, R_FACTOR>, grid, block, kernel_args, 0, stream)));
              break;
          case 2048:
              CUDA_CHECK((cudaLaunchCooperativeKernel((void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 2048, R_FACTOR>, grid, block, kernel_args, 0, stream)));
              break;
          default:
              CUDA_CHECK((cudaLaunchCooperativeKernel((void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 4096, R_FACTOR>, grid, block, kernel_args, 0, stream)));
              break;
      }
  };

  auto dispatch_by_factor = [&](auto k_padded) {
      switch(reduction_factor) {
          case 2: launch_iterative_sort(k_padded, std::integral_constant<int, 2>()); break;
          case 3: launch_iterative_sort(k_padded, std::integral_constant<int, 3>()); break;
          case 4: launch_iterative_sort(k_padded, std::integral_constant<int, 4>()); break;
          case 5: launch_iterative_sort(k_padded, std::integral_constant<int, 5>()); break;
          case 6: launch_iterative_sort(k_padded, std::integral_constant<int, 6>()); break;
          case 7: launch_iterative_sort(k_padded, std::integral_constant<int, 7>()); break;
          case 8: launch_iterative_sort(k_padded, std::integral_constant<int, 8>()); break;
          default: launch_iterative_sort(k_padded, std::integral_constant<int, 4>()); break; // Fallback
      }
  };

  if (k_padded_val == 4) dispatch_by_factor(std::integral_constant<int, 4>());
  else if (k_padded_val == 8) dispatch_by_factor(std::integral_constant<int, 8>());
  else if (k_padded_val == 16) dispatch_by_factor(std::integral_constant<int, 16>());
  else if (k_padded_val == 32) dispatch_by_factor(std::integral_constant<int, 32>());
  else dispatch_by_factor(std::integral_constant<int, 64>());
  
  CUDA_CHECK_LAUNCH();

  int num_reduction_loops = 0;
  if (num_partitions > 1) {
    int partitions_remaining = num_partitions;
    while (partitions_remaining > 1) {
      partitions_remaining = (partitions_remaining + reduction_factor - 1) / reduction_factor;
      num_reduction_loops++;
    }
  }

  if (num_reduction_loops % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }
  data->topk_stride = k_padded_val;
}

template <int K_PADDED, int kReductionFactor>
bool CheckSupportForFactor(int batch_size, int num_partitions, int partition_size) {
    constexpr int kBlockSize = 256;
    const int total_blocks = num_partitions * batch_size;
    void* kernel = nullptr;

    switch (partition_size) {
        case 1024:
            kernel = (void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 1024, kReductionFactor>;
            break;
        case 2048:
            kernel = (void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 2048, kReductionFactor>;
            break;
        default:
            kernel = (void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, 4096, kReductionFactor>;
            break;
    }
    return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

template <int K_PADDED>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
    // Check against the worst-case (largest) reduction factor for resource usage.
    constexpr int kMaxReductionFactor = 8;
    return CheckSupportForFactor<K_PADDED, kMaxReductionFactor>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
    if (k > kIterativeSortMaxK) {
        return false;
    }

    const int partition_size = EstimateBestPartitionSize(vocab_size);
    const int num_partitions = CeilDiv(vocab_size, partition_size);

    int k_padded_val;
    if (k <= 4) k_padded_val = 4;
    else if (k <= 8) k_padded_val = 8;
    else if (k <= 16) k_padded_val = 16;
    else if (k <= 32) k_padded_val = 32;
    else k_padded_val = 64;

    if (k_padded_val == 4) return CheckSupport<4>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 8) return CheckSupport<8>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 16) return CheckSupport<16>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 32) return CheckSupport<32>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 64) return CheckSupport<64>(batch_size, num_partitions, partition_size);
    
    return false;
}

}  // namespace iterative_sort
}  // namespace cuda
}  // namespace Generators

