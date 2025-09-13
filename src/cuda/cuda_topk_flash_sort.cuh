// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <type_traits>  // For std::integral_constant
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_hybrid_sort.cuh"

namespace Generators {
namespace cuda {
namespace flash_sort {

namespace cg = cooperative_groups;

// Utility to swap pointers, used during the reduction phase.
__host__ __device__ inline void swap_ptr(float*& a, float*& b) {
  float* tmp = a;
  a = b;
  b = tmp;
}

__host__ __device__ inline void swap_ptr(int*& a, int*& b) {
  int* tmp = a;
  a = b;
  b = tmp;
}

#define USE_BATCH_1_FLASH_SORT_KERNEL 1

#if USE_BATCH_1_FLASH_SORT_KERNEL
// --- Specialized Kernel for Batch Size = 1 ---
// This version is optimized for batch_size=1 by removing all batch-related
// indexing and pointer arithmetic, which can lead to better performance.
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortBs1Kernel(const float* __restrict__ input_scores,
                                   int* __restrict__ final_indices_out,
                                   float* __restrict__ final_scores_out,
                                   int* __restrict__ intermediate_indices_1,
                                   float* __restrict__ intermediate_scores_1,
                                   int* __restrict__ intermediate_indices_2,
                                   float* __restrict__ intermediate_scores_2,
                                   int vocab_size,
                                   int k_final) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int num_partitions = gridDim.x;

  // --- Stage 1: Find Top-K within each partition ---
  {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

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
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      intermediate_scores_1[offset] = thread_keys[0];
      intermediate_indices_1[offset] = thread_values[0];
    }
  }

  grid.sync();

  // --- Stage 2: Iterative Tree Reduction ---
  int* indices_in = intermediate_indices_1;
  float* scores_in = intermediate_scores_1;
  int* indices_out = intermediate_indices_2;
  float* scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = (partitions_remaining + 1) / 2;
    if (partition_idx < num_active_blocks) {
      constexpr int kSortSize = K_PADDED * 2;
      __shared__ float smem_scores[kSortSize];
      __shared__ int smem_indices[kSortSize];
      int first_child_partition = partition_idx * 2;
      int second_child_partition = first_child_partition + 1;
      int num_partitions_to_process = (second_child_partition < partitions_remaining) ? 2 : 1;
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        if (i < K_PADDED * num_partitions_to_process) {
          int part_idx = i / K_PADDED;
          int element_idx = i % K_PADDED;
          size_t global_offset = (first_child_partition + part_idx) * K_PADDED + element_idx;
          smem_scores[i] = scores_in[global_offset];
          smem_indices[i] = indices_in[global_offset];
        } else {
          smem_scores[i] = -FLT_MAX;
          smem_indices[i] = -1;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem_scores, smem_indices);
      if (threadIdx.x < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
        scores_out[out_offset] = smem_scores[threadIdx.x];
        indices_out[out_offset] = smem_indices[threadIdx.x];
      }
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(scores_in, scores_out);
    swap_ptr(indices_in, indices_out);
    grid.sync();
  }

  // --- Final Output ---
  if (partition_idx == 0 && threadIdx.x < k_final) {
    final_scores_out[threadIdx.x] = scores_in[threadIdx.x];
    final_indices_out[threadIdx.x] = indices_in[threadIdx.x];
  }
}
#else
#define FlashSortBs1Kernel FlashSortKernel
#endif

// --- General Kernel for Any Batch Size ---
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortKernel(const float* __restrict__ input_scores,
                                int* __restrict__ final_indices_out,
                                float* __restrict__ final_scores_out,
                                int* __restrict__ intermediate_indices_1,
                                float* __restrict__ intermediate_scores_1,
                                int* __restrict__ intermediate_indices_2,
                                float* __restrict__ intermediate_scores_2,
                                int vocab_size,
                                int k_final) {
  cg::grid_group grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  const float* batch_input_scores = input_scores + static_cast<size_t>(batch_idx) * vocab_size;
  const size_t batch_intermediate_offset = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
  int* batch_intermediate_indices_1 = intermediate_indices_1 + batch_intermediate_offset;
  float* batch_intermediate_scores_1 = intermediate_scores_1 + batch_intermediate_offset;
  int* batch_intermediate_indices_2 = intermediate_indices_2 + batch_intermediate_offset;
  float* batch_intermediate_scores_2 = intermediate_scores_2 + batch_intermediate_offset;
  int* batch_final_indices_out = final_indices_out + static_cast<size_t>(batch_idx) * k_final;
  float* batch_final_scores_out = final_scores_out + static_cast<size_t>(batch_idx) * k_final;

  // Stages 1, 2 and Final Output are identical to the merged version.
  // The only difference is the use of batch-adjusted pointers.
  // --- Stage 1: Find Top-K within each partition ---
  {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
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
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      batch_intermediate_scores_1[offset] = thread_keys[0];
      batch_intermediate_indices_1[offset] = thread_values[0];
    }
  }
  grid.sync();
  // --- Stage 2: Iterative Tree Reduction ---
  int* indices_in = batch_intermediate_indices_1;
  float* scores_in = batch_intermediate_scores_1;
  int* indices_out = batch_intermediate_indices_2;
  float* scores_out = batch_intermediate_scores_2;
  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = (partitions_remaining + 1) / 2;
    if (partition_idx < num_active_blocks) {
      constexpr int kSortSize = K_PADDED * 2;
      __shared__ float smem_scores[kSortSize];
      __shared__ int smem_indices[kSortSize];
      int first_child_partition = partition_idx * 2;
      int second_child_partition = first_child_partition + 1;
      int num_partitions_to_process = (second_child_partition < partitions_remaining) ? 2 : 1;
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        if (i < K_PADDED * num_partitions_to_process) {
          int part_idx = i / K_PADDED;
          int element_idx = i % K_PADDED;
          size_t global_offset = (first_child_partition + part_idx) * K_PADDED + element_idx;
          smem_scores[i] = scores_in[global_offset];
          smem_indices[i] = indices_in[global_offset];
        } else {
          smem_scores[i] = -FLT_MAX;
          smem_indices[i] = -1;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem_scores, smem_indices);
      if (threadIdx.x < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
        scores_out[out_offset] = smem_scores[threadIdx.x];
        indices_out[out_offset] = smem_indices[threadIdx.x];
      }
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(scores_in, scores_out);
    swap_ptr(indices_in, indices_out);
    grid.sync();
  }
  // --- Final Output ---
  if (partition_idx == 0 && threadIdx.x < k_final) {
    batch_final_scores_out[threadIdx.x] = scores_in[threadIdx.x];
    batch_final_indices_out[threadIdx.x] = indices_in[threadIdx.x];
  }
}

// --- Unified Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // For cooperative kernels, the total number of blocks in the grid is limited by what the device can support.
  // This limit is typically the number of SMs multiplied by the max active blocks per SM for a given kernel.
  // Instead of querying this at runtime (which is complex with templated kernels), we use a
  // conservative threshold. If the required grid size exceeds this, we fall back to a
  // non-cooperative algorithm like hybrid_sort, which is the next best for small k.
  const int total_blocks = num_partitions * batch_size;
  constexpr int kMaxCooperativeBlocks = 512;

  if (total_blocks > kMaxCooperativeBlocks) {
    // The grid is too large for a cooperative launch, so fall back to hybrid sort.
    hybrid_sort::RunTopK(data, stream, scores_in, vocab_size, batch_size, k);
    return;
  }

  // Buffer 1 is the final destination for the output.
  int* final_indices_out = data->intermediate_indices_1;
  float* final_scores_out = data->intermediate_scores_1;

  // Buffer 2 will be the FIRST intermediate buffer (for Stage 1 output).
  int* intermediate_indices_1 = data->intermediate_indices_2;
  float* intermediate_scores_1 = data->intermediate_scores_2;

  // Buffer 1 will be the SECOND intermediate buffer (for ping-pong reduction).
  int* intermediate_indices_2 = data->intermediate_indices_1;
  float* intermediate_scores_2 = data->intermediate_scores_1;

  data->topk_scores = final_scores_out;
  data->topk_indices = final_indices_out;
  data->topk_stride = k;

  void* kernel_args[9];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&final_indices_out;
  kernel_args[2] = (void*)&final_scores_out;
  kernel_args[3] = (void*)&intermediate_indices_1;
  kernel_args[4] = (void*)&intermediate_scores_1;
  kernel_args[5] = (void*)&intermediate_indices_2;
  kernel_args[6] = (void*)&intermediate_scores_2;
  kernel_args[7] = (void*)&vocab_size;
  kernel_args[8] = (void*)&k;
    
  // This lambda handles selecting the kernel and launching it with the correct K_PADDED value.
  auto launch_flash_sort = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 block(kBlockSize);

    // At runtime, choose the optimal kernel and configure the grid.
    if (batch_size == 1) {
      dim3 grid(num_partitions);
      switch (partition_size) {
        case 1024:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream));
          break;
        case 2048:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream));
          break;
        case 4096:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream));
          break;
        case 8192:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 8192>, grid, block, kernel_args, 0, stream));
          break;
      }
    } else {
      dim3 grid(num_partitions, batch_size);
      switch (partition_size) {
        case 1024:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream));
          break;
        case 2048:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream));
          break;
        case 4096:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream));
          break;
        case 8192:
          CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 8192>, grid, block, kernel_args, 0, stream));
          break;
      }
    }
  };

  // Select the padded K value at runtime and call the launch logic.
  if (k <= 4) {
    launch_flash_sort(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    launch_flash_sort(std::integral_constant<int, 8>());
  } else if (k <= 16) {
    launch_flash_sort(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    launch_flash_sort(std::integral_constant<int, 32>());
  } else if (k <= 64) {
    launch_flash_sort(std::integral_constant<int, 64>());
  } else {
    launch_flash_sort(std::integral_constant<int, kFlashSortMaxK>());
  }

  CUDA_CHECK_LAUNCH();
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kFlashSortMaxK) {
    return false;
  }

  // Check for cooperative launch support
  int cooperative_launch_support = 0;
  cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, 0);
  if (!cooperative_launch_support) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int partition_size = hybrid_sort::EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  const int total_blocks = num_partitions * batch_size;

  // Choose kernel using the same logic as in RunTopK.
  auto get_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    if (batch_size == 1) {
      switch (partition_size) {
        case 1024:
          return (void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 1024>;
        case 2048:
          return (void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 2048>;
        case 4096:
          return (void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 4096>;
        default:
          return (void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 8192>;
      }
    } else {
      switch (partition_size) {
        case 1024:
          return (void*)FlashSortKernel<K_PADDED, kBlockSize, 1024>;
        case 2048:
          return (void*)FlashSortKernel<K_PADDED, kBlockSize, 2048>;
        case 4096:
          return (void*)FlashSortKernel<K_PADDED, kBlockSize, 4096>;
        default:
          return (void*)FlashSortKernel<K_PADDED, kBlockSize, 8192>;
      }
    }
  };

  void* kernel = nullptr;
  int K_PADDED = 0;
  if (k <= 4) {
    kernel = get_kernel(std::integral_constant<int, 4>());
    K_PADDED = 4;
  } else if (k <= 8) {
    kernel = get_kernel(std::integral_constant<int, 8>());
    K_PADDED = 8;
  } else if (k <= 16) {
    kernel = get_kernel(std::integral_constant<int, 16>());
    K_PADDED = 16;
  } else if (k <= 32) {
    kernel = get_kernel(std::integral_constant<int, 32>());
    K_PADDED = 32;
  } else if (k <= 64) {
    kernel = get_kernel(std::integral_constant<int, 64>());
    K_PADDED = 64;
  } else {
    kernel = get_kernel(std::integral_constant<int, kFlashSortMaxK>());
    K_PADDED = kFlashSortMaxK;
  }

  // Size of shared memory used in kernel: thread_keys, thread_values, smem_score, smem_indices
  int shared_mem_bytes = (partition_size / kBlockSize + K_PADDED * 2) * (sizeof(float) + sizeof(int));

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));

  int max_blocks_per_sm = 0;
  // Assume blockDim is known or passed into IsSupported
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm,
      kernel,
      kBlockSize,
      shared_mem_bytes));

  int max_active_blocks = num_sm * max_blocks_per_sm;

  if (total_blocks > max_active_blocks) {
    return false;
  }

  return true;
}

}  // namespace flash_sort
}  // namespace cuda
}  // namespace Generators

