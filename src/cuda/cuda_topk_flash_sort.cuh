// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace cg = cooperative_groups;

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

// A single, cooperative kernel that performs a multi-stage Top-K sort.
// It uses cooperative groups to synchronize the entire grid between stages,
// eliminating the high overhead of multiple kernel launches.
// K_PADDED: The compile-time constant for K, padded for efficiency.
// kBlockSize: The number of threads per block.
// kPartitionSize: The size of the vocabulary partition each block handles in Stage 1.
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortKernel(const float* __restrict__ input_scores_in,
                                int* __restrict__ final_indices_out,
                                float* __restrict__ final_scores_out,
                                int* __restrict__ intermediate_indices_1,
                                float* __restrict__ intermediate_scores_1,
                                int* __restrict__ intermediate_indices_2,
                                float* __restrict__ intermediate_scores_2,
                                int vocab_size, int k_final) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int num_partitions = gridDim.x;

  // --- Stage 1: Find Top-K within each partition ---
  // Each thread block finds the top K_PADDED candidates from its assigned partition.
  {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    const int partition_start = partition_idx * kPartitionSize;
    const float* batch_scores_in = input_scores_in;  // Assuming batch_size = 1

    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];

    // Load data from global memory
    for (int i = 0; i < ItemsPerThread; ++i) {
      int global_idx = partition_start + threadIdx.x + i * kBlockSize;
      if (global_idx < vocab_size) {
        thread_keys[i] = batch_scores_in[global_idx];
        thread_values[i] = global_idx;
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = -1;
      }
    }

    // Sort within the block
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Write the top K_PADDED candidates to the intermediate buffer
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      intermediate_scores_1[offset] = thread_keys[0];
      intermediate_indices_1[offset] = thread_values[0];
    }
  }

  grid.sync();

  // --- Stage 2: Iterative Reduction using a binary tree approach ---
  // Blocks iteratively merge the top-K lists from the previous stage.
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

      // Load candidates from two child partitions into shared memory
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

      // Sort the merged list of 2*K_PADDED candidates
      bitonic::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem_scores, smem_indices);

      // Write the top K_PADDED results to the output buffer for this reduction level
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
  // Block 0 writes the final top-k results to the output buffers.
  if (partition_idx == 0 && threadIdx.x < k_final) {
    final_scores_out[threadIdx.x] = scores_in[threadIdx.x];
    final_indices_out[threadIdx.x] = indices_in[threadIdx.x];
  }
}

// Host-side launcher for the FlashSort kernel.
void RunTopKViaFlashSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  // This kernel is designed for batch_size=1 and requires cooperative launch.
  assert(batch_size == 1);

  constexpr int kBlockSize = 256;
  constexpr int kPartitionSize = 8192;
  const int num_partitions = CeilDiv(vocab_size, kPartitionSize);

  dim3 grid(num_partitions);
  dim3 block(kBlockSize);

  void* kernel_args[10];

  auto launch_flash_sort = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    kernel_args[0] = (void*)&scores_in;
    kernel_args[1] = (void*)&data->intermediate_indices_1;
    kernel_args[2] = (void*)&data->intermediate_scores_1;
    kernel_args[3] = (void*)&data->intermediate_indices_1;
    kernel_args[4] = (void*)&data->intermediate_scores_1;
    kernel_args[5] = (void*)&data->intermediate_indices_2;
    kernel_args[6] = (void*)&data->intermediate_scores_2;
    kernel_args[7] = (void*)&vocab_size;
    kernel_args[8] = (void*)&k;

    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, kPartitionSize>, grid, block, kernel_args, 0, stream));
    
    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
  };

  // Select the best padded K value to reduce wasted work.
  if (k == 1) {
    launch_flash_sort(std::integral_constant<int, 1>());
  } else if (k <= 2) {
    launch_flash_sort(std::integral_constant<int, 2>());
  } else if (k <= 4) {
    launch_flash_sort(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    launch_flash_sort(std::integral_constant<int, 16>());
  } else if (k <= 16) {
    launch_flash_sort(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    launch_flash_sort(std::integral_constant<int, 32>());
  } else /*if (k <= 64)*/ {
    launch_flash_sort(std::integral_constant<int, 64>());
  } 

  CUDA_CHECK_LAUNCH();
  data->topk_stride = k;
}

}  // namespace cuda
}  // namespace Generators
