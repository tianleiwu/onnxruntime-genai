// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_hybrid_sort.cuh" // For HybridSort_ReducePartitions

namespace Generators {
namespace cuda {
namespace cg = cooperative_groups;

// The new limit of 4096 uses 32KB of shared memory, which is within the 48KB hardware limit.
// The fast path could support vocab_size <= kMaxFlashSortCandidates * kPartitionSize / K_PADDED
// For example, when K_PADDED is 64, kPartitionSize=8192, the fast path could support vocab_size up to 262,144
constexpr int kMaxFlashSortCandidates = 4096;

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
// eliminating the high overhead of multiple kernel launches. This version is
// specialized for batch_size=1.
// K_PADDED: The compile-time constant for K, padded for efficiency.
// kBlockSize: The number of threads per block.
// kPartitionSize: The size of the vocabulary partition each block handles in Stage 1.
// UseFastPath: control which reduction path is compiled.
template <int K_PADDED, int kBlockSize, int kPartitionSize, bool UseFastPath>
__global__ void FlashSortBs1Kernel(const float* __restrict__ input_scores,
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
  {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    const int partition_start = partition_idx * kPartitionSize;

    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];

    // Load data from global memory
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

  // --- Stage 2: Reduction ---
  // Use `if constexpr` to ensure only one path is compiled into the kernel binary,
  // preventing the "too much shared data" error.
  if constexpr (UseFastPath) {
    // Fast Path: All candidates fit in one block's shared memory.
    // The master block (block 0) gathers all results and does a final sort.
    if (partition_idx == 0) {
      // This large allocation is now safe because the `else` branch is compiled out.
      __shared__ float smem_scores[kMaxFlashSortCandidates];
      __shared__ int smem_indices[kMaxFlashSortCandidates];

      const int num_total_candidates = num_partitions * K_PADDED;

      int sort_size = 1;
      while (sort_size < num_total_candidates) sort_size <<= 1;

      for (int i = threadIdx.x; i < sort_size; i += kBlockSize) {
        if (i < num_total_candidates) {
          smem_scores[i] = intermediate_scores_1[i];
          smem_indices[i] = intermediate_indices_1[i];
        } else {
          smem_scores[i] = -FLT_MAX;
          smem_indices[i] = -1;
        }
      }
      __syncthreads();

      switch (sort_size) {
        case 2:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2>(smem_scores, smem_indices); break;
        case 4:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 4>(smem_scores, smem_indices); break;
        case 8:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 8>(smem_scores, smem_indices); break;
        case 16:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 16>(smem_scores, smem_indices); break;
        case 32:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 32>(smem_scores, smem_indices); break;
        case 64:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 64>(smem_scores, smem_indices); break;
        case 128:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 128>(smem_scores, smem_indices); break;
        case 256:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 256>(smem_scores, smem_indices); break;
        case 512:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 512>(smem_scores, smem_indices); break;
        case 1024: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 1024>(smem_scores, smem_indices); break;
        case 2048: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2048>(smem_scores, smem_indices); break;
        case 4096: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 4096>(smem_scores, smem_indices); break;
      }
      __syncthreads();

      if (threadIdx.x < k_final) {
        final_scores_out[threadIdx.x] = smem_scores[threadIdx.x];
        final_indices_out[threadIdx.x] = smem_indices[threadIdx.x];
      }
    }
  } else {
    // Slower, General Path: Use iterative tree reduction for when candidates
    // do not fit in a single block's shared memory.
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
        bitonic::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem_scores, smem_indices);

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

    if (partition_idx == 0 && threadIdx.x < k_final) {
      final_scores_out[threadIdx.x] = scores_in[threadIdx.x];
      final_indices_out[threadIdx.x] = indices_in[threadIdx.x];
    }
  }
}


// Stage 1 kernel for the batched version of FlashSort.
template <int kBlockSize, int kPartitionSize, int K_PADDED>
__global__ void FlashSort_Stage1_BatchKernel(const float* __restrict__ scores_in,
                                             int* __restrict__ intermediate_indices,
                                             float* __restrict__ intermediate_scores,
                                             int vocab_size, int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kPartitionSize;
  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

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

  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  if (threadIdx.x < K_PADDED) {
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K_PADDED;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

// Host-side launcher for the FlashSort kernel.
void RunTopKViaFlashSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  const int kPartitionSize = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, kPartitionSize);

  // CHANGED: Templated lambda to launch the correct kernel specialization
  auto launch_bs1_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 grid(num_partitions);
    dim3 block(kBlockSize);
    void* kernel_args[9];
    kernel_args[0] = (void*)&scores_in;
    kernel_args[1] = (void*)&data->intermediate_indices_1;
    kernel_args[2] = (void*)&data->intermediate_scores_1;
    kernel_args[3] = (void*)&data->intermediate_indices_1;
    kernel_args[4] = (void*)&data->intermediate_scores_1;
    kernel_args[5] = (void*)&data->intermediate_indices_2;
    kernel_args[6] = (void*)&data->intermediate_scores_2;
    kernel_args[7] = (void*)&vocab_size;
    kernel_args[8] = (void*)&k;

    // NEW: Runtime check to select the compile-time path
    const int num_total_candidates = num_partitions * K_PADDED;
    if (num_total_candidates <= kMaxFlashSortCandidates) {
      switch (kPartitionSize) {
        case 1024: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 1024, true>, grid, block, kernel_args, 0, stream)); break;
        case 2048: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 2048, true>, grid, block, kernel_args, 0, stream)); break;
        case 4096: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 4096, true>, grid, block, kernel_args, 0, stream)); break;
        case 8192: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 8192, true>, grid, block, kernel_args, 0, stream)); break;
      }
    } else {
      switch (kPartitionSize) {
        case 1024: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 1024, false>, grid, block, kernel_args, 0, stream)); break;
        case 2048: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 2048, false>, grid, block, kernel_args, 0, stream)); break;
        case 4096: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 4096, false>, grid, block, kernel_args, 0, stream)); break;
        case 8192: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 8192, false>, grid, block, kernel_args, 0, stream)); break;
      }
    }

    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
  };

  auto launch_batch_pipeline = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 grid_stage1(num_partitions, batch_size);
    dim3 block_stage1(kBlockSize);

    switch (kPartitionSize) {
      case 1024:
        FlashSort_Stage1_BatchKernel<kBlockSize, 1024, K_PADDED><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 2048:
        FlashSort_Stage1_BatchKernel<kBlockSize, 2048, K_PADDED><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 4096:
        FlashSort_Stage1_BatchKernel<kBlockSize, 4096, K_PADDED><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 8192:
        FlashSort_Stage1_BatchKernel<kBlockSize, 8192, K_PADDED><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    HybridSort_ReducePartitions<K_PADDED>(data, stream, num_partitions, batch_size, k);
  };

  if (batch_size == 1) {
    if (k <= 16) launch_bs1_kernel(std::integral_constant<int, 16>());
    else if (k <= 32) launch_bs1_kernel(std::integral_constant<int, 32>());
    else if (k <= 64) launch_bs1_kernel(std::integral_constant<int, 64>()); // CHANGED: Added case for k up to 64
    else launch_bs1_kernel(std::integral_constant<int, kFlashSortMaxK>()); // Fallback for larger k
    data->topk_stride = k;
  } else {
    if (k <= 64) {
      launch_batch_pipeline(std::integral_constant<int, 64>());
    } else {
      launch_batch_pipeline(std::integral_constant<int, kHybridSortMaxK>());
    }
  }

  CUDA_CHECK_LAUNCH();
}

}  // namespace cuda
}  // namespace Generators