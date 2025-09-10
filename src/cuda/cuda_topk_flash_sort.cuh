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

// By using a union to alias shared memory between Stage 1 (CUB) and Stage 2 (reduction),
// we can maximize the number of candidates for the fast path without exceeding memory limits.
// 4096 candidates use 32KB, which is safe alongside CUB's storage requirements.
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

// Kernel for the "Fast Path": All intermediate candidates fit into one block's shared memory.
// It uses a union to alias shared memory between the CUB radix sort and the final reduction sort.
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortBs1Kernel_Fast(const float* __restrict__ input_scores,
                                        int* __restrict__ final_indices_out,
                                        float* __restrict__ final_scores_out,
                                        int* __restrict__ intermediate_indices_1,
                                        float* __restrict__ intermediate_scores_1,
                                        int vocab_size, int k_final) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int num_partitions = gridDim.x;

  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;

  // This union allows CUB's TempStorage and our reduction buffer to share the same memory space,
  // as they are used in different, synchronized stages of the kernel.
  __shared__ union SharedMemoryAlias {
    typename BlockRadixSort::TempStorage radix_storage;
    struct {
      float scores[kMaxFlashSortCandidates];
      int indices[kMaxFlashSortCandidates];
    } reduction_storage;
  } smem_alias;

  // Compile-time check to ensure the total shared memory usage is within the hardware limit.
  static_assert(sizeof(SharedMemoryAlias) <= 49152, "FlashSortBs1Kernel_Fast: Shared memory union exceeds 48KB limit.");

  // --- Stage 1: Find Top-K within each partition using CUB ---
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

    BlockRadixSort(smem_alias.radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      intermediate_scores_1[offset] = thread_keys[0];
      intermediate_indices_1[offset] = thread_values[0];
    }
  }

  grid.sync(); // Synchronize all blocks after Stage 1 is complete.

  // --- Stage 2: Final Reduction in a Single Block ---
  if (partition_idx == 0) {
    const int num_total_candidates = num_partitions * K_PADDED;
    int sort_size = 1;
    while (sort_size < num_total_candidates) sort_size <<= 1;

    for (int i = threadIdx.x; i < sort_size; i += kBlockSize) {
      if (i < num_total_candidates) {
        smem_alias.reduction_storage.scores[i] = intermediate_scores_1[i];
        smem_alias.reduction_storage.indices[i] = intermediate_scores_1[i];
      } else {
        smem_alias.reduction_storage.scores[i] = -FLT_MAX;
        smem_alias.reduction_storage.indices[i] = -1;
      }
    }
    __syncthreads();

    switch (sort_size) {
      case 2:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 4:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 4>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 8:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 8>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 16:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 16>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 32:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 32>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 64:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 64>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 128:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 128>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 256:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 256>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 512:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 512>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 1024: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 1024>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 2048: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2048>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
      case 4096: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 4096>(smem_alias.reduction_storage.scores, smem_alias.reduction_storage.indices); break;
    }
    __syncthreads();

    if (threadIdx.x < k_final) {
      final_scores_out[threadIdx.x] = smem_alias.reduction_storage.scores[threadIdx.x];
      final_indices_out[threadIdx.x] = smem_alias.reduction_storage.indices[threadIdx.x];
    }
  }
}

// Kernel for the "Slow Path": Iterative reduction for when candidates do not fit in one block.
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortBs1Kernel_Slow(const float* __restrict__ input_scores,
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

  // --- Stage 1: Find Top-K within each partition (same as fast path) ---
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


// Stage 1 kernel for the batched version of FlashSort.
template <int kBlockSize, int kPartitionSize, int K_PADDED>
__global__ void FlashSort_Stage1_BatchKernel(const float* __restrict__ scores_in,
                                             int* __restrict__ intermediate_indices,
                                             float* __restrict__ intermediate_scores,
                                             int vocab_size, int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
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

  auto launch_bs1_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 grid(num_partitions);
    dim3 block(kBlockSize);
    
    const int num_total_candidates = num_partitions * K_PADDED;

    if (num_total_candidates <= kMaxFlashSortCandidates) {
      // FAST PATH LAUNCHER
      void* kernel_args[7];
      kernel_args[0] = (void*)&scores_in;
      kernel_args[1] = (void*)&data->intermediate_indices_1; // Final indices out
      kernel_args[2] = (void*)&data->intermediate_scores_1; // Final scores out
      kernel_args[3] = (void*)&data->intermediate_indices_1; // Intermediate indices
      kernel_args[4] = (void*)&data->intermediate_scores_1; // Intermediate scores
      kernel_args[5] = (void*)&vocab_size;
      kernel_args[6] = (void*)&k;
      
      switch (kPartitionSize) {
        case 1024: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Fast<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream)); break;
        case 2048: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Fast<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream)); break;
        case 4096: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Fast<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream)); break;
        case 8192: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Fast<K_PADDED, kBlockSize, 8192>, grid, block, kernel_args, 0, stream)); break;
      }
    } else {
      // SLOW PATH LAUNCHER
      void* kernel_args[9];
      kernel_args[0] = (void*)&scores_in;
      kernel_args[1] = (void*)&data->intermediate_indices_1; // Final out is determined by reduction
      kernel_args[2] = (void*)&data->intermediate_scores_1; // Final out is determined by reduction
      kernel_args[3] = (void*)&data->intermediate_indices_1;
      kernel_args[4] = (void*)&data->intermediate_scores_1;
      kernel_args[5] = (void*)&data->intermediate_indices_2;
      kernel_args[6] = (void*)&data->intermediate_scores_2;
      kernel_args[7] = (void*)&vocab_size;
      kernel_args[8] = (void*)&k;
      
      switch (kPartitionSize) {
        case 1024: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Slow<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream)); break;
        case 2048: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Slow<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream)); break;
        case 4096: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Slow<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream)); break;
        case 8192: CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel_Slow<K_PADDED, kBlockSize, 8192>, grid, block, kernel_args, 0, stream)); break;
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
    else if (k <= 64) launch_bs1_kernel(std::integral_constant<int, 64>());
    else launch_bs1_kernel(std::integral_constant<int, kFlashSortMaxK>());
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
