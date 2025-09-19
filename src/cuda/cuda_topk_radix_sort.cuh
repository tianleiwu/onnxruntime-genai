// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_llm_sort.cuh"  // For EstimateBestPartitionSize

namespace Generators {
namespace cuda {
namespace radix_sort {
inline size_t GetTempStorageBytes(int vocab_size, cudaStream_t stream) {
 return 0;
}

// --- Stage 1: Find Top-K within each vocabulary partition ---
// This stage is identical in function to the first stage of llm_sort and flash_sort.
template <int kBlockSize, int kPartitionSize, int K_PADDED>
__global__ void OneStepSort_Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                                     int* __restrict__ intermediate_indices,
                                                     float* __restrict__ intermediate_scores,
                                                     int vocab_size, int num_partitions) {
  __shared__ typename Stage1TempStorage smem;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

// --- Stage 2: One-Step Reduction Kernel ---
// A single thread block is launched per batch item. It loads ALL candidates from
// Stage 1 into registers, sorts them using a striped layout, and writes the final top-k.
template <int kBlockSize, int K_PADDED, int kMaxPartitions>
__global__ void OneStepSort_Stage2_ReduceKernel(const float* __restrict__ scores_in,
                                                const int* __restrict__ indices_in,
                                                float* __restrict__ scores_out,
                                                int* __restrict__ indices_out,
                                                int num_partitions) {
  const int batch_idx = blockIdx.x;
  constexpr int kSortSize = K_PADDED * kMaxPartitions;
  constexpr int kItemsPerThread = (kSortSize + kBlockSize - 1) / kBlockSize;

  union SharedStorage {
    struct {
      __align__(128) float scores[K_PADDED];
      __align__(128) int indices[K_PADDED];
    } final_topk;

    // CUB's temporary storage for the block-wide radix sort.
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>::TempStorage cub_temp_storage;
  };
  __shared__ SharedStorage smem;

  // --- 1. Load data from Global into Registers (Blocked arrangement) ---
  float thread_scores[kItemsPerThread];
  int thread_indices[kItemsPerThread];
  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
  const int num_elements_to_load = num_partitions * K_PADDED;

  for (int i = 0; i < kItemsPerThread; ++i) {
    int load_idx = threadIdx.x * kItemsPerThread + i;
    if (load_idx < num_elements_to_load) {
      thread_scores[i] = scores_in[in_base_offset + load_idx];
      thread_indices[i] = indices_in[in_base_offset + load_idx];
    } else {
      thread_scores[i] = -FLT_MAX;
      thread_indices[i] = INT_MAX;
    }
  }

  // --- 2. Sort data held in registers ---
  cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>(smem.cub_temp_storage)
      .SortDescendingBlockedToStriped(thread_scores, thread_indices);

  __syncthreads();

  // --- 3. Write top-K results from registers to shared memory ---
  // After the striped sort, the top K_PADDED elements are in the first item slot
  // of the first K_PADDED threads.
  if (threadIdx.x < K_PADDED) {
    smem.final_topk.scores[threadIdx.x] = thread_scores[0];
    smem.final_topk.indices[threadIdx.x] = thread_indices[0];
  }

  __syncthreads();

  // --- 4. Write final top-k results from shared memory to global memory ---
  const size_t out_base_offset = static_cast<size_t>(batch_idx) * K_PADDED;
  if (threadIdx.x < K_PADDED) {
    scores_out[out_base_offset + threadIdx.x] = smem.final_topk.scores[threadIdx.x];
    indices_out[out_base_offset + threadIdx.x] = smem.final_topk.indices[threadIdx.x];
  }
}

// --- Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  // Use the same partitioning logic as LLM sort to ensure <= 64 partitions.
  const int partition_size = llm_sort::EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  assert(num_partitions <= 64);

  // For this sort, K must be padded to at least 4 for performance.
  // We can reuse the padding logic from flash_sort.
  int k_padded_val = kLlmSortMaxK;  // Max K for LLM sort
  if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 64)
    k_padded_val = 64;

  // --- Launch Stage 1 ---
  auto launch_stage1 = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int kBlockSize = 256;
    dim3 grid(num_partitions, batch_size);
    dim3 block(kBlockSize);

    // This launcher logic needs to match the partition sizes from llm_sort
    if (partition_size == llm_sort::kAllowedPartitionSizes[0]) {
      OneStepSort_Stage1_FindPartitionsTopK<kBlockSize, llm_sort::kAllowedPartitionSizes[0], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else if (partition_size == llm_sort::kAllowedPartitionSizes[1]) {
      OneStepSort_Stage1_FindPartitionsTopK<kBlockSize, llm_sort::kAllowedPartitionSizes[1], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else if (partition_size == llm_sort::kAllowedPartitionSizes[2]) {
      OneStepSort_Stage1_FindPartitionsTopK<kBlockSize, llm_sort::kAllowedPartitionSizes[2], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else {
      OneStepSort_Stage1_FindPartitionsTopK<kBlockSize, llm_sort::kAllowedPartitionSizes[3], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    }
  };

  // --- Launch Stage 2 ---
  auto launch_stage2 = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    // Use a larger block size for stage 2 to maximize parallelism for the load and sort.
    constexpr int kBlockSize = 1024;
    dim3 grid(batch_size);
    dim3 block(kBlockSize);
    OneStepSort_Stage2_ReduceKernel<kBlockSize, K_PADDED, 64><<<grid, block, 0, stream>>>(
        data->intermediate_scores_1, data->intermediate_indices_1,
        data->intermediate_scores_2, data->intermediate_indices_2, num_partitions);
  };

  // --- Dispatch based on padded k ---
  if (k_padded_val == 4) {
    launch_stage1(std::integral_constant<int, 4>());
    CUDA_CHECK_LAUNCH();
    launch_stage2(std::integral_constant<int, 4>());
  } else if (k_padded_val == 8) {
    launch_stage1(std::integral_constant<int, 8>());
    CUDA_CHECK_LAUNCH();
    launch_stage2(std::integral_constant<int, 8>());
  } else if (k_padded_val == 16) {
    launch_stage1(std::integral_constant<int, 16>());
    CUDA_CHECK_LAUNCH();
    launch_stage2(std::integral_constant<int, 16>());
  } else if (k_padded_val == 32) {
    launch_stage1(std::integral_constant<int, 32>());
    CUDA_CHECK_LAUNCH();
    launch_stage2(std::integral_constant<int, 32>());
  } else {  // k_padded_val == 64
    launch_stage1(std::integral_constant<int, 64>());
    CUDA_CHECK_LAUNCH();
    launch_stage2(std::integral_constant<int, 64>());
  }

  CUDA_CHECK_LAUNCH();

  // The final results are now in the second set of intermediate buffers.
  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k_padded_val;
}

}  // namespace onestep_sort
}  // namespace cuda
}  // namespace Generators