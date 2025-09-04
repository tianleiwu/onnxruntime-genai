// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {

// Top-K kernel using cub::BlockRadixSort directly on register data, with a balanced final write.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CubRegisterSort(const float* __restrict__ scores_in,
                                              int* __restrict__ intermediate_indices,
                                              float* __restrict__ intermediate_scores,
                                              int vocab_size,
                                              int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  // Specialize BlockRadixSort for our key-value pairs.
  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;

  // 1. Coalesced load from global memory directly into per-thread registers.
  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
      thread_keys[i] = batch_scores_in[global_idx];
      thread_values[i] = global_idx;
    } else {
      thread_keys[i] = -FLT_MAX;
      thread_values[i] = -1;
    }
  }

  // 2. Sort the keys and values held in registers across the entire block.
  // The result is a sorted, striped layout across the threads. This means
  // thread 0 has rank 0, thread 1 has rank 1, etc.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  // 3. The first K threads now hold the top K elements in their first item slot (`[0]`).
  // Write the final top K results to global memory. This write is naturally balanced
  // across the first K threads.
  if (threadIdx.x < K) {
    int offset = (batch_idx * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

// Max number of elements in intermediate_scores or intermediate_indices.
inline size_t GetHybridSortIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  return static_cast<size_t>(batch_size) * num_partitions * kHybridSortMaxK;
}

void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out,
                          int vocab_size, int batch_size, int k, float temperature, int partition_size) {
  constexpr int max_k = kHybridSortMaxK;
  constexpr int block_size = 256;
  static_assert(max_k <= block_size);

  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  switch (partition_size) {
    case 1024:
      FindBlockTopK_CubRegisterSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 2048:
      FindBlockTopK_CubRegisterSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 4096:
      FindBlockTopK_CubRegisterSort<block_size, 4096, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 8192:  // for vocab_size > 256 * 1024
      FindBlockTopK_CubRegisterSort<block_size, 8192, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    default:
      assert(false && "Unsupported partition_size");
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Reduce Phase
  int current_num_partitions = num_partitions;
  float* input_scores = data->intermediate_scores_1.get();
  int* input_indices = data->intermediate_indices.get();
  float* output_scores = data->intermediate_scores_2.get();
  int* output_indices = data->topk_indices.get();  // Re-use final output buffer for intermediate indices

  while (current_num_partitions > 1) {
    constexpr int partitions_per_block = 8;
    int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;

    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);

    bitonic::reduction::BlockReduceTopK<block_size, max_k, partitions_per_block><<<grid_reduce, block_reduce, 0, stream>>>(
        input_scores, input_indices, output_scores, output_indices, current_num_partitions);
    CUDA_CHECK(cudaGetLastError());

    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);

    current_num_partitions = num_blocks;
  }

  float* final_reduced_scores = input_scores;
  int* final_reduced_indices = input_indices;

  // --- FIX ---
  // The reduction phase uses a ping-pong buffer scheme. `final_reduced_scores` can
  // point to either `intermediate_scores_1` or `intermediate_scores_2`.
  // The subsequent Top-P sampling stage *requires* the raw scores to be
  // in `intermediate_scores_1`. We must ensure this condition is met.
  if (final_reduced_scores != data->intermediate_scores_1.get()) {
    // The final results are in the temp buffer, so copy them back to the primary buffer.
    // The size is batch_size * max_k because the reduction output has that stride.
    cudaMemcpyAsync(data->intermediate_scores_1.get(), final_reduced_scores,
                    static_cast<size_t>(batch_size) * max_k * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    final_reduced_scores = data->intermediate_scores_1.get();  // Update pointer to the canonical buffer
  }
  // --- END FIX ---

  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out, final_reduced_scores, final_reduced_indices,
                                 k, batch_size, max_k, temperature);
}

}  // namespace cuda
}  // namespace Generators
