// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {

// Kernel to compact strided data into a dense layout.
template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
    const int batch_idx = blockIdx.x;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int in_idx = batch_idx * input_stride + i;
        int out_idx = batch_idx * k + i;
        output[out_idx] = input[in_idx];
    }
}


template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CubRegisterSort(const float* __restrict__ scores_in,
                                              int* __restrict__ intermediate_indices,
                                              float* __restrict__ intermediate_scores,
                                              int vocab_size,
                                              int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;

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

  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  if (threadIdx.x < K) {
    int offset = (batch_idx * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

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

  // Stage 1: Find Top-K within partitions
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
    case 8192:
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
  // FIX: Use a dedicated intermediate buffer for indices to prevent race condition.
  int* output_indices = data->topk_indices_intermediate.get();

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

  if (final_reduced_scores != data->intermediate_scores_1.get()) {
    cudaMemcpyAsync(data->intermediate_scores_1.get(), final_reduced_scores,
                    static_cast<size_t>(batch_size) * max_k * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }

  // The final reduced indices are now in `final_reduced_indices`. We must copy them to the canonical `topk_indices` buffer.
  if (final_reduced_indices != data->topk_indices.get()){
      cudaMemcpyAsync(data->topk_indices.get(), final_reduced_indices,
                  static_cast<size_t>(batch_size) * max_k * sizeof(int), cudaMemcpyDeviceToDevice, stream);
  }
  
  // At this point, intermediate_scores_1 (raw logits) and topk_indices (indices) are both strided by max_k.
  final_reduced_scores = data->intermediate_scores_1.get();
  final_reduced_indices = data->topk_indices.get();

  // ApplySoftmaxToSortedTopK reads the strided inputs and produces COMPACT outputs.
  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out, final_reduced_scores, final_reduced_indices,
                                 k, batch_size, max_k, temperature);

  // FIX: The raw logits in `intermediate_scores_1` are still strided by `max_k`, but the sampling
  // pipeline expects them to be compact. We compact them here.
  CompactStridedData<float><<<batch_size, 256, 0, stream>>>(
      final_reduced_scores,
      data->intermediate_scores_2.get(),
      k, batch_size, max_k);

  // Copy the compacted raw logits back into the canonical buffer.
  cudaMemcpyAsync(data->intermediate_scores_1.get(), data->intermediate_scores_2.get(),
                  static_cast<size_t>(batch_size) * k * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
}

}  // namespace cuda
}  // namespace Generators