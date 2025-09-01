// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <cub/block/block_radix_sort.cuh>
#include <float.h>

namespace Generators {
namespace cuda {
namespace bitonic_v23 {
static const char* kAlgoDescription = "Bitonic v23 (CUB Register Sort)";

// A simple Key-Value struct for sorting.
struct KeyValue {
  float key;
  int value;
};

// Custom comparison operator for sorting KeyValue pairs in descending order.
struct CompareKeys {
  __device__ __forceinline__ bool operator()(const KeyValue& a, const KeyValue& b) const {
    if (a.key > b.key) return true;
    if (a.key < b.key) return false;
    return a.value < b.value;
  }
};

// Top-K kernel using cub::BlockRadixSort directly on register data.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CubRegisterSort(const float* __restrict__ scores_in,
                                              int* __restrict__ intermediate_indices,
                                              float* __restrict__ intermediate_scores,
                                              int vocab_size,
                                              int num_partitions) {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

    // Specialize BlockRadixSort for our KeyValue struct.
    // It will operate on data held in registers.
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
    // The result is a sorted, striped layout across the threads.
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // 3. The first K threads now hold the top K elements in their first item.
    // Write the final top K results to global memory.
    if (threadIdx.x < K) {
        int offset = (batch_idx * num_partitions + partition_idx) * K;
        intermediate_scores[offset + threadIdx.x] = thread_keys[0];
        intermediate_indices[offset + threadIdx.x] = thread_values[0];
    }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int partition_size_param) {
  const int max_k = kBitonicSortMaxK;
  constexpr int block_size = 256;

  const int num_partitions_effective = (vocab_size + partition_size_param - 1) / partition_size_param;
  dim3 grid_stage1(num_partitions_effective, batch_size);
  dim3 block_stage1(block_size);
  
  switch (partition_size_param) {
    case 256:
      FindBlockTopK_CubRegisterSort<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 512:
      FindBlockTopK_CubRegisterSort<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 1024:
      FindBlockTopK_CubRegisterSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 2048:
      FindBlockTopK_CubRegisterSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    default:
      // Should not be reached given the test configurations.
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Reduce Phase
  int current_num_partitions = num_partitions_effective;
  float* input_scores = data->scores_buffer.get();
  int* input_indices = data->indices_in.get();
  float* output_scores = data->scores_temp.get();
  int* output_indices = data->indices_sorted.get();

  while (current_num_partitions > 1) {
    constexpr int partitions_per_block = 8;
    int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;
    
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);
    
    // Using v19's efficient reduction kernel
    bitonic_v19::reduction::BlockReduceTopK<block_size, max_k, partitions_per_block><<<grid_reduce, block_reduce, 0, stream>>>(
        input_scores, input_indices,
        output_scores, output_indices,
        current_num_partitions);
    CUDA_CHECK(cudaGetLastError());

    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);

    current_num_partitions = num_blocks;
  }

  float* final_reduced_scores = input_scores;
  int* final_reduced_indices = input_indices;

  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out,
                                 final_reduced_scores, final_reduced_indices,
                                 k, batch_size, max_k, temperature);
}

} // namespace bitonic_v23
}  // namespace cuda
}  // namespace Generators
