// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <cub/config.cuh> 

namespace Generators {
namespace cuda {
namespace bitonic_v17 {
static const char* kAlgoDescription = "Bitonic v17 (Corrected Shared Memory Sort)";

/*
Coalesced Load: The kernel retains the efficient, coalesced memory load pattern from global to registers, followed by a write to shared memory.

Analysis of v17 and Path to v18
The current v17 kernel (FindBlockTopK_CorrectedSort) has one primary inefficiency:

Memory Access Pattern: It loads the entire partition from global memory into shared memory. This involves a strided access pattern (e.g., thread 0 loads index 0, thread 1 loads index 256, etc.), which is not ideal for memory bandwidth as it's not coalesced.

Over-sorting: It still sorts the entire partition (e.g., 2048 elements) in shared memory just to find the top K (e.g., 64) elements.

The clear next step is to combine the superior memory access pattern of v10 with a more intelligent in-block reduction that avoids sorting the entire partition.

Strategy for v18: Coalesced Reads and Parallel Merge-Reduction

Coalesced Load to Registers: We will revert to the highly efficient memory load from v10. Each thread will read a contiguous chunk of ElementsPerThread from global memory directly into its private registers.

Per-Thread Sort: Each thread will sort its small set of items within registers. This is extremely fast.

Parallel Merge-Reduction: Instead of sorting the whole partition, the threads will cooperate to merge their small, sorted lists to find the block-wide top K. Each thread will post its top candidates to shared memory, and then the block will perform a tree-based merge, where pairs of threads merge their candidate lists at each step until thread 0 holds the final top K for the entire block.
*/

// --- START: Device Helper Functions ---

struct KeyValue {
  float score;
  int index;
};

// Sorts an array of `N` elements held in registers.
template <int N>
__device__ void RegisterBitonicSort(float scores[N], int indices[N]) {
  // Build the bitonic sequence
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          bool is_greater = (scores[i] > scores[ixj]) || (scores[i] == scores[ixj] && indices[i] < indices[ixj]);
          if (is_greater != ascending) {
            float temp_s = scores[i]; scores[i] = scores[ixj]; scores[ixj] = temp_s;
            int temp_i = indices[i]; indices[i] = indices[ixj]; indices[ixj] = temp_i;
          }
        }
      }
    }
  }

  // Sort the bitonic sequence descending
  for (int j = N >> 1; j > 0; j >>= 1) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      int ixj = i ^ j;
      if (ixj > i) {
        if ((scores[i] < scores[ixj]) || (scores[i] == scores[ixj] && indices[i] > indices[ixj])) {
          float temp_s = scores[i]; scores[i] = scores[ixj]; scores[ixj] = temp_s;
          int temp_i = indices[i]; indices[i] = indices[ixj]; indices[ixj] = temp_i;
        }
      }
    }
  }
}

// Performs a full bitonic sort in shared memory for `SortSize` elements.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(KeyValue* smem_data) {
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          KeyValue* a = &smem_data[i];
          KeyValue* b = &smem_data[ixj];
          bool is_greater = (a->score > b->score) || (a->score == b->score && a->index < b->index);
          if (is_greater != ascending) {
            KeyValue temp = *a; *a = *b; *b = temp;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int j = SortSize >> 1; j > 0; j >>= 1) {
    for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
       int ixj = i ^ j;
       if (ixj > i) {
        KeyValue* a = &smem_data[i];
        KeyValue* b = &smem_data[ixj];
        if ((a->score < b->score) || (a->score == b->score && a->index > b->index)) {
          KeyValue temp = *a; *a = *b; *b = temp;
        }
       }
    }
    __syncthreads();
  }
}


// Top-K kernel using a simple and correct shared memory bitonic sort.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CorrectedSort(const float* __restrict__ scores_in,
                                            int* __restrict__ intermediate_indices,
                                            float* __restrict__ intermediate_scores,
                                            int vocab_size,
                                            int num_partitions) {
    // constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    __shared__ KeyValue smem_buffer[kPartitionSize];

    const int batch_idx = blockIdx.y;
    const int partition_idx = blockIdx.x;
    const int partition_start = partition_idx * kPartitionSize;
    const float* batch_scores_in = scores_in + batch_idx * vocab_size;

    // 1. Load data into shared memory
    for (int i = threadIdx.x; i < kPartitionSize; i += kBlockSize) {
        int global_idx = partition_start + i;
        if (global_idx < partition_start + kPartitionSize && global_idx < vocab_size) {
            smem_buffer[i].score = batch_scores_in[global_idx];
            smem_buffer[i].index = global_idx;
        } else {
            smem_buffer[i].score = -std::numeric_limits<float>::max();
            smem_buffer[i].index = -1;
        }
    }
    __syncthreads();

    // 2. Sort the entire partition in shared memory
    SharedMemBitonicSort<kBlockSize, kPartitionSize>(smem_buffer);

    // 3. Write top K to global memory
    if (threadIdx.x < K) {
        int offset = (batch_idx * num_partitions + partition_idx) * K;
        intermediate_scores[offset + threadIdx.x] = smem_buffer[threadIdx.x].score;
        intermediate_indices[offset + threadIdx.x] = smem_buffer[threadIdx.x].index;
    }
}

// Reduction kernel from v10, which is correct and efficient.
template <int kBlockSize, int K, int PartitionsPerBlock>
__global__ void BlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                float* __restrict__ scores_out, int* __restrict__ indices_out,
                                int num_partitions_in) {
  constexpr int SortSize = K * PartitionsPerBlock;
  __shared__ float smem_scores[SortSize];
  __shared__ int smem_indices[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  constexpr int ElementsPerThread = SortSize / kBlockSize;
  float reg_scores[ElementsPerThread];
  int reg_indices[ElementsPerThread];

  const int in_base_offset = batch_idx * num_partitions_in * K;
  const int out_base_offset = (batch_idx * gridDim.x + blockIdx.x) * K;

  for (int i = 0; i < ElementsPerThread; ++i) {
    int smem_idx = threadIdx.x + i * kBlockSize;
    if (smem_idx < K * num_partitions_to_process) {
      int partition_idx = smem_idx / K;
      int element_idx = smem_idx % K;
      int global_offset = in_base_offset + (block_start_partition + partition_idx) * K + element_idx;
      reg_scores[i] = scores_in[global_offset];
      reg_indices[i] = indices_in[global_offset];
    } else {
      reg_scores[i] = -std::numeric_limits<float>::max();
      reg_indices[i] = -1;
    }
  }

  RegisterBitonicSort<ElementsPerThread>(reg_scores, reg_indices);

  for (int i = 0; i < ElementsPerThread; ++i) {
    smem_scores[threadIdx.x + i * kBlockSize] = reg_scores[i];
    smem_indices[threadIdx.x + i * kBlockSize] = reg_indices[i];
  }
  __syncthreads();
  
  // Need to sort KeyValue pairs here for the reduction
  __shared__ KeyValue reduce_smem_buffer[SortSize];
  for(int i = threadIdx.x; i < SortSize; i += kBlockSize) {
      reduce_smem_buffer[i].score = smem_scores[i];
      reduce_smem_buffer[i].index = smem_indices[i];
  }
  __syncthreads();
  
  SharedMemBitonicSort<kBlockSize, SortSize>(reduce_smem_buffer);

  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = reduce_smem_buffer[threadIdx.x].index;
    scores_out[out_base_offset + threadIdx.x] = reduce_smem_buffer[threadIdx.x].score;
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
      FindBlockTopK_CorrectedSort<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 512:
      FindBlockTopK_CorrectedSort<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 1024:
      FindBlockTopK_CorrectedSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 2048:
      FindBlockTopK_CorrectedSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    default:
      assert(false && "Unsupported partition_size"); 
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
    
    BlockReduceTopK<block_size, max_k, partitions_per_block><<<grid_reduce, block_reduce, 0, stream>>>(
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

} // namespace bitonic_v17
}  // namespace cuda
}  // namespace Generators
