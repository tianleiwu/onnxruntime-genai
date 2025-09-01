// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>

namespace Generators {
namespace cuda {
namespace bitonic_v13 {
static const char* kAlgoDescription = "Bitonic v13 (Tiled FindBlockTopK with Shared Memory Merge)";

// --- START: Device Helper Functions ---

// Performs a full bitonic sort in shared memory for `SortSize` elements.
// The final result is sorted in descending order. This is a robust block-wide sorter.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(float* smem_scores, int* smem_indices) {
  // Phase 1: Create a single bitonic sequence of size SortSize.
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          
          // Stable sort condition: if scores are equal, use index as tie-breaker (smaller index is better).
          bool is_greater = (smem_scores[i] > smem_scores[ixj]) ||
                            (smem_scores[i] == smem_scores[ixj] && smem_indices[i] < smem_indices[ixj]);

          if (is_greater != ascending) {
            float temp_s = smem_scores[i]; smem_scores[i] = smem_scores[ixj]; smem_scores[ixj] = temp_s;
            int temp_i = smem_indices[i]; smem_indices[i] = smem_indices[ixj]; smem_indices[ixj] = temp_i;
          }
        }
      }
      __syncthreads();
    }
  }

  // Phase 2: Sort the single bitonic sequence into descending order.
  for (int j = SortSize >> 1; j > 0; j >>= 1) {
    for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
       int ixj = i ^ j;
       if (ixj > i) {
        if ((smem_scores[i] < smem_scores[ixj]) ||
            (smem_scores[i] == smem_scores[ixj] && smem_indices[i] > smem_indices[ixj])) {
          float temp_s = smem_scores[i]; smem_scores[i] = smem_scores[ixj]; smem_scores[ixj] = temp_s;
          int temp_i = smem_indices[i]; smem_indices[i] = smem_indices[ixj]; smem_indices[ixj] = temp_i;
        }
       }
    }
    __syncthreads();
  }
}

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

// --- END: Device Helper Functions ---

// This kernel finds the Top K elements within a large partition by processing it in smaller, fixed-size tiles.
// This avoids "over-sorting" by sorting small tiles and merging the results, instead of sorting the whole large partition.
template <int kBlockSize, int kTileSize, int K>
__global__ void FindBlockTopK_Tiled(const float* __restrict__ scores_in,
                                    int* __restrict__ intermediate_indices,
                                    float* __restrict__ intermediate_scores,
                                    int vocab_size,
                                    int num_partitions) {
  // Shared memory for one tile and for the running Top-K candidates.
  // The candidates buffer is 2*K to facilitate merging.
  __shared__ float smem_tile_scores[kTileSize];
  __shared__ int smem_tile_indices[kTileSize];
  __shared__ float smem_candidates_scores[2 * K];
  __shared__ int smem_candidates_indices[2 * K];

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;
  const int partition_size = (vocab_size + num_partitions - 1) / num_partitions;
  const int partition_start = partition_idx * partition_size;
  const int partition_end = min(partition_start + partition_size, vocab_size);

  // Initialize the candidate buffer with -inf
  for (int i = threadIdx.x; i < 2 * K; i += kBlockSize) {
    smem_candidates_scores[i] = -std::numeric_limits<float>::max();
    smem_candidates_indices[i] = -1;
  }
  __syncthreads();

  // Process the partition tile by tile
  for (int tile_start = partition_start; tile_start < partition_end; tile_start += kTileSize) {
    // Load a tile of data into shared memory
    for (int i = threadIdx.x; i < kTileSize; i += kBlockSize) {
      int global_idx = tile_start + i;
      if (global_idx < partition_end) {
        smem_tile_scores[i] = batch_scores_in[global_idx];
        smem_tile_indices[i] = global_idx;
      } else {
        smem_tile_scores[i] = -std::numeric_limits<float>::max();
        smem_tile_indices[i] = -1;
      }
    }
    __syncthreads();

    // Sort the tile
    SharedMemBitonicSort<kBlockSize, kTileSize>(smem_tile_scores, smem_tile_indices);

    // Merge the top K from the tile with the current candidates
    // The first K threads copy the top K from the sorted tile into the second half of the candidate buffer
    if (threadIdx.x < K) {
      smem_candidates_scores[K + threadIdx.x] = smem_tile_scores[threadIdx.x];
      smem_candidates_indices[K + threadIdx.x] = smem_tile_indices[threadIdx.x];
    }
    __syncthreads();

    // Sort the combined 2*K candidates. The new top K will be in the first half.
    SharedMemBitonicSort<kBlockSize, 2 * K>(smem_candidates_scores, smem_candidates_indices);
  }

  // Write the final top K for this partition to global memory
  if (threadIdx.x < K) {
    int offset = (batch_idx * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = smem_candidates_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_candidates_indices[threadIdx.x];
  }
}

// This reduction kernel is based on v10's efficient design. It fuses the reduction of multiple
// partitions into a single block, using register-level sorting for initial speed and then a
// shared memory sort to merge the results.
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

  // Step 1: Parallel load into registers.
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

  // Step 2: Sort within registers.
  RegisterBitonicSort<ElementsPerThread>(reg_scores, reg_indices);

  // Step 3: Write sorted chunks to shared memory.
  for (int i = 0; i < ElementsPerThread; ++i) {
    smem_scores[threadIdx.x + i * kBlockSize] = reg_scores[i];
    smem_indices[threadIdx.x + i * kBlockSize] = reg_indices[i];
  }
  __syncthreads();
  
  // Step 4: Merge pre-sorted chunks in shared memory.
  SharedMemBitonicSort<kBlockSize, SortSize>(smem_scores, smem_indices);

  // Step 5: Write final top K result for this block to global memory.
  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_indices[threadIdx.x];
    scores_out[out_base_offset + threadIdx.x] = smem_scores[threadIdx.x];
  }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int tile_size) {
  const int max_k = kBitonicSortMaxK; // The fixed size of intermediate results
  constexpr int block_size = 256;

  // Stage 1: Map Phase - Find top-k within each partition of the vocabulary using the tiled approach.
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);
  
  switch (tile_size) {
    case 256:
      FindBlockTopK_Tiled<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 512:
      FindBlockTopK_Tiled<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 1024:
      FindBlockTopK_Tiled<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 2048:
      FindBlockTopK_Tiled<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 4096:
      FindBlockTopK_Tiled<block_size, 4096, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    default:
      // For the specific use case, a smaller tile size is better to avoid over-sorting.
      // We fall back to 256 if an unsupported size is given.
      FindBlockTopK_Tiled<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Reduce Phase - Fused iterative reduction
  int current_num_partitions = num_partitions;
  float* input_scores = data->scores_buffer.get();
  int* input_indices = data->indices_in.get();
  float* output_scores = data->scores_temp.get();
  int* output_indices = data->indices_sorted.get();

  while (current_num_partitions > 1) {
    constexpr int partitions_per_block = 8; // Reduction factor per block
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

  // Final results are now in the `input` buffers
  float* final_reduced_scores = input_scores;
  int* final_reduced_indices = input_indices;

  // Stage 3: Final Copy and Softmax
  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out,
                                 final_reduced_scores, final_reduced_indices,
                                 k, batch_size, max_k, temperature);
}

} // namespace bitonic_v13
}  // namespace cuda
}  // namespace Generators
