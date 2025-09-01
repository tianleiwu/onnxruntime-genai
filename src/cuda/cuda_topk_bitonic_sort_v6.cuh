// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_helper.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>

namespace Generators {
namespace cuda {
namespace bitonic_v6 { // chunk permutation
static const char* kAlgoDescription = "Bitonic v6 (chunk permutation)";

// --- START: Device Helper Functions ---
// Performs a full bitonic sort in shared memory for `SortSize` elements.
// The final result is sorted in descending order.
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
        // Stable sort condition for final descending sort.
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
      // #pragma unroll
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
    // #pragma unroll
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


template <int kBlockSize, int kSortSize>
__global__ void FindBlockTopK_BitonicSort(const float* scores_in,
                                          int* intermediate_indices,
                                          float* intermediate_scores,
                                          int vocab_size,
                                          int num_partitions) {
  __shared__ float smem_scores[kSortSize];
  __shared__ int smem_indices[kSortSize];

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;
  const int partition_size = (vocab_size + num_partitions - 1) / num_partitions;
  const int partition_start = partition_idx * partition_size;

  constexpr int ElementsPerThread = kSortSize / kBlockSize;
  float reg_scores[ElementsPerThread];
  int reg_indices[ElementsPerThread];

  // Load data from global memory into registers
  for (int i = 0; i < ElementsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < partition_start + partition_size && global_idx < vocab_size) {
        reg_scores[i] = batch_scores_in[global_idx];
        reg_indices[i] = global_idx;
    } else {
        reg_scores[i] = -std::numeric_limits<float>::max();
        reg_indices[i] = -1;
    }
  }

  // Sort the elements within registers
  RegisterBitonicSort<ElementsPerThread>(reg_scores, reg_indices);
  
  // Write the sorted chunks from registers to shared memory
  for (int i = 0; i < ElementsPerThread; ++i) {
    smem_scores[threadIdx.x + i * kBlockSize] = reg_scores[i];
    smem_indices[threadIdx.x + i * kBlockSize] = reg_indices[i];
  }
  __syncthreads();

  // Merge the pre-sorted chunks in shared memory
  for (int k = ElementsPerThread * 2; k <= kSortSize; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
          for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
              int ixj = i ^ j;
              if (ixj > i) {
                  bool ascending = ((i & k) == 0);
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

  // Sort the final bitonic sequence descending
  for (int j = kSortSize >> 1; j > 0; j >>= 1) {
    for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
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

  // Have the first `max_k` threads write out the top results
  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

// "Mega Merge" FUSED REDUCTION KERNEL: Each block reduces multiple partitions in one go.
template <int kBlockSize, int K, int PartitionsPerBlock>
__global__ void BlockReduceTopK(const float* scores_in, const int* indices_in,
                                float* scores_out, int* indices_out,
                                int num_partitions_in) {
  constexpr int SortSize = K * PartitionsPerBlock;
  __shared__ float smem_scores[SortSize];
  __shared__ int smem_indices[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;

  // Determine how many partitions this block actually needs to process.
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const int in_base_offset = batch_idx * num_partitions_in * K;
  const int out_base_offset = (batch_idx * gridDim.x + blockIdx.x) * K;

  // Step 1: Parallel load of all necessary partitions into shared memory.
  for (int i = threadIdx.x; i < K * num_partitions_to_process; i += kBlockSize) {
    int partition_idx = i / K;
    int element_idx = i % K;
    int global_offset = in_base_offset + (block_start_partition + partition_idx) * K + element_idx;
    smem_scores[i] = scores_in[global_offset];
    smem_indices[i] = indices_in[global_offset];
  }

  // Pad the rest of the shared memory if this block has fewer partitions than the max.
  for (int i = K * num_partitions_to_process + threadIdx.x; i < SortSize; i += kBlockSize) {
      smem_scores[i] = -std::numeric_limits<float>::max();
      smem_indices[i] = -1;
  }
  __syncthreads();

  // Step 2: Perform a single, large sort on all loaded candidates.
  SharedMemBitonicSort<kBlockSize, SortSize>(smem_scores, smem_indices);

  // Step 3: Write the final top K result for this block to global memory.
  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_indices[threadIdx.x];
    scores_out[out_base_offset + threadIdx.x] = smem_scores[threadIdx.x];
  }
}


inline void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  const int max_k = kBitonicSortMaxK; // The fixed size of intermediate results
  constexpr int block_size = 256;

  // Stage 1: Map Phase - Find top-k within each partition of the vocabulary.
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  switch (sort_size) {
    case 256:
      FindBlockTopK_BitonicSort<block_size, 256><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 512:
      FindBlockTopK_BitonicSort<block_size, 512><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 1024:
      FindBlockTopK_BitonicSort<block_size, 1024><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 2048:
      FindBlockTopK_BitonicSort<block_size, 2048><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    case 4096:
      FindBlockTopK_BitonicSort<block_size, 4096><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions); 
      break;
    default:
      assert(false && "Unsupported sort_size"); 
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

} // namespace bitonic_v6
}  // namespace cuda
}  // namespace Generators