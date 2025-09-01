// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <cub/config.cuh>
#include <float.h>

namespace Generators {
namespace cuda {
namespace bitonic_v21 {
static const char* kAlgoDescription = "Bitonic v21 (Hierarchical Lock-Free Merge)";
constexpr int WARP_THREADS = 32;
// --- START: Device Helper Functions ---

struct KeyValue {
  float score;
  int index;
};

// Merges two sorted lists of KeyValue pairs.
__device__ inline void Merge(const KeyValue* a, int a_len, const KeyValue* b, int b_len, KeyValue* out, int k) {
  int a_ptr = 0;
  int b_ptr = 0;
  for (int i = 0; i < k; ++i) {
    bool a_is_better = (a_ptr < a_len) && 
                       ((b_ptr >= b_len) || (a[a_ptr].score > b[b_ptr].score) || 
                       (a[a_ptr].score == b[b_ptr].score && a[a_ptr].index < b[b_ptr].index));

    if (a_is_better) {
      out[i] = a[a_ptr++];
    } else if (b_ptr < b_len) {
      out[i] = b[b_ptr++];
    } else {
      // Should not happen if a_len + b_len >= k
      out[i] = {-FLT_MAX, -1};
    }
  }
}

// Sorts a small array of KeyValue pairs within a thread's registers.
template <int N>
__device__ inline void RegisterSort(KeyValue arr[N]) {
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      bool should_swap = (arr[j].score > arr[i].score) || 
                         (arr[j].score == arr[i].score && arr[j].index < arr[i].index);
      if (should_swap) {
        KeyValue temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }
}

// Top-K kernel using a lock-free, hierarchical merge approach.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_Hierarchical(const float* __restrict__ scores_in,
                                           int* __restrict__ intermediate_indices,
                                           float* __restrict__ intermediate_scores,
                                           int vocab_size,
                                           int num_partitions) {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    constexpr int WarpsPerBlock = kBlockSize / WARP_THREADS;

    __shared__ KeyValue smem_warp_topk[WarpsPerBlock][K];

    const int batch_idx = blockIdx.y;
    const int partition_idx = blockIdx.x;
    const int partition_start = partition_idx * kPartitionSize;
    const float* batch_scores_in = scores_in + batch_idx * vocab_size;

    // 1. Coalesced load into registers
    KeyValue thread_data[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i) {
        int global_idx = partition_start + threadIdx.x + i * kBlockSize;
        if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
            thread_data[i] = {batch_scores_in[global_idx], global_idx};
        } else {
            thread_data[i] = {-FLT_MAX, -1};
        }
    }
    
    // 2. Per-thread sort in registers
    RegisterSort<ItemsPerThread>(thread_data);

    // 3. Hierarchical Merge Reduction
    // First level: Merge within each warp
    KeyValue current_top[K];
    Merge(thread_data, ItemsPerThread, nullptr, 0, current_top, K);

    for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
        KeyValue neighbor_top[K];
        // Receive neighbor's data via shuffle
        for (int i = 0; i < K; ++i) {
            neighbor_top[i].score = __shfl_sync(0xFFFFFFFF, current_top[i].score, (threadIdx.x ^ offset) % WARP_THREADS);
            neighbor_top[i].index = __shfl_sync(0xFFFFFFFF, current_top[i].index, (threadIdx.x ^ offset) % WARP_THREADS);
        }
        
        KeyValue merged_top[K];
        Merge(current_top, K, neighbor_top, K, merged_top, K);
        
        for (int i=0; i<K; ++i) {
            current_top[i] = merged_top[i];
        }
    }

    // Lane 0 of each warp writes its results to shared memory
    if ((threadIdx.x % WARP_THREADS) == 0) {
        int warp_id = threadIdx.x / WARP_THREADS;
        for (int i = 0; i < K; ++i) {
            smem_warp_topk[warp_id][i] = current_top[i];
        }
    }
    __syncthreads();

    // Second level: A single warp merges the results from all warps
    if (threadIdx.x < WARP_THREADS) {
        int warp_id_to_merge = threadIdx.x;
        KeyValue my_warp_topk[K];
        
        if (warp_id_to_merge < WarpsPerBlock) {
            for(int i=0; i<K; ++i) {
                my_warp_topk[i] = smem_warp_topk[warp_id_to_merge][i];
            }
        } else {
            for(int i=0; i<K; ++i) {
                my_warp_topk[i] = {-FLT_MAX, -1};
            }
        }
        
        for (int offset = 1; offset < WarpsPerBlock; offset *= 2) {
            KeyValue neighbor_top[K];
            for (int i = 0; i < K; ++i) {
                neighbor_top[i].score = __shfl_sync(0xFFFFFFFF, my_warp_topk[i].score, threadIdx.x ^ offset);
                neighbor_top[i].index = __shfl_sync(0xFFFFFFFF, my_warp_topk[i].index, threadIdx.x ^ offset);
            }

            KeyValue merged_top[K];
            Merge(my_warp_topk, K, neighbor_top, K, merged_top, K);

            for (int i=0; i<K; ++i) {
                my_warp_topk[i] = merged_top[i];
            }
        }

        // Final result is in lane 0 of the first warp
        if (threadIdx.x == 0) {
            for(int i=0; i<K; ++i) {
                smem_warp_topk[0][i] = my_warp_topk[i];
            }
        }
    }
    __syncthreads();

    // 4. Write the final sorted top K to global memory
    if (threadIdx.x < K) {
        int offset = (batch_idx * num_partitions + partition_idx) * K;
        intermediate_scores[offset + threadIdx.x] = smem_warp_topk[0][threadIdx.x].score;
        intermediate_indices[offset + threadIdx.x] = smem_warp_topk[0][threadIdx.x].index;
    }
}


// Reduction kernel remains the same as it's already efficient.
namespace reduction {
    template <int kBlockSize, int SortSize, typename KeyValueT>
    __device__ void SharedMemBitonicSort(KeyValueT* smem_data);

    template <int kBlockSize, int K, int PartitionsPerBlock>
    __global__ void BlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                    float* __restrict__ scores_out, int* __restrict__ indices_out,
                                    int num_partitions_in);
}


void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int partition_size_param) {
  const int max_k = kBitonicSortMaxK;
  constexpr int block_size = 256;

  const int num_partitions_effective = (vocab_size + partition_size_param - 1) / partition_size_param;
  dim3 grid_stage1(num_partitions_effective, batch_size);
  dim3 block_stage1(block_size);
  
  switch (partition_size_param) {
    case 256:
      FindBlockTopK_Hierarchical<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 512:
      FindBlockTopK_Hierarchical<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 1024:
      FindBlockTopK_Hierarchical<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 2048:
      FindBlockTopK_Hierarchical<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
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

} // namespace bitonic_v21
}  // namespace cuda
}  // namespace Generators
