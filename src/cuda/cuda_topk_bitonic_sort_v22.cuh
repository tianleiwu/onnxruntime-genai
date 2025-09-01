// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <cub/config.cuh>
#include <float.h>

namespace Generators {
namespace cuda {
namespace bitonic_v22 {
static const char* kAlgoDescription = "Bitonic v22 (Tiled Sort in Shared Memory)";
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
      out[i] = {-FLT_MAX, -1};
    }
  }
}


// Performs a stable bitonic sort in shared memory for `SortSize` elements.
template <int kTileSize, int kThreadsPerTile>
__device__ void TileBitonicSort(KeyValue* tile_data, int thread_in_tile) {
  // Phase 1: Create a single bitonic sequence of size kTileSize.
  for (int k = 2; k <= kTileSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = thread_in_tile; i < kTileSize; i += kThreadsPerTile) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          bool is_greater = (tile_data[i].score > tile_data[ixj].score) ||
                            (tile_data[i].score == tile_data[ixj].score && tile_data[i].index < tile_data[ixj].index);

          if (is_greater != ascending) {
            KeyValue temp = tile_data[i];
            tile_data[i] = tile_data[ixj];
            tile_data[ixj] = temp;
          }
        }
      }
      __syncwarp();
    }
  }

  // Phase 2: Sort the single bitonic sequence into descending order.
  for (int j = kTileSize >> 1; j > 0; j >>= 1) {
    for (int i = thread_in_tile; i < kTileSize; i += kThreadsPerTile) {
       int ixj = i ^ j;
       if (ixj > i) {
        if ((tile_data[i].score < tile_data[ixj].score) ||
            (tile_data[i].score == tile_data[ixj].score && tile_data[i].index > tile_data[ixj].index)) {
          KeyValue temp = tile_data[i];
          tile_data[i] = tile_data[ixj];
          tile_data[ixj] = temp;
        }
       }
    }
    __syncwarp();
  }
}


// Top-K kernel using a tiled sort in shared memory.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_TiledSort(const float* __restrict__ scores_in,
                                        int* __restrict__ intermediate_indices,
                                        float* __restrict__ intermediate_scores,
                                        int vocab_size,
                                        int num_partitions) {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    constexpr int WarpsPerBlock = kBlockSize / WARP_THREADS;
    constexpr int kTileSize = kPartitionSize / WarpsPerBlock; // Each warp handles one tile
    
    __shared__ KeyValue smem_partition[kPartitionSize];
    __shared__ KeyValue smem_final_candidates[K];

    const int batch_idx = blockIdx.y;
    const int partition_idx = blockIdx.x;
    const int partition_start = partition_idx * kPartitionSize;
    const float* batch_scores_in = scores_in + batch_idx * vocab_size;

    // 1. Coalesced load from global to registers, then write to shared memory (blocked layout)
    for (int i = 0; i < ItemsPerThread; ++i) {
        int global_idx = partition_start + threadIdx.x * ItemsPerThread + i;
        int smem_idx = threadIdx.x * ItemsPerThread + i;
        if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
            smem_partition[smem_idx] = {batch_scores_in[global_idx], global_idx};
        } else {
            smem_partition[smem_idx] = {-FLT_MAX, -1};
        }
    }
    __syncthreads();

    // 2. Each warp sorts its own tile of the partition in shared memory
    int warp_id = threadIdx.x / WARP_THREADS;
    int thread_in_warp = threadIdx.x % WARP_THREADS;
    KeyValue* my_tile = smem_partition + warp_id * kTileSize;
    TileBitonicSort<kTileSize, WARP_THREADS>(my_tile, thread_in_warp);
    __syncthreads();

    // 3. One warp (warp 0) merges the top K from each tile
    if (warp_id == 0) {
        // First, load the top K from the first tile into a temporary buffer
        KeyValue current_top[K];
        for (int i = thread_in_warp; i < K; i += WARP_THREADS) {
             if (i < kTileSize) { // Guard against K > kTileSize
                current_top[i] = smem_partition[i];
             } else {
                current_top[i] = {-FLT_MAX, -1};
             }
        }

        // Sequentially merge the top K from remaining tiles
        for (int w = 1; w < WarpsPerBlock; ++w) {
            KeyValue neighbor_top[K];
            for (int i = thread_in_warp; i < K; i += WARP_THREADS) {
                if (i < kTileSize) {
                    neighbor_top[i] = smem_partition[w * kTileSize + i];
                } else {
                    neighbor_top[i] = {-FLT_MAX, -1};
                }
            }
            
            KeyValue merged_top[K];
            Merge(current_top, K, neighbor_top, K, merged_top, K);

            for (int i = thread_in_warp; i < K; i += WARP_THREADS) {
                current_top[i] = merged_top[i];
            }
        }
        
        // Write final result to a designated spot in shared memory
        for (int i = thread_in_warp; i < K; i += WARP_THREADS) {
            smem_final_candidates[i] = current_top[i];
        }
    }
    __syncthreads();

    // 4. Write the final sorted top K to global memory
    if (threadIdx.x < K) {
        int offset = (batch_idx * num_partitions + partition_idx) * K;
        intermediate_scores[offset + threadIdx.x] = smem_final_candidates[threadIdx.x].score;
        intermediate_indices[offset + threadIdx.x] = smem_final_candidates[threadIdx.x].index;
    }
}


void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int partition_size_param) {
  const int max_k = kBitonicSortMaxK;
  constexpr int block_size = 256;

  const int num_partitions_effective = (vocab_size + partition_size_param - 1) / partition_size_param;
  dim3 grid_stage1(num_partitions_effective, batch_size);
  dim3 block_stage1(block_size);
  
  switch (partition_size_param) {
    case 512: // Minimum partition size that gives each warp a reasonable tile size
      FindBlockTopK_TiledSort<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 1024:
      FindBlockTopK_TiledSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 2048:
      FindBlockTopK_TiledSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    default:
      // Fallback to v19 for smaller partition sizes where tiling provides less benefit.
      bitonic_v19::FindBlockTopK_Coalesced_FullSort<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
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

} // namespace bitonic_v22
}  // namespace cuda
}  // namespace Generators

