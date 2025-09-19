// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>  // For FLT_MAX
#include "cuda_topk.h"
#include <cub/cub.cuh>
#include "cuda_topk_select_sort.cuh"

// Note: This file now implements a HYBRID reduction strategy.
// For k <= 4, it uses the original fast iterative selection method.
// For k > 4, it uses the more scalable shared-memory bitonic sort.
// This is controlled by the `MaxK` template parameter, which allows for
// optimized shared memory sizing.

namespace Generators {
namespace cuda {
namespace distributed_select_sort {
using TopK_Pair = selection_sort::TopK_Pair;

// A simple struct to hold a key-value pair for inter-TB reduction.
// Has an indirection element id (`elem_id_indirection`) to point to
// the actual element id in the original array whereas element id
// (`elem_id`) is the local element id with respect to the processing TB.
struct TopK_Pair_Reduce {
  float u = -FLT_MAX;
  int p = -1;
  int p_indirection = -1;

  __device__ __forceinline__ void insert(float elem, int elem_id, int elem_id_indirection) {
    if (elem > u || (elem == u && elem_id_indirection < p_indirection)) {
      u = elem;
      p = elem_id;
      p_indirection = elem_id_indirection;
    }
  }

  __device__ __forceinline__ void init() {
    u = -FLT_MAX;
    p = -1;
    p_indirection = -1;
  }
};

// CUB reduction operator for finding the maximum value and its index.
__device__ __forceinline__ TopK_Pair_Reduce reduce_topk_pair_reduce_op(TopK_Pair_Reduce const& a, TopK_Pair_Reduce const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p_indirection < b.p_indirection) ? a
                                                                           : b;
}

// Helper to calculate the next power of 2 for bitonic sort padding.
__device__ __forceinline__ int next_power_of_2(int n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

// TODO(hasesh): Experiment with using co-operative groups for synchronization in the following kernel
// to see if it can give tangible perf gains
template <int kBlockSize, int MaxK>
__global__ void GetTopKKernelDistributedSelectSort(float* scores_in, float* scores_out, int* indices_out,
                                                   int vocab_size, int k,
                                                   int* top_k_distributed_lock, int* distributed_indices_out,
                                                   float* distributed_scores_out) {
  int top_k_shard = blockIdx.z;
  int tid = threadIdx.x;

  int num_top_k_shards = gridDim.z;

  TopK_Pair partial;

  // Use a very small number to blank out selected scores, avoiding picking them again.
  constexpr float MIN_FLOAT = -std::numeric_limits<float>::max();

  int* distributed_indices_out_curr = distributed_indices_out + top_k_shard * k;
  float* distributed_scores_out_curr = distributed_scores_out + top_k_shard * k;

  int start_i = kBlockSize * top_k_shard + tid;

  for (int ite = 0; ite < k; ite++) {
    partial.Init();

    for (auto elemId = start_i; elemId < vocab_size; elemId += kBlockSize * num_top_k_shards) {
      float elem = scores_in[elemId];
      partial.Insert(elem, elemId);
    }

    // reduce in thread block
    typedef cub::BlockReduce<TopK_Pair, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_Pair top_k_sequence = BlockReduce(temp_storage).Reduce(partial, selection_sort::reduce_topk_op);

    if (tid == 0) {
      distributed_scores_out_curr[ite] = top_k_sequence.u;
      distributed_indices_out_curr[ite] = top_k_sequence.p;
      scores_in[top_k_sequence.p] = MIN_FLOAT;
      __threadfence_block();
    }
    __syncthreads();
  }

  // All TBs flush their data and it should be visible to every other TB
  __threadfence();
  __syncthreads();

  // Signal that each threadblock has done its work using elected thread
  if (threadIdx.x == 0) {
    atomicAdd(top_k_distributed_lock, 1);
  }

  // The reduction threadblock
  if (blockIdx.z == 0) {
    // Elected thread spins while waiting for other TBs to complete their work
    if (threadIdx.x == 0) {
      int count_of_completed_TBs = 0;
      asm volatile("ld.volatile.global.s32 %0, [%1];"
                   : "=r"(count_of_completed_TBs)
                   : "l"(top_k_distributed_lock));
      while (count_of_completed_TBs < num_top_k_shards) {
        asm volatile("ld.volatile.global.s32 %0, [%1];"
                     : "=r"(count_of_completed_TBs)
                     : "l"(top_k_distributed_lock));
      }

      // Reset the lock for next kernel call
      atomicExch(top_k_distributed_lock, 0);
    }
    __syncthreads();

    if constexpr (MaxK <= 4) {
      // --- ORIGINAL REDUCTION for k <= 4 ---
      // Low overhead, faster for very few items, with optimized shared memory.
      constexpr int kMaxCandidates = topk_impl_details::kTopKDistributedSelectSortMaxShards * MaxK;
      __shared__ float smem_scores[kMaxCandidates];
      __shared__ int smem_indices[kMaxCandidates];

      for (int i = threadIdx.x; i < num_top_k_shards * k; i += kBlockSize) {
        smem_scores[i] = distributed_scores_out[i];
        smem_indices[i] = distributed_indices_out[i];
      }
      __syncthreads();

      TopK_Pair_Reduce reduce_partial;
      for (int ite = 0; ite < k; ite++) {
        reduce_partial.init();
        for (int i = threadIdx.x; i < num_top_k_shards * k; i += kBlockSize) {
          reduce_partial.insert(smem_scores[i], i, smem_indices[i]);
        }
        typedef cub::BlockReduce<TopK_Pair_Reduce, kBlockSize> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        TopK_Pair_Reduce top_k_sequence_reduced = BlockReduce(temp_storage).Reduce(reduce_partial, reduce_topk_pair_reduce_op);

        if (tid == 0) {
          scores_out[ite] = top_k_sequence_reduced.u;
          indices_out[ite] = top_k_sequence_reduced.p_indirection;
          smem_scores[top_k_sequence_reduced.p] = MIN_FLOAT;
          __threadfence_block();
        }
        __syncthreads();
      }
    } else {
      // --- OPTIMIZED REDUCTION for k > 4 using Bitonic Sort ---
      // Higher overhead, but much more scalable and parallel.
      constexpr int kMaxCandidates = topk_impl_details::kTopKDistributedSelectSortMaxShards * MaxK;
      __shared__ float smem_scores[kMaxCandidates];
      __shared__ int smem_indices[kMaxCandidates];
      const int num_candidates = num_top_k_shards * k;

      for (int i = tid; i < num_candidates; i += kBlockSize) {
        smem_scores[i] = distributed_scores_out[i];
        smem_indices[i] = distributed_indices_out[i];
      }

      const int sort_size = next_power_of_2(num_candidates);
      for (int i = num_candidates + tid; i < sort_size; i += kBlockSize) {
        smem_scores[i] = -FLT_MAX;
        smem_indices[i] = INT_MAX;
      }
      __syncthreads();

      for (int outer_k = 2; outer_k <= sort_size; outer_k <<= 1) {
        for (int inner_j = outer_k >> 1; inner_j > 0; inner_j >>= 1) {
          for (int i = tid; i < sort_size; i += kBlockSize) {
            const int ixj = i ^ inner_j;
            if (ixj > i) {
              float a_i = smem_scores[i];
              float a_j = smem_scores[ixj];
              int idx_i = smem_indices[i];
              int idx_j = smem_indices[ixj];
              bool ascending = ((i & outer_k) == 0);
#if STABLE_TOPK
              bool is_i_greater = (a_i > a_j) || (a_i == a_j && idx_i < idx_j);
#else
              bool is_i_greater = a_i > a_j;
#endif
              if (is_i_greater != ascending) {
                smem_scores[i] = a_j;
                smem_scores[ixj] = a_i;
                smem_indices[i] = idx_j;
                smem_indices[ixj] = idx_i;
              }
            }
          }
          __syncthreads();
        }
      }
      __syncthreads();

      if (tid < k) {
        scores_out[tid] = smem_scores[tid];
        indices_out[tid] = smem_indices[tid];
      }
    }
  }
}

void LaunchGetDistributedSelectSortTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size,
                                        int k, int top_k_shards,
                                        int* top_k_distributed_lock, int* distributed_indices_out, float* distributed_scores_out) {
  const int block_size = 1024;
  dim3 grid(1, 1, top_k_shards);
  dim3 block(block_size, 1, 1);

  if (k <= 4) {
    GetTopKKernelDistributedSelectSort<block_size, 4><<<grid, block, 0, stream>>>(scores_in, scores_out, indices_out, vocab_size, k,
                                                                                  top_k_distributed_lock,
                                                                                  distributed_indices_out,
                                                                                  distributed_scores_out);
  } else {
    GetTopKKernelDistributedSelectSort<block_size, topk_impl_details::kTopKDistributedSelectSortMaxTopK><<<grid, block, 0, stream>>>(
        scores_in, scores_out, indices_out, vocab_size, k,
        top_k_distributed_lock,
        distributed_indices_out,
        distributed_scores_out);
  }
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));  // caller shall check IsSupported before calling this function.

  float* topk_scores = data->intermediate_scores_1;
  int* topk_indices = data->intermediate_indices_1;

  // Use the "copy-on-write" strategy as the input scores will be mutated by the stage 1 kernel
  float* mutable_scores = data->intermediate_scores_2;
  size_t buffer_size = vocab_size * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));
  LaunchGetDistributedSelectSortTopK(stream, mutable_scores, topk_scores, topk_indices, vocab_size, k,
                                     data->top_k_distributed_select_sort_shards,
                                     data->top_k_distributed_select_sort_lock,
                                     data->top_k_distributed_select_sort_keys,
                                     data->top_k_distributed_select_sort_values);
  CUDA_CHECK_LAUNCH();

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  return (batch_size <= topk_impl_details::kTopKDistributedSelectSortMaxBatchSize) &&
         (k <= topk_impl_details::kTopKDistributedSelectSortMaxTopK) &&
         (vocab_size >= topk_impl_details::kTopKDistributedSelectSortMinVocabSize);
}

}  // namespace distributed_select_sort
}  // namespace cuda
}  // namespace Generators

