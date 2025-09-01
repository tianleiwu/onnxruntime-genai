// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_topk_helper.h"
#include <float.h> // For FLT_MAX

namespace Generators {
namespace cuda {
namespace bitonic_v20 {
/*
A dead lock is caused by the spin-lock implementation (while (atomicCAS(&smem_lock, 0, 1) != 0);) combined with how threads are scheduled in warps.

A warp is a group of 32 threads that execute the same instruction at the same time. If one thread in a warp acquires the lock, the other 31 threads in that same warp that are also trying to acquire the lock will spin forever. They cannot make progress because the lock-holding thread is part of their warp, and it is waiting for the HeapSiftDown operation to complete. This creates a standstill within the warp, leading to the hang you observed.
*/
static const char* kAlgoDescription = "Bitonic v20 (Coalesced Read + Heap Selection)";

// --- START: Device Helper Functions ---

struct KeyValue {
  float score;
  int index;
};

// Heapify (sift-down) operation for the min-heap
__device__ void HeapSiftDown(KeyValue* heap, int i, int n) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    // In our min-heap, "smaller" means a lower score, or a higher index for ties.
    if (left < n && ((heap[left].score < heap[smallest].score) || 
                   (heap[left].score == heap[smallest].score && heap[left].index > heap[smallest].index))) {
        smallest = left;
    }

    if (right < n && ((heap[right].score < heap[smallest].score) ||
                    (heap[right].score == heap[smallest].score && heap[right].index > heap[smallest].index))) {
        smallest = right;
    }

    if (smallest != i) {
        KeyValue temp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = temp;
        HeapSiftDown(heap, smallest, n);
    }
}

// Build a min-heap from an array
__device__ void BuildMinHeap(KeyValue* heap, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        HeapSiftDown(heap, i, n);
    }
}

// Final sort for the top K elements before writing to global memory
template <int K>
__device__ void SortTopK(KeyValue* top_k_heap) {
    // This is essentially a heap sort which results in ascending order
    for (int i = K - 1; i > 0; i--) {
        KeyValue temp = top_k_heap[0];
        top_k_heap[0] = top_k_heap[i];
        top_k_heap[i] = temp;
        HeapSiftDown(top_k_heap, 0, i);
    }

    // Reverse the array to get descending order for TopK
    for (int i = 0; i < K / 2; i++) {
        KeyValue temp = top_k_heap[i];
        top_k_heap[i] = top_k_heap[K - 1 - i];
        top_k_heap[K - 1 - i] = temp;
    }
}

// Top-K kernel using coalesced reads and a parallel heap selection algorithm.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_HeapSelect(const float* __restrict__ scores_in,
                                         int* __restrict__ intermediate_indices,
                                         float* __restrict__ intermediate_scores,
                                         int vocab_size,
                                         int num_partitions) {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    __shared__ KeyValue smem_heap[K];
    __shared__ int smem_lock;

    const int batch_idx = blockIdx.y;
    const int partition_idx = blockIdx.x;
    const int partition_start = partition_idx * kPartitionSize;
    const float* batch_scores_in = scores_in + batch_idx * vocab_size;

    // 1. Coalesced load into registers
    KeyValue thread_data[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i) {
        int global_idx = partition_start + threadIdx.x * ItemsPerThread + i;
        if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
            thread_data[i] = {batch_scores_in[global_idx], global_idx};
        } else {
            thread_data[i] = {-FLT_MAX, -1};
        }
    }
    
    // 2. Initialize heap from the first K threads' first items
    if (threadIdx.x == 0) {
        smem_lock = 0;
    }
    __syncthreads();

    if (threadIdx.x < K) {
        smem_heap[threadIdx.x] = thread_data[0];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        BuildMinHeap(smem_heap, K);
    }
    __syncthreads();

    // 3. Each thread processes its remaining elements
    for (int i = 0; i < ItemsPerThread; ++i) {
        // Skip the first item from the first K threads as it was used for initialization
        if (threadIdx.x < K && i == 0) continue;

        float score = thread_data[i].score;
        int index = thread_data[i].index;

        if (score > -FLT_MAX) { // Only process valid items
            // If the current element is greater than the smallest element in the heap (non-atomic read)
            if (score > smem_heap[0].score || (score == smem_heap[0].score && index < smem_heap[0].index)) {
                
                // Acquire lock to enter critical section
                while (atomicCAS(&smem_lock, 0, 1) != 0);
                __threadfence_block();

                // Re-check condition inside critical section to ensure heap wasn't updated by another thread
                if (score > smem_heap[0].score || (score == smem_heap[0].score && index < smem_heap[0].index)) {
                    smem_heap[0].score = score;
                    smem_heap[0].index = index;
                    HeapSiftDown(smem_heap, 0, K);
                }

                // Release lock
                __threadfence_block();
                atomicExch(&smem_lock, 0);
            }
        }
    }
    __syncthreads();
    
    // 4. One thread sorts the final K elements in shared memory
    if (threadIdx.x == 0) {
        SortTopK<K>(smem_heap);
    }
    __syncthreads();
    
    // 5. Write the final sorted top K to global memory
    if (threadIdx.x < K) {
        int offset = (batch_idx * num_partitions + partition_idx) * K;
        intermediate_scores[offset + threadIdx.x] = smem_heap[threadIdx.x].score;
        intermediate_indices[offset + threadIdx.x] = smem_heap[threadIdx.x].index;
    }
}


// Reduction kernel remains the same as it's already efficient.
namespace reduction {
    template <int kBlockSize, int SortSize>
    __device__ void SharedMemBitonicSort(KeyValue* smem_data); // Forward declaration

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
      FindBlockTopK_HeapSelect<block_size, 256, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 512:
      FindBlockTopK_HeapSelect<block_size, 512, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 1024:
      FindBlockTopK_HeapSelect<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
      break;
    case 2048:
      FindBlockTopK_HeapSelect<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions_effective); 
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

} // namespace bitonic_v20
}  // namespace cuda
}  // namespace Generators

