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

// --- START: Added for debugging ---
// Helper function to print the first N elements of a device array.
template <typename T>
void DebugPrintDeviceData(cudaStream_t stream, const T* d_array, int n, int batch_size, int stride, const char* name) {
  int count = std::min(n * batch_size, 256); // Print at most 256 elements
  std::vector<T> h_array(count);
  CUDA_CHECK(cudaMemcpyAsync(h_array.data(), d_array, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("--- Debug Print: %s ---\n", name);
  for (int b = 0; b < batch_size; ++b) {
    if (b >= 2 && count > 128) continue; // Limit printing for large batches
    printf("Batch %d:\n", b);
    for (int i = 0; i < std::min(n, count); ++i) {
      if constexpr (std::is_same_v<T, float>) {
        printf("%d: %f, ", i, h_array[b * stride + i]);
      } else if constexpr (std::is_same_v<T, int>) {
        printf("%d: %d, ", i, h_array[b * stride + i]);
      }
    }
    printf("\n");
  }
  printf("----------------------------------------\n");
}
// --- END: Added for debugging ---

// --- START: New Device Helper Function ---
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
// --- END: New Device Helper Function ---


template <int kBlockSize, int kSortSize>
__global__ void FindBlockTopK_BitonicSort(const float* scores_in,
                                          int* intermediate_indices,
                                          float* intermediate_scores,
                                          int vocab_size,
                                          int num_partitions) {
  // Shared memory for sorting one partition. Its size must be a power of 2.
  __shared__ float smem_scores[kSortSize];
  __shared__ int smem_indices[kSortSize];

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;
  const int partition_size = (vocab_size + num_partitions - 1) / num_partitions;
  const int partition_start = partition_idx * partition_size;

  // Load data from global to shared memory
  for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
    int global_idx = partition_start + i;
    if (i < partition_size && global_idx < vocab_size) {
      smem_scores[i] = batch_scores_in[global_idx];
      smem_indices[i] = global_idx;
    } else {
      // Pad with minimum values to ensure they are sorted to the end
      smem_scores[i] = -std::numeric_limits<float>::max();
      smem_indices[i] = -1;
    }
  }
  __syncthreads();

  // Perform a full sort on the data in shared memory.
  SharedMemBitonicSort<kBlockSize, kSortSize>(smem_scores, smem_indices);

  // Have the first `max_k` threads write out the top results
  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

// NEW KERNEL for pairwise on-device reduction.
// Each thread block merges two sorted lists of size K into one sorted list of size K.
template <int kBlockSize, int K>
__global__ void PairwiseReduceTopK(const float* scores_in, const int* indices_in,
                                   float* scores_out, int* indices_out,
                                   int num_partitions_in) {
  const int batch_idx = blockIdx.y;
  const int output_partition_idx = blockIdx.x;

  // Each thread block handles the merge of two partitions into one.
  const int input_partition_idx1 = output_partition_idx * 2;
  const int input_partition_idx2 = output_partition_idx * 2 + 1;

  // Shared memory to hold the 2*K candidates from the two partitions.
  __shared__ float smem_scores[K * 2];
  __shared__ int smem_indices[K * 2];

  const int in_base_offset = batch_idx * num_partitions_in * K;
  const int num_output_partitions = (num_partitions_in + 1) / 2;
  const int out_base_offset = (batch_idx * num_output_partitions + output_partition_idx) * K;

  // Step 1: Cooperatively load the 2*K elements into shared memory.
  for (int i = threadIdx.x; i < K; i += kBlockSize) {
    // Load from first partition
    int p1_offset = in_base_offset + input_partition_idx1 * K + i;
    smem_scores[i] = scores_in[p1_offset];
    smem_indices[i] = indices_in[p1_offset];

    // Load from second partition (if it exists)
    if (input_partition_idx2 < num_partitions_in) {
      int p2_offset = in_base_offset + input_partition_idx2 * K + i;
      smem_scores[K + i] = scores_in[p2_offset];
      smem_indices[K + i] = indices_in[p2_offset];
    } else {
      // If there's an odd number of partitions, the last one is just passed through.
      // Pad with minimum values.
      smem_scores[K + i] = -std::numeric_limits<float>::max();
      smem_indices[K + i] = -1;
    }
  }
  __syncthreads();

  // Step 2: Perform a full bitonic sort on the 2*K elements in shared memory.
  SharedMemBitonicSort<kBlockSize, K * 2>(smem_scores, smem_indices);

  // Step 3: Write the top K results back to global memory.
  for (int i = threadIdx.x; i < K; i += kBlockSize) {
    indices_out[out_base_offset + i] = smem_indices[i];
    scores_out[out_base_offset + i] = smem_scores[i];
  }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  constexpr int block_size = 256;
  const int max_k = kBitonicSortMaxK; // The fixed size of intermediate results

  // Stage 1: Map Phase - Find top-k within each partition of the vocabulary.
  // This produces `num_partitions` sorted lists of size `max_k`.
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  // Use scores_buffer for intermediate scores to avoid conflicts with ping-pong buffers.
  switch (sort_size) {
    case 512:
      FindBlockTopK_BitonicSort<block_size, 512><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions);
      break;
    case 1024:
      FindBlockTopK_BitonicSort<block_size, 1024><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions);
      break;
    case 2048:
      FindBlockTopK_BitonicSort<block_size, 2048><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions);
      break;
    case 4096:
      FindBlockTopK_BitonicSort<block_size, 4096><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions);
      break;
    default:
      assert(false && "Unsupported sort_size");
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // // --- DEBUG ---
  // bool enable_debug_print = (batch_size == 1 && vocab_size >= 32000);
  // if (enable_debug_print) {
  //   DebugPrintDeviceData(stream, data->scores_buffer.get(), 64, batch_size, num_partitions * max_k, "After Stage 1 (Scores)");
  //   DebugPrintDeviceData(stream, data->indices_in.get(), 64, batch_size, num_partitions * max_k, "After Stage 1 (Indices)");
  // }

  // Stage 2: Reduce Phase - OPTIMIZED with iterative pairwise reduction
  int current_num_partitions = num_partitions;

  // Set up ping-pong buffers. The initial input is the output of Stage 1.
  float* input_scores = data->scores_buffer.get();
  int* input_indices = data->indices_in.get();
  // Use scores_temp and indices_sorted for the ping-pong outputs.
  float* output_scores = data->scores_temp.get();
  int* output_indices = data->indices_sorted.get();

  int loop_count = 0;
  while (current_num_partitions > 1) {
    int next_num_partitions = (current_num_partitions + 1) / 2;
    dim3 grid_reduce(next_num_partitions, batch_size);
    dim3 block_reduce(block_size);

    PairwiseReduceTopK<block_size, max_k><<<grid_reduce, block_reduce, 0, stream>>>(
        input_scores, input_indices,
        output_scores, output_indices,
        current_num_partitions);
    CUDA_CHECK(cudaGetLastError());

    // // --- DEBUG ---
    // if (enable_debug_print) {
    //   char msg_s[100];
    //   sprintf(msg_s, "After Reduce Iter %d (Scores) - %d partitions", loop_count, next_num_partitions);
    //   DebugPrintDeviceData(stream, output_scores, 64, batch_size, next_num_partitions * max_k, msg_s);
    //   char msg_i[100];
    //   sprintf(msg_i, "After Reduce Iter %d (Indices) - %d partitions", loop_count, next_num_partitions);
    //   DebugPrintDeviceData(stream, output_indices, 64, batch_size, next_num_partitions * max_k, msg_i);
    // }

    // Swap buffers for the next iteration.
    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);

    current_num_partitions = next_num_partitions;
    loop_count++;
  }

  // After the loop, the final reduced results are in the `input` buffers
  float* final_reduced_scores = input_scores;
  int* final_reduced_indices = input_indices;

  // Stage 3: Final Copy and Softmax
  // This stage now operates on the much smaller, fully reduced list of top-k candidates.
  CopyAndSoftmax<false>(stream, batch_size, indices_out, scores_out, final_reduced_indices, final_reduced_scores, k, temperature, max_k);
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  TopKConfig chosen_config;
  if (k <= 8) {
    chosen_config.algorithm = TopKAlgorithm::SELECTION_SORT;
  } else if (k <= 64) {
    // TODO: replace it by a lookup table from offline benchmark.
    chosen_config = BenchmarkAndGetBestAlgorithm(data, stream, vocab_size, batch_size, k);
  }

  switch (chosen_config.algorithm) {
    case TopKAlgorithm::SELECTION_SORT:
      RunTopKViaSelectionSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
    case TopKAlgorithm::BITONIC_SORT:
      RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, chosen_config.num_partitions, chosen_config.sort_size);
      break;
    default:
      RunTopKViaFullSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
  }
}

}  // namespace cuda
}  // namespace Generators

