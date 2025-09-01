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
namespace bitonic_v0 {
static const char* kAlgoDescription = "Bitonic v0 (before optimizations)";

// In file: cuda_topk_bitonic_sort_v0.cuh

// Restore this kernel to its original state. The logic here is correct.
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

  // --- In-place Bitonic Sort (ascending) ---
  for (int k = 2; k <= kSortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0) {  // Sort ascending
            if (smem_scores[i] > smem_scores[ixj]) {
              float temp_s = smem_scores[i];
              smem_scores[i] = smem_scores[ixj];
              smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i];
              smem_indices[i] = smem_indices[ixj];
              smem_indices[ixj] = temp_i;
            }
          } else {  // Sort descending
            if (smem_scores[i] < smem_scores[ixj]) {
              float temp_s = smem_scores[i];
              smem_scores[i] = smem_scores[ixj];
              smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i];
              smem_indices[i] = smem_indices[ixj];
              smem_indices[ixj] = temp_i;
            }
          }
        }
      }
      __syncthreads();
    }
  }
  // Final pass to make the whole array descending by reversing the ascending array
  for (int i = threadIdx.x; i < kSortSize / 2; i += kBlockSize) {
    if (smem_scores[i] < smem_scores[kSortSize - 1 - i]) {
      float temp_s = smem_scores[i];
      smem_scores[i] = smem_scores[kSortSize - 1 - i];
      smem_scores[kSortSize - 1 - i] = temp_s;
      int temp_i = smem_indices[i];
      smem_indices[i] = smem_indices[kSortSize - 1 - i];
      smem_indices[kSortSize - 1 - i] = temp_i;
    }
  }
  __syncthreads();

  // Have the first `max_k` threads write out the top results
  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  constexpr int block_size = 256;
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  switch (sort_size) {
    case 256:
      FindBlockTopK_BitonicSort<block_size, 256><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->indices_in.get(), data->scores_buffer.get(), vocab_size, num_partitions);
      break;
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

  // Stage 2: Sort the small intermediate buffer using CUB Segmented Sort
  const int num_intermediate_results_per_batch = num_partitions * kBitonicSortMaxK;
  const int total_intermediate_results = batch_size * num_intermediate_results_per_batch;

  float* sorted_scores = data->scores_temp.get();
  int* sorted_indices = data->indices_sorted.get();

  LaunchPopulateOffsets(data->offsets.get(), num_intermediate_results_per_batch, batch_size, stream);

  size_t temp_storage_bytes_needed = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes_needed, data->scores_buffer.get(), sorted_scores, data->indices_in.get(), sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));

  if (data->temp_storage_bytes < temp_storage_bytes_needed) {
    std::cerr << "FATAL ERROR in RunTopKViaMapReduceBitonicSort_v0: Pre-allocated temp_buffer is too small." << std::endl;
    return;
  }

  // Call CUB with the correct total size and number of segments (batch_size).
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(data->temp_buffer.get(), temp_storage_bytes_needed, data->scores_buffer.get(), sorted_scores, data->indices_in.get(), sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));

  // Stage 3: Final Copy and Softmax
  // Use the per-batch stride to correctly extract the top k.
  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out,
                                 sorted_scores, sorted_indices,
                                 k, batch_size, num_intermediate_results_per_batch, temperature);
}
} // bitonic_v0
}  // namespace cuda
}  // namespace Generators
