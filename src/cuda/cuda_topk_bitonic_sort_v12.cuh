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
namespace bitonic_v12 {

static const char* kAlgoDescription = "Bitonic v12 (Hybrid: v11 map + v0 CUB reduce)";

// --- START: Device Helper Functions (from v11) ---

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
            float temp_s = smem_scores[i];
            smem_scores[i] = smem_scores[ixj];
            smem_scores[ixj] = temp_s;
            int temp_i = smem_indices[i];
            smem_indices[i] = smem_indices[ixj];
            smem_indices[ixj] = temp_i;
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
          float temp_s = smem_scores[i];
          smem_scores[i] = smem_scores[ixj];
          smem_scores[ixj] = temp_s;
          int temp_i = smem_indices[i];
          smem_indices[i] = smem_indices[ixj];
          smem_indices[ixj] = temp_i;
        }
      }
    }
    __syncthreads();
  }
}

// Sorts an array of `N` elements held in registers.
template <int N>
__device__ void RegisterBitonicSort(float scores[N], int indices[N]);

// Specialized version for N=2, unrolled for max performance.
template <>
__device__ void RegisterBitonicSort<2>(float scores[2], int indices[2]) {
  bool is_greater = (scores[0] > scores[1]) || (scores[0] == scores[1] && indices[0] < indices[1]);
  if (!is_greater) {
    float temp_s = scores[0];
    scores[0] = scores[1];
    scores[1] = temp_s;
    int temp_i = indices[0];
    indices[0] = indices[1];
    indices[1] = temp_i;
  }
}

// Generic version for other N.
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
            float temp_s = scores[i];
            scores[i] = scores[ixj];
            scores[ixj] = temp_s;
            int temp_i = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = temp_i;
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
          float temp_s = scores[i];
          scores[i] = scores[ixj];
          scores[ixj] = temp_s;
          int temp_i = indices[i];
          indices[i] = indices[ixj];
          indices[ixj] = temp_i;
        }
      }
    }
  }
}
// --- END: Device Helper Functions ---

// This kernel is from v11, replacing the original v0 map-stage kernel.
// It finds the top-k elements within a single data partition.
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

  // Load a contiguous chunk of data from global memory into registers (coalesced access)
  for (int i = 0; i < ElementsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x * ElementsPerThread + i;
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
    smem_scores[threadIdx.x * ElementsPerThread + i] = reg_scores[i];
    smem_indices[threadIdx.x * ElementsPerThread + i] = reg_indices[i];
  }
  __syncthreads();

  // Merge the pre-sorted chunks in shared memory
  SharedMemBitonicSort<kBlockSize, kSortSize>(smem_scores, smem_indices);

  // Have the first `max_k` threads write out the top results
  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  // Stage 1: Map Phase - Find top-k within each partition using the optimized kernel.
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

  // Stage 2: Reduce Phase - Sort the intermediate results using CUB Segmented Sort (unchanged from original v0).
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
