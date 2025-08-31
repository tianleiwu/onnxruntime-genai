// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_helper.h"
#include <cub/cub.cuh>
#include <limits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace Generators {
namespace cuda {

// ------------------------------------------------------------------
// START of Consolidated Softmax implementation
// ------------------------------------------------------------------
/*
template <int kBlockSize, bool DoCopyIndices, bool IsLogSoftmax = false>
__global__ void ProcessTopK(
    // Outputs
    float* final_scores,
    int* final_indices, // Ignored if DoCopyIndices is false
    // Inputs
    const float* input_scores,
    const int* input_indices, // Ignored if DoCopyIndices is false
    // Parameters
    int k,
    int input_stride,
    float temperature) {
  const int batch_idx = blockIdx.x;
  const float* batch_input_scores = input_scores + batch_idx * input_stride;
  const int* batch_input_indices = DoCopyIndices ? (input_indices + batch_idx * input_stride) : nullptr;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // STEP 1: Find max of (scores / temperature) in parallel using a grid-stride loop.
  // This is safe because max(x/t) = max(x)/t for t > 0.
  float thread_max = -std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_max = max(thread_max, batch_input_scores[i]);
  }

  // Reduce to find the block-wide maximum raw score
  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced / temperature;
  }
  __syncthreads();

  // STEP 2: Find sum of exp((scores / temperature) - max_val) in parallel.
  float thread_sum_exp = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_sum_exp += expf((batch_input_scores[i] / temperature) - block_max_val);
  }

  // Reduce to find the block-wide sum of exponents.
  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_sum_exp, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads();

  // STEP 3: Write final results using a grid-stride loop.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if constexpr (DoCopyIndices) {
      // This path is only used when k is small (<= 64), so this loop will
      // typically only execute once per thread.
      final_indices[batch_idx * k + i] = batch_input_indices[i];
    }
    
    float scaled_score = batch_input_scores[i] / temperature;
    if constexpr (IsLogSoftmax) {
      // Use log() for better precision to match the original implementation.
      final_scores[batch_idx * k + i] = (scaled_score - block_max_val) - log(block_sum_exp);
    } else {
      float thread_exp = expf(scaled_score - block_max_val);
      final_scores[batch_idx * k + i] = (block_sum_exp > 0.0f) ? (thread_exp / block_sum_exp) : 0.0f;
    }
  }
}

template<bool DoCopyIndices, bool IsLogSoftmax>
void ApplySoftmaxToTopK(cudaStream_t stream,
                        float* final_scores,
                        int* final_indices,
                        const float* input_scores,
                        const int* input_indices,
                        int k,
                        int batch_size,
                        int input_stride,
                        float temperature) {
  dim3 grid(batch_size);
  // Using a block size of 256 is a good default.
  dim3 block(256);
  ProcessTopK<256, DoCopyIndices, IsLogSoftmax><<<grid, block, 0, stream>>>(
      final_scores,
      final_indices,
      input_scores,
      input_indices,
      k,
      input_stride,
      temperature
  );
  CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations to prevent linker errors
template void ApplySoftmaxToTopK<true, false>(cudaStream_t, float*, int*, const float*, const int*, int, int, int, float);
template void ApplySoftmaxToTopK<true, true>(cudaStream_t, float*, int*, const float*, const int*, int, int, int, float);
template void ApplySoftmaxToTopK<false, false>(cudaStream_t, float*, int*, const float*, const int*, int, int, int, float);
template void ApplySoftmaxToTopK<false, true>(cudaStream_t, float*, int*, const float*, const int*, int, int, int, float);
*/

// --- START of Specialized Softmax for Sorted Input and k <= 256 ---

template <int kBlockSize, bool DoCopyIndices>
__global__ void CopyAndSoftmaxKernel(int* final_indices, float* final_scores,
                                     const int* sorted_indices, const float* sorted_scores,
                                     int k, float temperature, int input_stride) {
  const int batch_idx = blockIdx.x;
  const float* batch_sorted_scores = sorted_scores + batch_idx * input_stride;

  // This implementation uses CUB for parallel reduction, which is efficient.
  // The key fix is to correctly broadcast the reduction result (which CUB places in thread 0)
  // to all other threads in the block using shared memory.
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // STEP 1: Find max_val in parallel
  // Each thread (where threadIdx.x < k) loads its score and applies temperature.
  float thread_score = (threadIdx.x < k)
                          ? (batch_sorted_scores[threadIdx.x] / temperature)
                          : -std::numeric_limits<float>::max();

  // CUB reduces the values, placing the result in thread 0.
  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_score, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced;
  }
  __syncthreads(); // Ensure block_max_val is visible to all threads.

  // STEP 2: Find sum_exp in parallel
  // Each thread calculates its contribution to the sum using the correct max value.
  float thread_exp = (threadIdx.x < k) ? expf(thread_score - block_max_val) : 0.0f;

  // CUB reduces the contributions, placing the result in thread 0.
  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_exp, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads(); // Ensure block_sum_exp is visible to all threads.

  // STEP 3: Write final results
  // Each thread (where threadIdx.x < k) writes its final index and calculated probability.
  if (threadIdx.x < k) {
    if constexpr (DoCopyIndices) {
      const int* batch_sorted_indices = sorted_indices + batch_idx * input_stride;
      final_indices[batch_idx * k + threadIdx.x] = batch_sorted_indices[threadIdx.x];
    }
    // Handle case where sum_exp is zero to prevent division by zero (NaN).
    final_scores[batch_idx * k + threadIdx.x] = (block_sum_exp > 0.0f) ? (thread_exp / block_sum_exp) : 0.0f;
  }
}


// --- START of Specialized Softmax for Sorted Input ---

template <int kBlockSize, bool DoCopyIndices>
__global__ void ProcessSortedTopK(
    float* final_scores,
    int* final_indices,
    const float* sorted_input_scores,
    const int* sorted_input_indices,
    int k,
    int input_stride,
    float temperature) {

  const int batch_idx = blockIdx.x;
  const float* batch_scores = sorted_input_scores + batch_idx * input_stride;
  [[maybe_unused]] const int* batch_indices = DoCopyIndices ? (sorted_input_indices + batch_idx * input_stride) : nullptr;

  // For sorted input, the max score is always the first element.
  __shared__ float max_val;
  if (threadIdx.x == 0) {
    max_val = batch_scores[0] / temperature;
  }
  __syncthreads();

  // Cooperatively calculate sum_exp in parallel.
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float sum_exp;

  float thread_sum_exp = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_sum_exp += expf((batch_scores[i] / temperature) - max_val);
  }

  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_sum_exp, cub::Sum());
  if (threadIdx.x == 0) {
    sum_exp = sum_exp_reduced;
  }
  __syncthreads();
  
  // All threads write final results.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if constexpr (DoCopyIndices) {
      final_indices[batch_idx * k + i] = batch_indices[i];
    }
    float scaled_score = batch_scores[i] / temperature;
    float thread_exp = expf(scaled_score - max_val);
    final_scores[batch_idx * k + i] = (sum_exp > 0.0f) ? (thread_exp / sum_exp) : 0.0f;
  }
}

template<bool DoCopyIndices>
void ApplySoftmaxToSortedTopK(cudaStream_t stream,
                              float* final_scores,
                              int* final_indices,
                              const float* sorted_input_scores,
                              const int* sorted_input_indices,
                              int k,
                              int batch_size,
                              int input_stride,
                              float temperature) {
  dim3 grid(batch_size);
  dim3 block(256);

  if (k <= 256) {
    CopyAndSoftmaxKernel<256, DoCopyIndices><<<grid, block, 0, stream>>>(final_indices, final_scores, sorted_input_indices, sorted_input_scores, k, temperature, input_stride);
  } else {
    ProcessSortedTopK<256, DoCopyIndices><<<grid, block, 0, stream>>>(
        final_scores,
        final_indices,
        sorted_input_scores,
        sorted_input_indices,
        k,
        input_stride,
        temperature);
  }
  
  CUDA_CHECK(cudaGetLastError());
}

template void ApplySoftmaxToSortedTopK<true>(cudaStream_t stream,
                              float* final_scores,
                              int* final_indices,
                              const float* sorted_input_scores,
                              const int* sorted_input_indices,
                              int k,
                              int batch_size,
                              int input_stride,
                              float temperature);

template void ApplySoftmaxToSortedTopK<false>(cudaStream_t stream,
                              float* final_scores,
                              int* final_indices,
                              const float* sorted_input_scores,
                              const int* sorted_input_indices,
                              int k,
                              int batch_size,
                              int input_stride,
                              float temperature);


}  // namespace cuda
}  // namespace Generators

