// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk.h"
#include <cub/cub.cuh>
#include <limits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Robust CUDA error checking macro
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

namespace Generators {
namespace cuda {

inline int CeilDiv(int a, int b){
  return (a + (b - 1)) / b;
}

// ------------------------------------------------------------------
// START of Map-Reduce implementations
// ------------------------------------------------------------------

template <int kBlockSize, bool use_cub = false>
__global__ void CopyAndSoftmaxKernel(int* final_indices, float* final_scores,
                                     const int* sorted_indices, const float* sorted_scores,
                                     int k, float temperature, int input_stride) {
  const int batch_idx = blockIdx.x;
  const int* batch_sorted_indices = sorted_indices + batch_idx * input_stride;
  const float* batch_sorted_scores = sorted_scores + batch_idx * input_stride;

  if constexpr (use_cub) {
    // STEP 1: All threads cooperatively copy the final indices.
    for (int i = threadIdx.x; i < k; i += kBlockSize) {
      final_indices[batch_idx * k + i] = batch_sorted_indices[i];
    }

    // STEP 2: Perform softmax using a parallel reduction to match the reference implementation's math.
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce_max;
    typename BlockReduce::TempStorage reduce_sum;
  } temp_storage;

    // Each thread loads its score and applies temperature. Pad with a large negative for non-participating threads.
  float thread_score = -std::numeric_limits<float>::max();
  if (threadIdx.x < k) {
    thread_score = batch_sorted_scores[threadIdx.x] / temperature;
  }

    // Parallel reduction to find the maximum score.
  float max_val = BlockReduce(temp_storage.reduce_max).Reduce(thread_score, cub::Max());
  __syncthreads();

    // Calculate `exp(score - max)` for each thread's score.
  float thread_exp = 0.0f;
  if (threadIdx.x < k) {
    thread_exp = expf(thread_score - max_val);
  }

    // Parallel reduction to find the sum of the exponentials.
  float sum_exp = BlockReduce(temp_storage.reduce_sum).Reduce(thread_exp, cub::Sum());
  __syncthreads();

    // STEP 3: All threads write the final softmax probability.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    final_indices[batch_idx * k + i] = batch_sorted_indices[i];
    float scaled_score = batch_sorted_scores[i] / temperature;
    final_scores[batch_idx * k + i] = expf(scaled_score - max_val) / sum_exp;
    }
  } else {
    __shared__ float top_k_scores_smem[64];  // max_k

    // Cooperatively load the top k scores into shared memory
    if (threadIdx.x < k) {
      top_k_scores_smem[threadIdx.x] = batch_sorted_scores[threadIdx.x] / temperature;
    }
    __syncthreads();

    // Thread 0 computes max and sum_exp for softmax
    __shared__ float max_val;
    __shared__ float sum_exp;
    if (threadIdx.x == 0) {
      max_val = -std::numeric_limits<float>::max();
      for (int i = 0; i < k; i++) {
        if (top_k_scores_smem[i] > max_val) {
          max_val = top_k_scores_smem[i];
        }
      }
      sum_exp = 0.0f;
      for (int i = 0; i < k; i++) {
        sum_exp += expf(top_k_scores_smem[i] - max_val);
      }
    }
    __syncthreads();

    // All threads write final results
    for (int i = threadIdx.x; i < k; i += kBlockSize) {
      final_indices[batch_idx * k + i] = batch_sorted_indices[i];
      final_scores[batch_idx * k + i] = expf(top_k_scores_smem[i] - max_val) / sum_exp;
    }
  }
}

template<bool use_cub>
void CopyAndSoftmax(cudaStream_t stream, int batch_size, 
                    int* final_indices, float* final_scores,
                    const int* sorted_indices, const float* sorted_scores,
                    int k, float temperature, int input_stride) {
  dim3 grid(batch_size);
  dim3 block(256);
  CopyAndSoftmaxKernel<256, use_cub><<<grid, block, 0, stream>>>(final_indices, final_scores, sorted_indices, sorted_scores, k, temperature, input_stride);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace Generators
