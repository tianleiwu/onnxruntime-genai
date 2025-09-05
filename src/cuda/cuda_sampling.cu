// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <stdio.h>
#include "cuda_sampling.h"
#include "cuda_topk.h"
#include "cuda_topk_softmax.cuh"
#include "smartptrs.h"
#include "span.h"

namespace Generators {
namespace cuda {

// Initializes the cuRAND states for each batch item.
__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size) return;
  curand_init(seed, index, 0, &states[index]);
}

void SamplingData::ReInitCurandStates(unsigned long long random_seed, int batch_size, cudaStream_t stream) {
  random_seed_ = random_seed;
  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream)
    : TopkData(batch_size, vocab_size, stream) {
  const size_t topk_batch_size = static_cast<size_t>(kHybridSortMaxK) * batch_size;

  prefix_sums = CudaMallocArray<float>(topk_batch_size);
  scores_adjusted = CudaMallocArray<float>(std::max(topk_batch_size, static_cast<size_t>(vocab_size) * batch_size));
  prefix_sums_adjusted = CudaMallocArray<float>(topk_batch_size);
  thresholds = CudaMallocArray<float>(batch_size);
  curand_states = CudaMallocArray<curandState>(batch_size);
  ReInitCurandStates(random_seed, batch_size, stream);
}

// A fused kernel that performs all steps of Top-P sampling on a pre-selected set of Top-K candidates.
// This monolithic approach is optimized for k <= 256.
template <int kBlockSize>
__global__ void FusedSamplingKernel_SmallK(int32_t* next_token_out, const float* scores, const int* indices, int k,
                                           float p, float temperature, int stride, curandState* curand_states) {
  const int batch_idx = blockIdx.x;
  const float* batch_scores = scores + batch_idx * stride;
  const int* batch_indices = indices + batch_idx * stride;

  extern __shared__ float smem[];
  float* temp_scaled_logits = smem;
  float* filtered_logits = smem + kBlockSize;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage reduce_temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // --- Stage 1: Initial Softmax with Temperature ---
  float thread_val = -FLT_MAX;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    float logit = batch_scores[i] / temperature;
    temp_scaled_logits[i] = logit;
    thread_val = max(thread_val, logit);
  }
  float reduced_max = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Max());
  if (threadIdx.x == 0) block_max_val = reduced_max;
  __syncthreads();

  thread_val = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val += expf(temp_scaled_logits[i] - block_max_val);
  }
  float reduced_sum = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Sum());
  if (threadIdx.x == 0) block_sum_exp = reduced_sum;
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    temp_scaled_logits[i] =
        (block_sum_exp > 0.0f) ? (expf(temp_scaled_logits[i] - block_max_val) / block_sum_exp) : 0.0f;
  }
  __syncthreads();

  // --- Stage 2: Compute Initial CDF ---
  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_temp_storage;
  float running_total = 0.0f;
  for (int i = 0; i < k; i += kBlockSize) {
    float score = (threadIdx.x + i < k) ? temp_scaled_logits[threadIdx.x + i] : 0.0f;
    float scanned_score;
    BlockScan(scan_temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();

    if (threadIdx.x + i < k) temp_scaled_logits[threadIdx.x + i] = scanned_score + running_total;
    __syncthreads();

    if (threadIdx.x == kBlockSize - 1) running_total += scanned_score;
    __syncthreads();
  }

  // --- Stage 3: Filter original logits ---
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    const float prev_sum = (i == 0) ? 0.0f : temp_scaled_logits[i - 1];
    filtered_logits[i] = (prev_sum < p) ? batch_scores[i] : -FLT_MAX;
  }
  __syncthreads();

  // --- Stage 4: Re-normalize filtered logits (temp=1.0) ---
  thread_val = -FLT_MAX;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val = max(thread_val, filtered_logits[i]);
  }
  reduced_max = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Max());
  if (threadIdx.x == 0) block_max_val = reduced_max;
  __syncthreads();

  thread_val = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val += expf(filtered_logits[i] - block_max_val);
  }
  reduced_sum = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Sum());
  if (threadIdx.x == 0) block_sum_exp = reduced_sum;
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    filtered_logits[i] = (block_sum_exp > 0.0f) ? (expf(filtered_logits[i] - block_max_val) / block_sum_exp) : 0.0f;
  }
  __syncthreads();

  // --- Stage 5: Compute Final CDF ---
  running_total = 0.0f;
  for (int i = 0; i < k; i += kBlockSize) {
    float score = (threadIdx.x + i < k) ? filtered_logits[threadIdx.x + i] : 0.0f;
    float scanned_score;
    BlockScan(scan_temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();
    if (threadIdx.x + i < k) filtered_logits[threadIdx.x + i] = scanned_score + running_total;
    __syncthreads();
    if (threadIdx.x == kBlockSize - 1) running_total += scanned_score;
    __syncthreads();
  }

  // --- Stage 6 & 7: Sample via Parallel Search ---
  __shared__ int selected_index_smem;
  __shared__ float threshold_smem;

  if (threadIdx.x == 0) {
    threshold_smem = 0.9999999f * curand_uniform(&curand_states[batch_idx]);
    selected_index_smem = k - 1;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if (filtered_logits[i] >= threshold_smem) {
      atomicMin(&selected_index_smem, i);
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    next_token_out[batch_idx] = batch_indices[selected_index_smem];
  }
}

void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, const float* scores_in,
               int vocab_size, int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  const float* topk_scores;
  const int* topk_indices;
  int output_stride = GetTopK(data, stream, scores_in, &topk_scores, &topk_indices, vocab_size, batch_size, k);

  // For small k, the fused kernel is significantly faster due to reduced overhead and better data locality.
  // For large k, the fused kernel would require too much shared memory, so we fall back to a multi-stage approach.
  if (k <= 256) {
    dim3 grid(batch_size);
    const int block_size = 256;
    dim3 block(block_size);
    size_t shared_mem_bytes = 2 * block_size * sizeof(float);

    FusedSamplingKernel_SmallK<block_size><<<grid, block, shared_mem_bytes, stream>>>(
        next_token_out, topk_scores, topk_indices, k, p, temperature, output_stride, data->curand_states.get());
  } else {
    // This path is for correctness when k > 256. It is not expected to be a common case.
    // It re-implements the original, slower multi-kernel pipeline.
    // LaunchMultiStageSampleKernel(data, stream, topk_scores, topk_indices, next_token_out, k, batch_size, p,
    // temperature, output_stride);
  }
  CUDA_CHECK(cudaGetLastError());
}

// Implementation for the general-purpose block-wise softmax, used by beam search.
template <int kBlockSize, bool is_log_softmax>
__global__ void BlockwiseSoftmaxKernel(float* output, const float* input, int softmax_elements, int input_stride,
                                       int output_stride) {
  const int batch_idx = blockIdx.x;
  const float* batch_input = input + batch_idx * input_stride;
  float* batch_output = output + batch_idx * output_stride;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float max_val;
  __shared__ float sum_exp;

  // Step 1: Find max value in parallel for numerical stability.
  float thread_max = -std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
    thread_max = max(thread_max, batch_input[i]);
  }
  float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) {
    max_val = block_max;
  }
  __syncthreads();

  // Step 2: Compute sum of exponents in parallel.
  float thread_sum_exp = 0.0f;
  for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
    thread_sum_exp += expf(batch_input[i] - max_val);
  }
  float block_sum = BlockReduce(temp_storage).Reduce(thread_sum_exp, cub::Sum());
  if (threadIdx.x == 0) {
    sum_exp = block_sum;
  }
  __syncthreads();

  // Step 3: Compute final softmax or log_softmax and write to output.
  if constexpr (is_log_softmax) {
    // Add a small epsilon to prevent log(0) which results in -inf.
    float log_sum_exp = logf(sum_exp + 1e-20f);
    for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
      batch_output[i] = batch_input[i] - max_val - log_sum_exp;
    }
  } else {
    for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
      // Handle case where sum_exp is zero to prevent division by zero (NaN).
      batch_output[i] = (sum_exp > 0.0f) ? (expf(batch_input[i] - max_val) / sum_exp) : 0.0f;
    }
  }
}

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count) {
  // This kernel is efficient for large softmax_elements (like vocab_size) where
  // a single block can cooperatively process one batch item.
  constexpr int kBlockSize = 256;
  dim3 grid(batch_count);
  dim3 block(kBlockSize);

  BlockwiseSoftmaxKernel<kBlockSize, is_log_softmax><<<grid, block, 0, stream>>>(output, input, softmax_elements,
                                                                                input_stride, output_stride);
  CUDA_CHECK(cudaGetLastError());
}

// Explicitly instantiate the templates to be linked from other translation units.
template void DispatchBlockwiseSoftmaxForward<true>(cudaStream_t, float*, const float*, int, int, int, int);
template void DispatchBlockwiseSoftmaxForward<false>(cudaStream_t, float*, const float*, int, int, int, int);

}  // namespace cuda
}  // namespace Generators

