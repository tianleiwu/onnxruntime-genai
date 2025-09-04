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

#include "cuda_sampling.h"
#include "cuda_topk.h"
#include "smartptrs.h"
#include "span.h"

// Add this to cuda_sampling.cu after the includes
#define DEBUG_SAMPLING 1

#if DEBUG_SAMPLING
#include <vector>
#include <iomanip>
// Helper to print the first `n` elements of a device vector
void PrintDeviceVector(const float* d_ptr, size_t n, const char* name, cudaStream_t stream) {
  // We only print the first batch item for clarity.
  std::vector<float> h_vec(n);
  cudaMemcpyAsync(h_vec.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);  // Sync to ensure data is ready on host

  std::cout << "\n--- DEBUG: " << name << " (batch 0) ---" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < n; ++i) {
    std::cout << h_vec[i] << " ";
    if ((i > 0) && (i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << "\n------------------------------------------------" << std::endl;
}
#endif

namespace Generators {
namespace cuda {

constexpr int kMaxThreads = 1024;
constexpr int kGPUWarpSize = 32;

__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size)
    return;

  curand_init(seed, index, 0, &states[index]);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream) : TopkData(batch_size, vocab_size, stream) {
  const size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  prefix_sums = CudaMallocArray<float>(vocab_batch_size);
  scores_adjusted = CudaMallocArray<float>(vocab_batch_size);
  prefix_sums_adjusted = CudaMallocArray<float>(vocab_batch_size);

  thresholds = CudaMallocArray<float>(batch_size);
  curand_states = CudaMallocArray<curandState>(batch_size);

  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// --- REMOVED incorrect softmax and other helpers, replacing with a single new kernel below ---

// --- Sampling Kernels ---

// Performs a correct segmented prefix sum. Each block handles one segment (one batch item).
template <int kBlockSize>
__global__ void CorrectPrefixSumKernel(const float* scores, float* prefix_sums, int sample_range) {
  int batch = blockIdx.x;
  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ float chunk_total;

  float running_total = 0.0f;

  for (int i = 0; i < sample_range; i += kBlockSize) {
    int global_index = threadIdx.x + i + batch * sample_range;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;

    float scanned_score;
    BlockScan(temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();

    if (local_index < sample_range) {
      prefix_sums[global_index] = scanned_score + running_total;
    }
    __syncthreads();

    if (threadIdx.x == kBlockSize - 1) {
      chunk_total = scanned_score;
    }
    __syncthreads();
    running_total += chunk_total;
  }
}

// Filters raw logits based on a pre-computed prefix scan of probabilities.
__global__ void FilterOnTopPKernel(
    float* filtered_logits_out,
    const float* prefix_sums_in,
    const float* raw_logits_in,
    int sample_range,
    float p) {
  const int batch_idx = blockIdx.x;

  for (int i = threadIdx.x; i < sample_range; i += blockDim.x) {
    const int global_idx = batch_idx * sample_range + i;
    const float prev_sum = (i == 0) ? 0.0f : prefix_sums_in[global_idx - 1];

    if (prev_sum < p) {
      filtered_logits_out[global_idx] = raw_logits_in[global_idx];
    } else {
      filtered_logits_out[global_idx] = -std::numeric_limits<float>::max();
    }
  }
}

// NEW KERNEL: A CUB-based softmax that correctly handles temperature for re-normalization.
template <int kBlockSize>
__global__ void RenormalizeSoftmaxKernel(float* final_scores,
                                         const float* input_scores,
                                         int k, float temperature) {
  const int batch_idx = blockIdx.x;
  const float* batch_input_scores = input_scores + batch_idx * k;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // Step 1: Find max_val in parallel on temperature-scaled scores
  float thread_score = (threadIdx.x < k)
                           ? (batch_input_scores[threadIdx.x] / temperature)
                           : -std::numeric_limits<float>::max();

  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_score, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced;
  }
  __syncthreads();

  // Step 2: Find sum_exp in parallel
  float thread_exp = (threadIdx.x < k) ? expf(thread_score - block_max_val) : 0.0f;

  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_exp, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads();

  // Step 3: Write final probabilities
  if (threadIdx.x < k) {
    final_scores[batch_idx * k + threadIdx.x] = (block_sum_exp > 0.0f) ? (thread_exp / block_sum_exp) : 0.0f;
  }
}

__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size) {
    thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
  }
}

// Corrected sampling kernel with proper loop bounds.
template <int kBlockSize>
__global__ void SampleKernel(float* prefix_sums, int* indices, int* index_out, int sample_range, int indices_stride, float* thresholds) {
  int batch = blockIdx.x;

  __shared__ int first_index;
  if (threadIdx.x == 0) {
    first_index = sample_range - 1;
  }
  __syncthreads();

  bool found_candidate = false;
  for (int index = threadIdx.x; index < sample_range; index += blockDim.x) {
    if (!found_candidate) {
      float sum = prefix_sums[batch * sample_range + index];
      if (sum >= thresholds[batch]) {
        atomicMin(&first_index, index);
        found_candidate = true;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_out[batch] = indices[batch * indices_stride + first_index];
  }
}

void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, float* scores, int* indices, int* index_out, int sample_range, int batch_size, int indices_stride, float p, int k, float temperature) {
  dim3 grid(batch_size);
  dim3 block(256);

  // Stage 1: Compute a correct prefix sum of the initial Top-K probabilities.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(scores, data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  PrintDeviceVector(data->prefix_sums.get(), k, "CDF of Initial Probs", stream);
#endif

  // Stage 2: Filter the raw logits based on the computed cumulative probability.
  const float* raw_topk_logits = data->intermediate_scores_1.get();
  FilterOnTopPKernel<<<grid, block, 0, stream>>>(
      data->scores_adjusted.get(),
      data->prefix_sums.get(),
      raw_topk_logits,
      k, p);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  PrintDeviceVector(data->scores_adjusted.get(), k, "Filtered Raw Logits", stream);
#endif

  // Stage 3: Re-normalize the filtered logits via softmax, now with temperature.
  RenormalizeSoftmaxKernel<256><<<grid, block, 0, stream>>>(
      data->prefix_sums_adjusted.get(),
      data->scores_adjusted.get(),
      k, temperature);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  PrintDeviceVector(data->prefix_sums_adjusted.get(), k, "Re-Normalized Probs", stream);
#endif

  // Stage 4: Compute a prefix sum of the new, re-normalized probabilities for sampling.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  PrintDeviceVector(data->prefix_sums.get(), k, "Final CDF for Sampling", stream);
#endif

  // Stage 5: Generate random thresholds and sample one token per batch item.
  RandomThresholdKernel<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(data->curand_states.get(), data->thresholds.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());

  SampleKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums.get(), indices, index_out, sample_range, indices_stride, data->thresholds.get());
  CUDA_CHECK(cudaGetLastError());
}

// Main sampling entry point
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size, int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  TopkData* topk_data = data;
  GetTopKSubset(topk_data, stream, scores_in, data->topk_probs.get(), data->topk_indices.get(), vocab_size, batch_size, k, temperature);

#if DEBUG_SAMPLING
  std::cout << "\n\n========= NEW SAMPLING CALL (k=" << k << ", p=" << p << ") =========" << std::endl;
  PrintDeviceVector(data->intermediate_scores_1.get(), k, "Initial Raw Top-K Logits", stream);
  PrintDeviceVector(data->topk_probs.get(), k, "Initial Top-K Probs (after 1st softmax)", stream);
#endif

  int sample_range = k;
  int indices_stride = k;
  LaunchSampleKernel(data, stream, data->topk_probs.get(), data->topk_indices.get(), next_token_out, sample_range, batch_size, indices_stride, p, k, temperature);
}
}  // namespace cuda
}  // namespace Generators