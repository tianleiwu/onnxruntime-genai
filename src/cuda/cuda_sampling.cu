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
#include "smartptrs.h"
#include "span.h"

// Add this to cuda_sampling.cu after the includes
#define DEBUG_SAMPLING 1

#if DEBUG_SAMPLING
#include <vector>
#include <iomanip>
// Helper to print `n` elements of a device vector starting at `offset_elements`
void PrintDeviceVector(const float* d_ptr, size_t n, size_t offset_elements, const char* name, cudaStream_t stream) {
  std::vector<float> h_vec(n);
  cudaMemcpyAsync(h_vec.data(), d_ptr + offset_elements, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);  // Sync to ensure data is ready on host

  std::cout << "\n--- DEBUG: " << name << " ---" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < n; ++i) {
    std::cout << h_vec[i] << " ";
    if ((i > 0) && (i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << "\n------------------------------------------------" << std::endl;
}
bool EnableDebug(int batch_size, int k) { return (k == 10 && batch_size == 10); }
#endif

namespace Generators {
namespace cuda {

__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size) return;

  curand_init(seed, index, 0, &states[index]);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream)
    : TopkData(batch_size, vocab_size, stream) {
  const size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  prefix_sums = CudaMallocArray<float>(vocab_batch_size);
  scores_adjusted = CudaMallocArray<float>(vocab_batch_size);
  prefix_sums_adjusted = CudaMallocArray<float>(vocab_batch_size);

  thresholds = CudaMallocArray<float>(batch_size);
  curand_states = CudaMallocArray<curandState>(batch_size);

  if (batch_size == 10 && vocab_size == 32000)
    printf("random seed: %llu, batch_size: %d, vocab_size: %d\n", random_seed, batch_size, vocab_size);
  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// --- Sampling Kernels ---

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
__global__ void FilterOnTopPKernel(float* filtered_logits_out, const float* prefix_sums_in, const float* raw_logits_in,
                                   int sample_range, float p) {
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

// A CUB-based softmax that correctly handles temperature for re-normalization.
template <int kBlockSize>
__global__ void RenormalizeSoftmaxKernel(float* final_scores, const float* input_scores, int k, float temperature) {
  const int batch_idx = blockIdx.x;
  const float* batch_input_scores = input_scores + batch_idx * k;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  float thread_max = -std::numeric_limits<float>::max();
  if (threadIdx.x < k) {
    thread_max = batch_input_scores[threadIdx.x] / temperature;
  }

  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced;
  }
  __syncthreads();

  float thread_exp_sum = 0.0f;
  if (threadIdx.x < k) {
    thread_exp_sum = expf(batch_input_scores[threadIdx.x] / temperature - block_max_val);
  }

  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_exp_sum, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads();

  if (threadIdx.x < k) {
    final_scores[batch_idx * k + threadIdx.x] =
        (block_sum_exp > 0.0f) ? (expf(batch_input_scores[threadIdx.x] / temperature - block_max_val) / block_sum_exp)
                               : 0.0f;
  }
}

// __global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;
//   if (index < batch_size) {
//     thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
//   }
// }

// REPLACE the existing RandomThresholdKernel with this debug version
__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size) {
    thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
  }
}

template <int kBlockSize>
__global__ void SampleKernel(float* prefix_sums, int* indices, int* index_out, int sample_range, int indices_stride,
                             float* thresholds, bool enable_debug) {
  int batch = blockIdx.x;

  // Only one thread per block will perform the search to ensure no race conditions.
  if (threadIdx.x == 0) {
    float threshold = thresholds[batch];
    int selected_index = sample_range - 1;  // Default to the last valid token

    // Simple linear search to find the first token whose CDF is >= the threshold
    for (int i = 0; i < sample_range; i++) {
      if (prefix_sums[batch * sample_range + i] >= threshold) {
        selected_index = i;
        break;
      }
    }

    // DEBUG: Print the inputs and output of the sampling decision
    if (enable_debug) {
      printf("Batch %d --- Threshold: %f, Selected Index: %d, CDF at Index: %f\n", batch, threshold, selected_index,
             prefix_sums[batch * sample_range + selected_index]);
    }

    index_out[batch] = indices[batch * indices_stride + selected_index];
  }
}

void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, float* scores, int* indices, int* index_out,
                        int sample_range, int batch_size, int indices_stride, float p, int k, float temperature) {
  dim3 grid(batch_size);
  dim3 block(256);

  // Stage 1: Compute a correct prefix sum of the initial Top-K probabilities.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(scores, data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  bool enable_debug = EnableDebug(batch_size, k);
  if (enable_debug) {
    cudaStreamSynchronize(stream);  // Sync before printing
    PrintDeviceVector(data->prefix_sums.get(), k, 0 * k, "CDF of Initial Probs [Batch 0]", stream);
    PrintDeviceVector(data->prefix_sums.get(), k, 1 * k, "CDF of Initial Probs [Batch 1]", stream);
  }
#endif

  // Stage 2: Filter the raw logits based on the computed cumulative probability.
  const float* raw_topk_logits = data->intermediate_scores_1.get();
  FilterOnTopPKernel<<<grid, block, 0, stream>>>(data->scores_adjusted.get(), data->prefix_sums.get(), raw_topk_logits,
                                                 k, p);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  if (enable_debug) {
    cudaStreamSynchronize(stream);
    PrintDeviceVector(data->scores_adjusted.get(), k, 0 * k, "Filtered Raw Logits [Batch 0]", stream);
    PrintDeviceVector(data->scores_adjusted.get(), k, 1 * k, "Filtered Raw Logits [Batch 1]", stream);
  }
#endif

  // Stage 3: Re-normalize the filtered logits via softmax, now with temperature.
  RenormalizeSoftmaxKernel<256>
      <<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), data->scores_adjusted.get(), k, temperature);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  if (enable_debug) {
    cudaStreamSynchronize(stream);
    PrintDeviceVector(data->prefix_sums_adjusted.get(), k, 0 * k, "Re-Normalized Probs [Batch 0]", stream);
    PrintDeviceVector(data->prefix_sums_adjusted.get(), k, 1 * k, "Re-Normalized Probs [Batch 1]", stream);
  }
#endif

  // Stage 4: Compute a prefix sum of the new, re-normalized probabilities for sampling.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());
#if DEBUG_SAMPLING
  if (enable_debug) {
    cudaStreamSynchronize(stream);
    PrintDeviceVector(data->prefix_sums.get(), k, 0 * k, "Re-Normalized Probs [Batch 0]", stream);
    PrintDeviceVector(data->prefix_sums.get(), k, 1 * k, "Re-Normalized Probs [Batch 1]", stream);
  }
#endif

  // Stage 5: Generate random thresholds and sample one token per batch item.
  RandomThresholdKernel<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(data->curand_states.get(), data->thresholds.get(),
                                                                      batch_size);
  CUDA_CHECK(cudaGetLastError());

  SampleKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums.get(), indices, index_out, sample_range,
                                                indices_stride, data->thresholds.get(), enable_debug);
  CUDA_CHECK(cudaGetLastError());
}

void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size,
               int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  TopkData* topk_data = data;
  GetTopKSubset(topk_data, stream, scores_in, data->topk_probs.get(), data->topk_indices.get(), vocab_size, batch_size,
                k, temperature);

#if DEBUG_SAMPLING
  if (EnableDebug(batch_size, k)) {
    std::cout << "\n\n========= NEW SAMPLING CALL (k=" << k << ", p=" << p << ") =========" << std::endl;
    // Print raw logits assuming a STRIDE of 64 (for hybrid sort debugging)
    PrintDeviceVector(data->intermediate_scores_1.get(), k, 0 * 64, "Initial Raw Top-K Logits [Batch 0]", stream);
    PrintDeviceVector(data->intermediate_scores_1.get(), k, 1 * 64, "Initial Raw Top-K Logits [Batch 1]", stream);
    // Print probs/CDF assuming a COMPACT stride of k
    PrintDeviceVector(data->topk_probs.get(), k, 0 * k, "Initial Top-K Probs [Batch 0]", stream);
    PrintDeviceVector(data->topk_probs.get(), k, 1 * k, "Initial Top-K Probs [Batch 1]", stream);
  }
#endif

  int sample_range = k;
  int indices_stride = k;
  LaunchSampleKernel(data, stream, data->topk_probs.get(), data->topk_indices.get(), next_token_out, sample_range,
                     batch_size, indices_stride, p, k, temperature);
}
}  // namespace cuda
}  // namespace Generators