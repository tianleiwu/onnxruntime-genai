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

namespace Generators {
namespace cuda {

// Initializes the cuRAND states for each batch item.
__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size) return;
  curand_init(seed, index, 0, &states[index]);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream)
    : TopkData(batch_size, vocab_size, stream) {
  const size_t topk_batch_size = static_cast<size_t>(kHybridSortMaxK) * batch_size;

  prefix_sums = CudaMallocArray<float>(topk_batch_size);
  scores_adjusted = CudaMallocArray<float>(topk_batch_size);
  prefix_sums_adjusted = CudaMallocArray<float>(topk_batch_size);
  thresholds = CudaMallocArray<float>(batch_size);
  curand_states = CudaMallocArray<curandState>(batch_size);

  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Computes an inclusive prefix sum (scan) on the input scores.
// This is used to generate the Cumulative Distribution Function (CDF).
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

// Filters logits based on the Top-P cumulative probability.
// Logits that fall outside the probability mass `p` are set to -infinity.
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

// Applies temperature to raw logits and computes their softmax probability.
template <int kBlockSize>
__global__ void ApplyTemperatureAndSoftmax(float* final_scores, const float* input_scores, int k, float temperature) {
  const int batch_idx = blockIdx.x;
  const float* batch_input_scores = input_scores + batch_idx * k;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // Find max value in the block for numerical stability
  float thread_score =
      (threadIdx.x < k) ? (batch_input_scores[threadIdx.x] / temperature) : -std::numeric_limits<float>::max();
  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_score, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced;
  }
  __syncthreads();

  // Compute sum of exponents
  float thread_exp = (threadIdx.x < k) ? expf(thread_score - block_max_val) : 0.0f;
  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_exp, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads();

  // Compute final softmax probability
  if (threadIdx.x < k) {
    final_scores[batch_idx * k + threadIdx.x] = (block_sum_exp > 0.0f) ? (thread_exp / block_sum_exp) : 0.0f;
  }
}

// Generates a random float threshold [0.0, 1.0) for each batch item.
__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size) {
    thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
  }
}

// Samples a single token per batch item based on the CDF and a random threshold.
template <int kBlockSize>
__global__ void SampleKernel(const float* prefix_sums, const int* indices, int* index_out, int sample_range,
                             float* thresholds) {
  int batch = blockIdx.x;

  if (threadIdx.x == 0) {
    float threshold = thresholds[batch];
    int selected_index = sample_range - 1;  // Default to last element

    // Find the first element in the CDF that is >= the threshold.
    for (int i = 0; i < sample_range; i++) {
      if (prefix_sums[batch * sample_range + i] >= threshold) {
        selected_index = i;
        break;
      }
    }
    // Convert from the local index within the top-k set to the original vocabulary index.
    index_out[batch] = indices[batch * sample_range + selected_index];
  }
}

// Orchestrates the multi-stage Top-P sampling process.
void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, const float* scores, const int* indices,
                        int* index_out, int sample_range, int batch_size, float p, int k, float temperature) {
  dim3 grid(batch_size);
  dim3 block(256);

  // Stage 1: Apply temperature and softmax to the raw top-k logits to get initial probabilities.
  ApplyTemperatureAndSoftmax<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), scores, k, temperature);
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Compute prefix sum (CDF) of initial probabilities for Top-P filtering.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());

  // Stage 3: Filter the raw logits based on the computed cumulative probability `p`.
  FilterOnTopPKernel<<<grid, block, 0, stream>>>(data->scores_adjusted.get(), data->prefix_sums.get(), scores, k, p);
  CUDA_CHECK(cudaGetLastError());

  // Stage 4: Re-normalize the filtered logits via softmax.
  ApplyTemperatureAndSoftmax<256><<<grid, block, 0, stream>>>(
      data->prefix_sums_adjusted.get(), data->scores_adjusted.get(), k, 1.0f);  // Temp is 1.0f for re-normalization
  CUDA_CHECK(cudaGetLastError());

  // Stage 5: Compute a prefix sum of the re-normalized probabilities to create the final CDF for sampling.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), data->prefix_sums.get(), k);
  CUDA_CHECK(cudaGetLastError());

  // Stage 6: Generate random thresholds and sample one token per batch item from the final CDF.
  RandomThresholdKernel<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(data->curand_states.get(), data->thresholds.get(),
                                                                      batch_size);
  CUDA_CHECK(cudaGetLastError());

  SampleKernel<256>
      <<<grid, block, 0, stream>>>(data->prefix_sums.get(), indices, index_out, sample_range, data->thresholds.get());
  CUDA_CHECK(cudaGetLastError());
}

void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, const float* scores_in, int vocab_size,
               int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // Step 1: Get the raw scores (logits) and indices of the top `k` candidates.
  // The results are placed in `data->output_scores` and `data->output_indices`.
  TopkData* topk_data = data;
  GetTopKSubset(topk_data, stream, scores_in, data->output_scores.get(), data->output_indices.get(), vocab_size,
                batch_size, k);

  // Step 2: Launch the sampling kernel to perform Top-P sampling on the selected candidates.
  int sample_range = k;
  LaunchSampleKernel(data, stream, data->output_scores.get(), data->output_indices.get(), next_token_out, sample_range,
                     batch_size, p, k, temperature);
}

}  // namespace cuda
}  // namespace Generators
