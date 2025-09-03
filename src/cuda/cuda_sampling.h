// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_common.h"
#include <curand_kernel.h>

namespace Generators {
namespace cuda {

// This struct holds all the device memory buffers required for both Top-K and Sampling operations.
// It acts as a shared workspace to avoid reallocating memory between pipeline stages.
struct SamplingData {
  SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream);
  cuda_unique_ptr<int> indices_sorted;
  cuda_unique_ptr<float> scores_sorted;
  cuda_unique_ptr<float> scores_buffer;
  cuda_unique_ptr<float> prefix_sums;
  cuda_unique_ptr<float> scores_temp;
  cuda_unique_ptr<float> scores_adjusted;
  cuda_unique_ptr<float> prefix_sums_adjusted;
  cuda_unique_ptr<float> thresholds;
  cuda_unique_ptr<int> indices_in;
  cuda_unique_ptr<int> offsets;
  cuda_unique_ptr<unsigned char> temp_buffer;
  cuda_unique_ptr<curandState> curand_states;
  size_t temp_storage_bytes = 0;
};

void GetSample(SamplingData* data, cudaStream_t stream, int32_t* d_next_token, float* d_scores, int vocab_size, int batch_size, int k, float p, float temperature);

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements, int input_stride, int output_stride, int batch_count);

void DispatchBlockwiseSoftmaxForwardWithTemperature(cudaStream_t stream, float* output, const float* input, int softmax_elements, int input_stride, int output_stride, int batch_count, float temperature);

}  // namespace cuda
}  // namespace Generators
