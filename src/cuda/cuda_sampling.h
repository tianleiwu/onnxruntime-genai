// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <curand_kernel.h>

#include <memory>

#include "cuda_common.h"
#include "cuda_topk.h"  // For TopkData

namespace Generators {
namespace cuda {

// This struct now ONLY holds buffers for the SAMPLING stage.
// Top-K buffers are managed separately in the TopkData struct.
struct SamplingData : public TopkData {
  SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream);

  // Buffers for final sampling logic
  cuda_unique_ptr<float> prefix_sums;
  cuda_unique_ptr<float> scores_adjusted;
  cuda_unique_ptr<float> prefix_sums_adjusted;
  cuda_unique_ptr<float> thresholds;
  cuda_unique_ptr<curandState> curand_states;

  // Temporary buffers that might be needed by sampling kernels
  cuda_unique_ptr<float> scores_temp_sampling;
  cuda_unique_ptr<float> scores_buffer_sampling;
};

void GetSample(SamplingData* sampling_data, cudaStream_t stream, int32_t* d_next_token, float* d_scores, int vocab_size, int batch_size, int k, float p, float temperature);

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements, int input_stride, int output_stride, int batch_count);

void DispatchBlockwiseSoftmaxForwardWithTemperature(cudaStream_t stream, float* output, const float* input, int softmax_elements, int input_stride, int output_stride, int batch_count, float temperature);

}  // namespace cuda
}  // namespace Generators
