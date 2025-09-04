// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>

#include "cuda_common.h"

namespace Generators {
namespace cuda {

struct SamplingData;

// This struct holds all the device memory buffers required for Top-K operations.
struct TopkData {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream);
  cuda_unique_ptr<int> indices_in;
  cuda_unique_ptr<float> scores_buffer;  // The raw scores (before temperature and softmax)
  cuda_unique_ptr<float> scores_temp;
  cuda_unique_ptr<unsigned char> temp_buffer;
  cuda_unique_ptr<int> offsets;
  size_t temp_storage_bytes = 0;

  // Buffers for top-k results, which are input to sampling
  cuda_unique_ptr<float> scores_sorted;
  cuda_unique_ptr<int> indices_sorted;
};

void GetTopKSubset(TopkData* topk_data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

enum class TopKAlgorithm { SELECTION_SORT,
                           HYBRID_SORT,
                           FULL_SORT };

template <bool DoCopyIndices>
void ApplySoftmaxToSortedTopK(cudaStream_t stream, float* final_scores,
                              int* final_indices,
                              const float* sorted_input_scores,
                              const int* sorted_input_indices, int k,
                              int batch_size, int input_stride,
                              float temperature);

void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size);
void RunTopKViaSelectionSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int partition_size);

}  // namespace cuda
}  // namespace Generators
