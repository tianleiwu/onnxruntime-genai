// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>

#include "cuda_common.h"

namespace Generators {
namespace cuda {

constexpr int kHybridSortMaxK = 64;  // up to 256.

// This struct holds all the device memory buffers required for Top-K operations.
struct TopkData {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream);

  // --- Intermediate Buffers for Top-K Algorithms ---

  // Used to hold initial vocabulary indices for full sort, and intermediate
  // indices during the reduction phase of hybrid sort.
  cuda_unique_ptr<int> intermediate_indices;

  // Primary buffer for holding raw scores.
  // - Full sort: Holds the fully sorted raw scores.
  // - Selection sort: Holds the top-k raw scores.
  // - Hybrid sort: Holds intermediate and final reduced raw scores.
  // **IMPORTANT**: This buffer is read by the Top-P sampling stage.
  cuda_unique_ptr<float> intermediate_scores_1;

  // A secondary "ping-pong" buffer used by the hybrid sort's reduction phase.
  cuda_unique_ptr<float> intermediate_scores_2;

  // General-purpose temporary storage for CUB's DeviceSegmentedRadixSort (full sort).
  cuda_unique_ptr<unsigned char> cub_temp_storage;
  size_t cub_temp_storage_bytes = 0;

  // Stores the start offset of each batch segment for CUB's segmented sort.
  cuda_unique_ptr<int> batch_offsets;

  // --- Final Output Buffers (Input to Sampling Stage) ---

  // Stores the final top-k probabilities after softmax.
  cuda_unique_ptr<float> topk_probs;

  // Stores the final top-k indices.
  cuda_unique_ptr<int> topk_indices;
};

void GetTopKSubset(TopkData* topk_data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out,
                   int vocab_size, int batch_size, int k, float temperature);

enum class TopKAlgorithm { SELECTION_SORT, HYBRID_SORT, FULL_SORT };

template <bool DoCopyIndices>
void ApplySoftmaxToSortedTopK(cudaStream_t stream, float* final_scores, int* final_indices,
                              const float* sorted_input_scores, const int* sorted_input_indices, int k, int batch_size,
                              int input_stride, float temperature);

void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size);
void RunTopKViaSelectionSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out,
                             int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out,
                        int vocab_size, int batch_size, int k, float temperature);
void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out,
                          int vocab_size, int batch_size, int k, float temperature, int partition_size);

}  // namespace cuda
}  // namespace Generators
