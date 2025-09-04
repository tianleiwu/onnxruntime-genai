// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"

namespace Generators {
namespace cuda {

__global__ void PopulateIndices(int* indices, int vocab_size, int batch_size) {
  int global_index = threadIdx.x + blockIdx.x * blockDim.x;
  int index = global_index % vocab_size;
  if (global_index < vocab_size * batch_size) {
    indices[global_index] = index;
  }
}

void LaunchPopulateIndices(int* indices, int vocab_size, int batch_size, cudaStream_t stream) {
  dim3 grid(CeilDiv(batch_size * vocab_size, 256), 1, 1);
  dim3 block(256, 1, 1);
  PopulateIndices<<<grid, block, 0, stream>>>(indices, vocab_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void PopulateOffsets(int* offsets, int vocab_size, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size + 1) offsets[index] = index * vocab_size;
}

void LaunchPopulateOffsets(int* offsets, int vocab_size, int batch_size, cudaStream_t stream) {
  dim3 grid(CeilDiv(batch_size, 128), 1, 1);
  dim3 block(128, 1, 1);
  PopulateOffsets<<<grid, block, 0, stream>>>(offsets, vocab_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Sorting Kernel Launcher
void LaunchSortPairs(void* d_temp_storage, size_t temp_storage_bytes,
                     const float* d_keys_in, float* d_keys_out, const int* d_values_in,
                     int* d_values_out, int num_items, int num_segments,
                     int* d_offsets, cudaStream_t stream) {
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0,
      sizeof(float) * 8, stream));
}

inline size_t GetFullSortCubTempStorageBytes(int num_items, int num_segments, cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_storage_bytes, (float*)nullptr, (float*)nullptr, (int*)nullptr,
      (int*)nullptr, num_items, num_segments, (int*)nullptr, (int*)nullptr, 0,
      sizeof(float) * 8, stream));
  return temp_storage_bytes;
}

void LaunchSort(TopkData* data, cudaStream_t stream,
                float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size) {
  // Sort indices and scores
  LaunchPopulateOffsets(data->offsets.get(), vocab_size, batch_size, stream);
  LaunchPopulateIndices(data->indices_in.get(), vocab_size, batch_size, stream);
  LaunchSortPairs(data->temp_buffer.get(), data->temp_storage_bytes,
                  scores_in, scores_out, data->indices_in.get(),
                  indices_out, vocab_size * batch_size, batch_size,
                  data->offsets.get(), stream);
}

void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, float* scores_in,
                        float* scores_out, int* indices_out, int vocab_size,
                        int batch_size, int k, float temperature) {
  // Step 1: Perform a full, segmented sort on the input scores.
  float* sorted_scores = data->scores_buffer.get();
  int* sorted_indices = data->indices_in.get();
  LaunchSort(data, stream, scores_in, sorted_scores, sorted_indices, vocab_size, batch_size);

  // Step 2: Launch a specialized kernel that leverages the pre-sorted nature of the data.
  ApplySoftmaxToSortedTopK<true>(stream, scores_out, indices_out, sorted_scores,
                                 sorted_indices, k, batch_size, vocab_size, temperature);
}

}  // namespace cuda
}  // namespace Generators
