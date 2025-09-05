// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/device/device_segmented_radix_sort.cuh>
#include "cuda_topk.h"
#include "cuda_topk_hybrid_sort.cuh"  // For CompactStridedData

namespace Generators {
namespace cuda {

// Kernel to initialize an index tensor where indices[i] = i % vocab_size.
__global__ void PopulateIndices(int* indices, int vocab_size, int batch_size) {
  int global_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_index < vocab_size * batch_size) {
    indices[global_index] = global_index % vocab_size;
  }
}

void LaunchPopulateIndices(int* indices, int vocab_size, int batch_size, cudaStream_t stream) {
  dim3 grid(CeilDiv(batch_size * vocab_size, 256), 1, 1);
  dim3 block(256, 1, 1);
  PopulateIndices<<<grid, block, 0, stream>>>(indices, vocab_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Kernel to create the segment offsets required by CUB's segmented sort.
// Each batch item is a segment.
__global__ void PopulateOffsets(int* offsets, int vocab_size, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size + 1) {
    offsets[index] = index * vocab_size;
  }
}

void LaunchPopulateOffsets(int* offsets, int vocab_size, int batch_size, cudaStream_t stream) {
  // We need batch_size + 1 offsets for CUB's segmented sort.
  dim3 grid(CeilDiv(batch_size + 1, 256), 1, 1);
  dim3 block(256, 1, 1);
  PopulateOffsets<<<grid, block, 0, stream>>>(offsets, vocab_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Wrapper for CUB's segmented radix sort for (float, int) key-value pairs.
void LaunchSortPairs(void* d_temp_storage, size_t temp_storage_bytes, const float* d_keys_in, float* d_keys_out,
                     const int* d_values_in, int* d_values_out, int num_items, int num_segments, int* d_offsets,
                     cudaStream_t stream) {
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments,
      d_offsets, d_offsets + 1, 0, sizeof(float) * 8, stream));
}

// Helper to determine the size of temporary storage needed by CUB for the sort.
inline size_t GetFullSortCubTempStorageBytes(int num_items, int num_segments, cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_storage_bytes, (float*)nullptr, (float*)nullptr, (int*)nullptr, (int*)nullptr, num_items,
      num_segments, (int*)nullptr, (int*)nullptr, 0, sizeof(float) * 8, stream));
  return temp_storage_bytes;
}

// Performs a full sort on the entire vocabulary for each batch item.
void LaunchSort(TopkData* data, cudaStream_t stream, const float* scores_in, float* scores_out, int* indices_out,
                int vocab_size, int batch_size) {
  LaunchPopulateOffsets(data->batch_offsets.get(), vocab_size, batch_size, stream);
  LaunchPopulateIndices(data->intermediate_indices_1.get(), vocab_size, batch_size, stream);
  LaunchSortPairs(data->cub_temp_storage.get(), data->cub_temp_storage_bytes, scores_in, scores_out,
                  data->intermediate_indices_1.get(), indices_out, vocab_size * batch_size, batch_size,
                  data->batch_offsets.get(), stream);
}

// Interface for running Top-K using a full sort.
void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, const float* scores_in, float* scores_out,
                        int* indices_out, int vocab_size, int batch_size, int k) {
  // Step 1: Perform a full, segmented sort on the input scores.
  // The fully sorted raw scores are placed in `intermediate_scores_1`.
  // The fully sorted indices are placed in `intermediate_indices_1`.
  float* full_sorted_scores = data->intermediate_scores_1.get();
  int* full_sorted_indices = data->intermediate_indices_1.get();
  LaunchSort(data, stream, scores_in, full_sorted_scores, full_sorted_indices, vocab_size, batch_size);

  // Step 2: The results are now sorted but strided by `vocab_size`.
  // We must extract the top `k` elements and place them into the compact
  // [batch_size, k] output buffers to fulfill the API contract.
  CompactStridedData<float><<<batch_size, 256, 0, stream>>>(full_sorted_scores, scores_out, k, batch_size, vocab_size);
  CompactStridedData<int><<<batch_size, 256, 0, stream>>>(full_sorted_indices, indices_out, k, batch_size, vocab_size);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace Generators
