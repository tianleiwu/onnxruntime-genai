// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"

namespace Generators {
namespace cuda {

__global__ void FillRandom(float* array, curandState* states, int n, int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int batch_idx = (static_cast<long long>(i) * batch_size) / n;
    array[i] = curand_uniform(&states[batch_idx]);
  }
}

void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size) {
  FillRandom<<<(total_size + 255) / 256, 256, 0, stream>>>(data, batch_state, total_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

int GetHybridSortPartitionSize(int vocab_size, int batch_size) {
  if (vocab_size >= 147456) {
    return (vocab_size > 256 * 1024) ? 8192 : 4096;
  } else {
    if (vocab_size >= 65536 || batch_size >= 4 && vocab_size > 49152) {
      return 2048;
    }
  }

  return 1024;
}

bool UseSelectSort(int vocab_size, int batch_size, int k) {
  assert(k <= 64);

  if (k <= 4 || vocab_size < 1024 || (k <= 8 && vocab_size < 147456)) {
    return true;
  }

  return false;
}

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream) {
  // The intermediate buffers are used by hybrid sort algorithms.
  int partition_size = GetHybridSortPartitionSize(vocab_size, batch_size);
  size_t intermediate_buffer_elements = GetHybridSortIntermediateSize(batch_size, vocab_size, partition_size);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  // Selection sort uses buffer of batch_size * 64 elements, which is smaller than intermediate_buffer_elements.
  size_t max_buffer_elements = std::max(vocab_batch_size, intermediate_buffer_elements);

  this->intermediate_indices = CudaMallocArray<int>(max_buffer_elements);
  this->intermediate_scores_1 = CudaMallocArray<float>(max_buffer_elements);
  this->intermediate_scores_2 = CudaMallocArray<float>(max_buffer_elements);
  this->topk_indices = CudaMallocArray<int>(max_buffer_elements);
  this->topk_probs = CudaMallocArray<float>(max_buffer_elements);
  this->batch_offsets = CudaMallocArray<int>(batch_size + 1);

  this->cub_temp_storage_bytes = GetFullSortCubTempStorageBytes(vocab_batch_size, batch_size, stream);
  this->cub_temp_storage = CudaMallocArray<unsigned char>(this->cub_temp_storage_bytes);
}

void GetTopKSubset(TopkData* topk_data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  assert(topk_data != nullptr);
  assert(topk_data->intermediate_indices != nullptr);  // The caller shall allocate the buffer.

  if (k > 64) {
    RunTopKViaFullSort(topk_data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
    return;
  }

  if (UseSelectSort(vocab_size, batch_size, k)) {
    RunTopKViaSelectionSort(topk_data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
    return;
  }

  int partition_size = GetHybridSortPartitionSize(vocab_size, batch_size);
  RunTopKViaHybridSort(topk_data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, partition_size);
}

}  // namespace cuda
}  // namespace Generators
