// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// This file is adapted from the ONNX Runtime project.
// The original code can be found at:
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/reduction/topk_impl.cuh

#include "cub/cub.cuh"
#include "cub/util_type.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"

#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>

#include "cuda_topk.h"

namespace Generators {
namespace cuda {

namespace baseline {

// Inlined from common.cuh
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

// Helper to calculate Ceil(a/b)
template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b) {
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

#define BT Generators::cuda::baseline::GridDim::maxThreadsPerBlock

__global__ void FillInput(const float* input_x_batch, float* temp_v, int* temp_i, int vocab_size) {
  for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < vocab_size; id += blockDim.x * gridDim.x) {
    temp_v[id] = input_x_batch[id];
    temp_i[id] = id;
  }
}

__global__ void FillOutput(const float* temp_v, const int* temp_i, float* output_v_batch, int* output_i_batch, int k) {
  for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < k; id += blockDim.x * gridDim.x) {
    output_v_batch[id] = temp_v[id];
    output_i_batch[id] = temp_i[id];
  }
}

}  // namespace baseline

template<bool CompactOutput>
void RunTopKViaBaselineSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  using namespace Generators::cuda::baseline;

  // General case using CUB radix sort.
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size, 0, sizeof(float) * 8, stream);
  if (data->cub_temp_storage_bytes < temp_bytes) {
      data->cub_temp_storage = CudaMallocArray<unsigned char>(temp_bytes);
      data->cub_temp_storage_bytes = temp_bytes;
  }
  auto* temp_storage = data->cub_temp_storage.get();
  
  auto* final_scores_buffer = data->intermediate_scores_1.get();
  auto* final_indices_buffer = data->intermediate_indices_1.get();

  for (int i = 0; i < batch_size; i++) {
    const float* current_scores_in = scores_in + static_cast<size_t>(i) * vocab_size;
    
    auto* workspace_scores = data->intermediate_scores_2.get();
    auto* workspace_indices = data->intermediate_indices_2.get();

    if constexpr (CompactOutput) {
      float* final_scores_out = final_scores_buffer + static_cast<size_t>(i) * k;
      int* final_indices_out = final_indices_buffer + static_cast<size_t>(i) * k;
      
      float* temp_in_scores = workspace_scores;
      int* temp_in_indices = workspace_indices;
      float* temp_out_scores;
      int* temp_out_indices;

      if (batch_size == 1) {
        temp_out_scores = final_scores_buffer;
        temp_out_indices = final_indices_buffer;
      } else {
        temp_out_scores = workspace_scores + vocab_size;
        temp_out_indices = workspace_indices + vocab_size;
      }

      auto blocks_per_grid_V = (int)ceil(static_cast<float>(vocab_size) / BT);
      FillInput<<<blocks_per_grid_V, BT, 0, stream>>>(current_scores_in, temp_in_scores, temp_in_indices, vocab_size);
      
      cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, temp_in_scores, temp_out_scores, temp_in_indices, temp_out_indices, vocab_size, 0, sizeof(float) * 8, stream);

      if (batch_size != 1) {
        auto blocks_per_grid_K = (int)ceil(static_cast<float>(k) / BT);
      FillOutput<<<blocks_per_grid_K, BT, 0, stream>>>(temp_out_scores, temp_out_indices, final_scores_out, final_indices_out, k);
      }
    } else { // Strided output path
      // Sort from workspace directly into the final strided destination buffer.
      float* final_scores_out = final_scores_buffer + static_cast<size_t>(i) * vocab_size;
      int* final_indices_out = final_indices_buffer + static_cast<size_t>(i) * vocab_size;

      auto blocks_per_grid_V = (int)ceil(static_cast<float>(vocab_size) / BT);
      FillInput<<<blocks_per_grid_V, BT, 0, stream>>>(current_scores_in, workspace_scores, workspace_indices, vocab_size);
      
      cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, workspace_scores, final_scores_out, workspace_indices, final_indices_out, vocab_size, 0, sizeof(float) * 8, stream);
    }
  }

  data->topk_scores = final_scores_buffer;
  data->topk_indices = final_indices_buffer;
  data->topk_stride = CompactOutput ? k : vocab_size;
}

#undef BT

template void RunTopKViaBaselineSort<true>(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
template void RunTopKViaBaselineSort<false>(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

}  // namespace cuda
}  // namespace Generators

