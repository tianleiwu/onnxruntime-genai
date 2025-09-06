// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>  // For FLT_MAX
#include <cub/cub.cuh>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// A simple struct to hold a key-value pair for reduction.
struct TopK_Pair {
  int p = -1;
  float u = -FLT_MAX;

  __device__ __forceinline__ void insert(float elem, int elem_id) {
    if (elem > u || (elem == u && elem_id < p)) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -FLT_MAX;
    p = -1;
  }
};

// CUB reduction operator for finding the maximum value and its index.
__device__ __forceinline__ TopK_Pair reduce_topk_op(TopK_Pair const& a, TopK_Pair const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p < b.p) ? a
                                                   : b;
}

// Kernel to find the top K elements using iterative selection sort.
// In each of the k iterations, it finds the single largest element remaining in the vocabulary.
template <int kBlockSize>
__global__ void GetTopKKernel(float* scores_in, float* scores_out, int* indices_out, int batch_size, int vocab_size,
                              int k) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_Pair partial;

  // Use a very small number to blank out selected scores, avoiding picking them again.
  constexpr float MIN_FLOAT = -std::numeric_limits<float>::max();

  for (int ite = 0; ite < k; ite++) {
    partial.init();
    // Each thread block processes one batch item.
    // Threads cooperatively scan the vocabulary to find the max element.
    for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
      float elem = scores_in[elemId + batch * vocab_size];
      partial.insert(elem, elemId);
    }
    // Reduce within the thread block to find the block's top element.
    typedef cub::BlockReduce<TopK_Pair, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_Pair top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op);

    // Thread 0 writes the result and blanks out the selected score.
    if (tid == 0) {
      scores_out[ite + batch * k] = top_k_sequence.u;
      indices_out[ite + batch * k] = top_k_sequence.p;

      // Set the max value to a large negative number so it isn't picked again.
      scores_in[batch * vocab_size + top_k_sequence.p] = MIN_FLOAT;
      __threadfence_block();  // Ensure the write is visible to other threads in the next iteration.
    }
    __syncthreads();
  }
}

void LaunchGetTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size,
                   int batch_size, int k) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(1024, 1, 1);  // Use a large block size for better hardware utilization.
  GetTopKKernel<1024><<<grid, block, 0, stream>>>(scores_in, scores_out, indices_out, batch_size, vocab_size, k);
  CUDA_CHECK(cudaGetLastError());
}

void RunTopKViaSelectionSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  // This function implements a "copy-on-write" strategy for safety and performance.
  // 1. It copies the const `scores_in` to a mutable intermediate buffer.
  // 2. It runs the high-performance in-place selection sort kernel on that buffer.
  // This preserves the const-correctness of the API while using the fastest kernel.
  float* mutable_scores = data->intermediate_scores_2.get();
  size_t buffer_size = static_cast<size_t>(batch_size) * vocab_size * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));

  float* topk_scores = data->intermediate_scores_1.get();
  int* topk_indices = data->intermediate_indices_1.get();
  LaunchGetTopK(stream, mutable_scores, topk_scores, topk_indices, vocab_size, batch_size, k);

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}
}  // namespace cuda
}  // namespace Generators
