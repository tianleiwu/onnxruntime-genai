// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: cuda_topk_helper.h can be merged into cuda_topk.h
#pragma once

#include "cuda_topk.h"
#include <cub/cub.cuh>
#include <limits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Robust CUDA error checking macro
// TODO: use Generators::OnCudaError instead of exit in CUDA_CHECK
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)
#endif

namespace Generators {
namespace cuda {

inline int CeilDiv(int a, int b){
  return (a + (b - 1)) / b;
}

template<bool DoCopyIndices>
void ApplySoftmaxToSortedTopK(cudaStream_t stream,
                              float* final_scores,
                              int* final_indices,
                              const float* sorted_input_scores,
                              const int* sorted_input_indices,
                              int k,
                              int batch_size,
                              int input_stride,
                              float temperature);

}  // namespace cuda
}  // namespace Generators
