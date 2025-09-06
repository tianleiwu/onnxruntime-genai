// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// This file is adapted from the ONNX Runtime project.
// The original code can be found at:
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/math/topk_impl.cuh

#include "cuda_topk.h"
#include <limits>
#include <cmath>

namespace Generators {
namespace cuda {
namespace bitonic {

template <typename T>
struct KV {
  T key;
  int val;
};

// Macros adapted from topk_impl.cuh (assuming largest=1, sorted=1, i.e. the output is a sorted list of the largest k elements)
#define TRIVIAL (std::numeric_limits<float>::lowest())
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? m : n)))
#define IS_SMALLER(n, m) (n.key < m.key || (!(n.key > m.key) && n.val > m.val))

__global__ void BitonicTopK(const float* X, float* V, int* I, int K, int aligned_K, int vocab_size, int aligned_vocab_size) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;
  extern __shared__ char shared_mem[];
  auto S = (KV<float>*)(shared_mem);

  const float* X_batch = X + static_cast<size_t>(bid) * vocab_size;
  float* V_batch = V + static_cast<size_t>(bid) * K;
  int* I_batch = I + static_cast<size_t>(bid) * K;

  for (auto i = tid; i < aligned_vocab_size; i += bdim) {
    S[i].key = i < vocab_size ? X_batch[i] : TRIVIAL;
    S[i].val = i;
  }
  __syncthreads();

  for (int len = 1; len < aligned_K; len <<= 1) {
    auto dir = len << 1;
    for (auto inc = len; inc > 0; inc >>= 1) {
      auto low = tid & (inc - 1);
      auto i = (tid << 1) - low;
      auto j = i + inc;
      if (j < aligned_vocab_size) {
        auto reverse = (dir & i) == 0;
        auto swap = reverse ^ IS_SMALLER(S[i], S[j]);
        if (swap) {
          auto tmp = S[i];
          S[i] = S[j];
          S[j] = tmp;
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }

  for (int len = aligned_K; len < aligned_vocab_size; len <<= 1) {
    auto dir = len << 1;
    auto i = (tid << 1) - (tid & (len - 1));
    auto j = i + len;
    if (i % dir < aligned_K && j < aligned_vocab_size) {
      S[i] = BIGGER(S[i], S[j]);
    }
    __syncthreads();
    for (auto inc = aligned_K >> 1; inc > 0; inc >>= 1) {
      auto ii = (tid << 1) - (tid & (inc - 1));
      auto jj = ii + inc;
      if (ii % dir < aligned_K && jj < aligned_vocab_size) {
        auto reverse = (dir & ii) == 0;
        auto swap = reverse ^ IS_SMALLER(S[ii], S[jj]);
        if (swap) {
          auto tmp = S[ii];
          S[ii] = S[jj];
          S[jj] = tmp;
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }

  auto start = aligned_K - K;
  if (tid >= start && tid < aligned_K) {
    auto to = aligned_K - 1 - tid;
    V_batch[to] = S[tid].key;
    I_batch[to] = S[tid].val;
  }
}

} // namespace bitonic

#define ALIGN(N) static_cast<int>(pow(2, ceil(log2(static_cast<double>(N)))))

void RunTopKViaBitonicSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
    using namespace bitonic;
    int aligned_vocab_size = ALIGN(vocab_size);
    int aligned_k = ALIGN(k);
    constexpr int block_size = 256;
    BitonicTopK<<<batch_size, block_size, aligned_vocab_size * sizeof(KV<float>), stream>>>(
        scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, aligned_k, vocab_size,
        aligned_vocab_size);
    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
    data->topk_stride = k;
}

#undef ALIGN
#undef TRIVIAL
#undef BIGGER
#undef IS_SMALLER

}  // namespace cuda
}  // namespace Generators
