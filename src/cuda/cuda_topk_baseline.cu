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
// #include <cub/block/block_scan.cuh>
// #include <cub/device/device_radix_sort.cuh>
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

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

// Simplified TArray from cuda_utils.h
template <typename T, int32_t capacity = 8>
struct TArray {
  TArray() = default;
  TArray(const std::vector<T>& vec) : size_(static_cast<int32_t>(vec.size())) {
    assert(0 <= size_ && size_ <= capacity);
    memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  __host__ __device__ int32_t Size() const { return size_; }
  __host__ __device__ T& operator[](int32_t index) { return data_[index]; }
  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const { return data_[index]; }

 private:
  int32_t size_ = 0;
  T data_[capacity] = {};
};

template <typename T>
struct KV {
  T key;
  int val;
};

// Macros adapted from topk_impl.cuh (assuming largest=1, sorted=1)
#define BT Generators::cuda::baseline::GridDim::maxThreadsPerBlock
#define ALIGN(N) static_cast<int>(pow(2, ceil(log2(static_cast<double>(N)))))
#define FROM(idx) (left_dim + (idx) * mid_dim + right_dim)
#define TO(idx) (left_dim * K / dimension + (idx) * mid_dim + right_dim)
#define TRIVIAL (std::numeric_limits<float>::lowest())
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? m : n)))
#define SMALLER(n, m) (n.key < m.key ? n : (n.key > m.key ? m : (n.val < m.val ? m : n)))
#define IS_SMALLER(n, m) (n.key < m.key || (!(n.key > m.key) && n.val > m.val))
#define LESS(n, m) ((n) <= (m) ? (n) : (m))

// Helper functions for RadixTopK adapted from topk_impl.cuh for float/int
__device__ __forceinline__ bool Equal(const float& t0, const float& t1) {
  return !(t0 > t1 || t1 > t0);
}

template <typename T>
__device__ __forceinline__ bool SamePrefix(const T* t0, const T* t1, int skip) {
  return ((*t0) ^ (*t1)) >> skip == 0;
}

__device__ __forceinline__ bool SamePrefix(const float* f0, const float* f1, int skip) {
  return SamePrefix((const int*)f0, (const int*)f1, skip);
}

template <typename T>
__device__ __forceinline__ int Radix(const T* t, int skip) {
  return ((*t) >> skip) & 255;
}

__device__ __forceinline__ int Radix(const float* f, int skip) {
  return Radix((const int*)f, skip);
}

template <typename T>
__device__ void SetByte(T* t, int byte) {
  (*t) |= byte;
}

__device__ __forceinline__ void SetByte(float* f, int byte) {
  SetByte((int*)f, byte);
}

// --- End of inlined definitions ---

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

template <int THREADS, int KPT>
__global__ void RadixTopK(const float* X, float* V, int* I, int K, int vocab_size) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  extern __shared__ char shared_mem[];
  auto H = (unsigned int*)shared_mem;

  const float* X_batch = X + static_cast<size_t>(bid) * vocab_size;
  float* V_batch = V + static_cast<size_t>(bid) * K;
  int* I_batch = I + static_cast<size_t>(bid) * K;

  float Kth = 0.0f, sign = 1.0f;
  typedef cub::BlockScan<unsigned int, THREADS> BlockScan;
  typedef cub::BlockReduce<unsigned int, THREADS> BlockReduce;
  typedef cub::BlockRadixSort<float, THREADS, KPT, int> BlockRadixSort;
  __shared__ union {
    typename BlockScan::TempStorage scan;
    typename BlockReduce::TempStorage reduce;
    typename BlockRadixSort::TempStorage sort;
  } temp_storage;

  unsigned int positive = 0, negative = 0;
  for (int x_i = tid; x_i < vocab_size; x_i += blockDim.x) {
    float x = X_batch[x_i];
    if (x > 0.0f) {
      ++positive;
    } else if (x < 0.0f) {
      ++negative;
    }
  }
  __syncthreads();
  positive = BlockReduce(temp_storage.reduce).Sum(positive);
  __syncthreads();
  negative = BlockReduce(temp_storage.reduce).Sum(negative);
  if (0 == tid) {
    H[0] = positive;
    H[1] = negative;
  }
  __syncthreads();
  positive = H[0];
  negative = H[1];
  if (K <= positive || vocab_size - K + 1 <= negative) {
    auto KK = K;
    if (KK > positive) {
      KK = vocab_size - KK + 1;
      sign = -1.0f;
    }
    __syncthreads();
    for (int byte = sizeof(float) - 1; byte > -1; --byte) {
      if (tid < 256) H[tid] = 0;
      __syncthreads();
      auto skip = 8 * byte, prev_skip = 8 * (byte + 1);
      for (int x_i = tid; x_i < vocab_size; x_i += blockDim.x) {
        float x = sign * X_batch[x_i];
        if (x > 0.0f && (byte == sizeof(float) - 1 || SamePrefix(&x, &Kth, prev_skip))) {
          atomicAdd(&H[Radix(&x, skip)], 1);
        }
      }
      __syncthreads();
      for (int radix = 255; radix > 0; --radix) {
        if (H[radix] < KK) {
          KK -= H[radix];
        } else {
          SetByte(&Kth, radix << skip);
          break;
        }
      }
      __syncthreads();
    }
    Kth *= sign;
  }

  unsigned int superior = 0, equal = 0;
  for (int x_i = tid; x_i < vocab_size; x_i += blockDim.x) {
    auto x = X_batch[x_i];
    if (x > Kth) {
      ++superior;
    } else if (Equal(x, Kth)) {
      ++equal;
    }
  }
  __syncthreads();
  auto all_superior = superior;
  all_superior = BlockReduce(temp_storage.reduce).Sum(all_superior);
  if (0 == tid) {
    H[0] = all_superior;
  }
  __syncthreads();
  all_superior = H[0];
  BlockScan(temp_storage.scan).ExclusiveSum(superior, superior);
  __syncthreads();
  BlockScan(temp_storage.scan).ExclusiveSum(equal, equal);
  __syncthreads();
  auto equal_quota = K - all_superior - equal;
  auto output_i = superior + LESS(K - all_superior, equal);
  for (int x_i = tid; x_i < vocab_size; x_i += blockDim.x) {
    auto x = X_batch[x_i];
    if (x > Kth) {
      V_batch[output_i] = x;
      I_batch[output_i] = x_i;
      ++output_i;
    } else if (Equal(x, Kth) && equal_quota > 0) {
      V_batch[output_i] = x;
      I_batch[output_i] = x_i;
      ++output_i;
      --equal_quota;
    }
  }
  __syncthreads();

  // Sort the final K results
  float keys[KPT];
  int vals[KPT];
  for (int k_i = tid, k_c = 0; k_c < KPT; k_i += blockDim.x, ++k_c) {
    if (k_i < K) {
      keys[k_c] = V_batch[k_i];
      vals[k_c] = I_batch[k_i];
    } else {
      keys[k_c] = std::numeric_limits<float>::lowest();
    }
  }
  __syncthreads();
  BlockRadixSort(temp_storage.sort).SortDescending(keys, vals);
  __syncthreads();
  for (int k_c = 0; k_c < KPT; ++k_c) {
    auto k_i = tid * KPT + k_c;
    if (k_i < K) {
      V_batch[k_i] = keys[k_c];
      I_batch[k_i] = vals[k_c];
    }
  }
}

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

void RunTopKViaBaselineSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  using namespace Generators::cuda::baseline;

  int aligned_vocab_size = ALIGN(vocab_size);
  if (aligned_vocab_size <= BT) {
    int aligned_k = ALIGN(k);
    BitonicTopK<<<batch_size, BT, aligned_vocab_size * sizeof(KV<float>), stream>>>(
        scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, aligned_k, vocab_size,
        aligned_vocab_size);
    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
    data->topk_stride = k;
    return;
  }

  if (k <= BT * 16) {
    if (BT * 2 >= k) {
      RadixTopK<BT, 2><<<batch_size, BT, 256 * sizeof(unsigned int), stream>>>(scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, vocab_size);
    } else if (BT * 4 >= k) {
      RadixTopK<BT, 4><<<batch_size, BT, 256 * sizeof(unsigned int), stream>>>(scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, vocab_size);
    } else if (BT * 8 >= k) {
      RadixTopK<BT, 8><<<batch_size, BT, 256 * sizeof(unsigned int), stream>>>(scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, vocab_size);
    } else {
      RadixTopK<BT, 16><<<batch_size, BT, 256 * sizeof(unsigned int), stream>>>(scores_in, data->intermediate_scores_1.get(), data->intermediate_indices_1.get(), k, vocab_size);
    }
    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
    data->topk_stride = k;
    return;
  }

  // Path 3: General case using CUB's device-wide radix sort inside a loop over batches.
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size, 0, sizeof(float) * 8, stream);
  if (data->cub_temp_storage_bytes < temp_bytes) {
      data->cub_temp_storage = CudaMallocArray<unsigned char>(temp_bytes);
      data->cub_temp_storage_bytes = temp_bytes;
  }
  auto* temp_storage = data->cub_temp_storage.get();

  auto* input_key = data->intermediate_scores_1.get();
  auto* output_key = data->intermediate_scores_2.get();
  auto* input_value = data->intermediate_indices_1.get();
  auto* output_value = data->intermediate_indices_2.get();

  auto blocks_per_grid_V = (int)ceil(static_cast<float>(vocab_size) / BT);
  auto blocks_per_grid_K = (int)ceil(static_cast<float>(k) / BT);

  for (int i = 0; i < batch_size; i++) {
    const float* current_scores_in = scores_in + static_cast<size_t>(i) * vocab_size;
    float* current_scores_out = data->intermediate_scores_1.get() + static_cast<size_t>(i) * k;
    int* current_indices_out = data->intermediate_indices_1.get() + static_cast<size_t>(i) * k;

    FillInput<<<blocks_per_grid_V, BT, 0, stream>>>(current_scores_in, input_key, input_value, vocab_size);
    cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, vocab_size, 0, sizeof(float) * 8, stream);
    FillOutput<<<blocks_per_grid_K, BT, 0, stream>>>(output_key, output_value, current_scores_out, current_indices_out, k);
  }
  data->topk_scores = data->intermediate_scores_1.get();
  data->topk_indices = data->intermediate_indices_1.get();
  data->topk_stride = k;
}

#undef BT
#undef ALIGN
#undef FROM
#undef TO
#undef TRIVIAL
#undef BIGGER
#undef SMALLER
#undef IS_SMALLER
#undef LESS
}  // namespace cuda
}  // namespace Generators

