// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk.h"
#include <cub/cub.cuh>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// Robust CUDA error checking macro
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

namespace Generators {
namespace cuda {

// Populate Kernels and Launchers
__global__ void PopulateIndices(int* indices, int size, int batch_size) {
  int global_index = threadIdx.x + blockIdx.x * blockDim.x;
  int index = global_index % size;
  if (global_index < size * batch_size) {
    indices[global_index] = index;
  }
}

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream) {
  dim3 grid((batch_size * size + 255) / 256, 1, 1);
  dim3 block(256, 1, 1);
  PopulateIndices<<<grid, block, 0, stream>>>(indices, size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void PopulateOffsets(int* offsets, int size, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size + 1)
    offsets[index] = index * size;
}

void LaunchPopulateOffsets(int* offsets, int size, int batch_size, cudaStream_t stream) {
  dim3 grid((batch_size + 1 + 127) / 128, 1, 1);
  dim3 block(128, 1, 1);
  PopulateOffsets<<<grid, block, 0, stream>>>(offsets, size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Sorting Kernel Launcher
void LaunchSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size) {
  LaunchPopulateOffsets(data->offsets.get(), vocab_size, batch_size, stream);
  LaunchPopulateIndices(data->indices_in.get(), vocab_size, batch_size, stream);
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(data->temp_buffer.get(), data->temp_storage_bytes, scores_in, scores_out,
                                                                data->indices_in.get(), indices_out, vocab_size * batch_size, batch_size, data->offsets.get(),
                                                                data->offsets.get() + 1, 0, sizeof(float) * 8, stream));
}

// START of improved Top-K kernel (Selection Sort approach)
struct TopK_2 {
  int p = INT_MAX;
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

__device__ __forceinline__ TopK_2 reduce_topk_op_2(TopK_2 const& a, TopK_2 const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p < b.p) ? a
                                                   : b;
}

template <int kBlockSize>
__global__ void GetTopKKernelRaw(int* indices_out, float* scores_in, float* scores_out, int batch_size, int vocab_size, int k) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_2 partial;
  float const MAX_T_VAL = FLT_MAX;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
    for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
      float elem = scores_in[elemId + batch * vocab_size];
      partial.insert(elem, elemId);
    }
    typedef cub::BlockReduce<TopK_2, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_2 top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2);

    if (tid == 0) {
      scores_out[ite + batch * k] = top_k_sequence.u;
      indices_out[ite + batch * k] = top_k_sequence.p;
      scores_in[batch * vocab_size + top_k_sequence.p] = -MAX_T_VAL;
    }
    __threadfence_block();
    __syncthreads();
  }
}

void LaunchImprovedGetTopKRaw(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(1024, 1, 1);
  GetTopKKernelRaw<1024><<<grid, block, 0, stream>>>(indices_out, scores_in, scores_out, batch_size, vocab_size, k);
  CUDA_CHECK(cudaGetLastError());
}

template <int kBlockSize>
__global__ void CopyAndSoftmaxKernel(int* final_indices, float* final_scores,
                                     const int* sorted_indices, const float* sorted_scores,
                                     int k, float temperature, int input_stride) {
  const int batch_idx = blockIdx.x;
  const int* batch_sorted_indices = sorted_indices + batch_idx * input_stride;
  const float* batch_sorted_scores = sorted_scores + batch_idx * input_stride;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce_max;
    typename BlockReduce::TempStorage reduce_sum;
  } temp_storage;

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    final_indices[batch_idx * k + i] = batch_sorted_indices[i];
  }

  float thread_score = -std::numeric_limits<float>::max();
  if (threadIdx.x < k) {
    thread_score = batch_sorted_scores[threadIdx.x] / temperature;
  }

  float max_val = BlockReduce(temp_storage.reduce_max).Reduce(thread_score, cub::Max());
  __syncthreads();

  float thread_exp = 0.0f;
  if (threadIdx.x < k) {
    thread_exp = expf(thread_score - max_val);
  }

  float sum_exp = BlockReduce(temp_storage.reduce_sum).Reduce(thread_exp, cub::Sum());
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    float scaled_score = batch_sorted_scores[i] / temperature;
    final_scores[batch_idx * k + i] = expf(scaled_score - max_val) / sum_exp;
  }
}

template <int kBlockSize, int kSortSize>
__global__ void FindBlockTopK_BitonicSort(const float* scores_in,
                                          int* intermediate_indices,
                                          float* intermediate_scores,
                                          int vocab_size,
                                          int num_partitions) {
  __shared__ float smem_scores[kSortSize];
  __shared__ int smem_indices[kSortSize];
  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const float* batch_scores_in = scores_in + batch_idx * vocab_size;
  const int partition_size = (vocab_size + num_partitions - 1) / num_partitions;
  const int partition_start = partition_idx * partition_size;

  for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
    int global_idx = partition_start + i;
    if (i < partition_size && global_idx < vocab_size) {
      smem_scores[i] = batch_scores_in[global_idx];
      smem_indices[i] = global_idx;
    } else {
      smem_scores[i] = -std::numeric_limits<float>::max();
      smem_indices[i] = -1;
    }
  }
  __syncthreads();

  for (int k = 2; k <= kSortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0) {
            if (smem_scores[i] > smem_scores[ixj]) {
              float temp_s = smem_scores[i]; smem_scores[i] = smem_scores[ixj]; smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i]; smem_indices[i] = smem_indices[ixj]; smem_indices[ixj] = temp_i;
            }
          } else {
            if (smem_scores[i] < smem_scores[ixj]) {
              float temp_s = smem_scores[i]; smem_scores[i] = smem_scores[ixj]; smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i]; smem_indices[i] = smem_indices[ixj]; smem_indices[ixj] = temp_i;
            }
          }
        }
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < kSortSize / 2; i += kBlockSize) {
    if (smem_scores[i] < smem_scores[kSortSize - 1 - i]) {
      float temp_s = smem_scores[i]; smem_scores[i] = smem_scores[kSortSize - 1 - i]; smem_scores[kSortSize - 1 - i] = temp_s;
      int temp_i = smem_indices[i]; smem_indices[i] = smem_indices[kSortSize - 1 - i]; smem_indices[kSortSize - 1 - i] = temp_i;
    }
  }
  __syncthreads();

  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

void RunTopKViaSelectionSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  // The selection sort kernel modifies scores_in, so the calling test must pass a copy.
  float* raw_top_k_scores = data->scores_buffer.get();
  int* temp_indices = data->indices_sorted.get();

  LaunchImprovedGetTopKRaw(stream, scores_in, raw_top_k_scores, temp_indices, vocab_size, batch_size, k);

  dim3 grid(batch_size);
  dim3 block(256);
  CopyAndSoftmaxKernel<256><<<grid, block, 0, stream>>>(indices_out, scores_out, temp_indices, raw_top_k_scores, k, temperature, k);
  CUDA_CHECK(cudaGetLastError());
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  constexpr int block_size = 256;
  float* intermediate_scores = data->scores_temp.get(); // Use scores_temp for intermediate results
  int* intermediate_indices = data->indices_in.get();

  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  switch (sort_size) {
    case 512: FindBlockTopK_BitonicSort<block_size, 512><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions); break;
    case 1024: FindBlockTopK_BitonicSort<block_size, 1024><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions); break;
    case 2048: FindBlockTopK_BitonicSort<block_size, 2048><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions); break;
    case 4096: FindBlockTopK_BitonicSort<block_size, 4096><<<grid_stage1, block_stage1, 0, stream>>>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions); break;
    default: assert(false && "Unsupported sort_size"); break;
  }
  CUDA_CHECK(cudaGetLastError());

  int num_intermediate_results_per_batch = num_partitions * kBitonicSortMaxK;
  int total_intermediate_results = batch_size * num_intermediate_results_per_batch;
  float* final_raw_scores = data->scores_buffer.get(); // Final raw scores go here
  int* sorted_indices = data->indices_sorted.get();

  LaunchPopulateOffsets(data->offsets.get(), num_intermediate_results_per_batch, batch_size, stream);

  size_t temp_storage_bytes_needed = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes_needed, intermediate_scores, final_raw_scores, intermediate_indices, sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));
  if (data->temp_storage_bytes < temp_storage_bytes_needed) {
    std::cerr << "FATAL ERROR in RunTopKViaMapReduceBitonicSort: Pre-allocated temp_buffer is too small." << std::endl;
    return;
  }
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(data->temp_buffer.get(), temp_storage_bytes_needed, intermediate_scores, final_raw_scores, intermediate_indices, sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));

  dim3 grid_stage3(batch_size);
  dim3 block_stage3(256);
  CopyAndSoftmaxKernel<256><<<grid_stage3, block_stage3, 0, stream>>>(indices_out, scores_out, sorted_indices, final_raw_scores, k, temperature, num_intermediate_results_per_batch);
  CUDA_CHECK(cudaGetLastError());
}

void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  float* sorted_scores = data->scores_buffer.get();
  int* sorted_indices = data->indices_in.get();
  LaunchSort(data, stream, scores_in, sorted_scores, sorted_indices, vocab_size, batch_size);

  dim3 grid(batch_size);
  dim3 block(256);
  CopyAndSoftmaxKernel<256><<<grid, block, 0, stream>>>(indices_out, scores_out, sorted_indices, sorted_scores, k, temperature, vocab_size);
  CUDA_CHECK(cudaGetLastError());
}

enum class TopKAlgorithm { SELECTION_SORT, BITONIC_SORT, FULL_SORT };
struct TopKConfig {
  TopKAlgorithm algorithm = TopKAlgorithm::FULL_SORT;
  int num_partitions = 0;
  int block_size = 256;
  int sort_size = 0;
};

using BenchmarkingCacheKey = std::tuple<int, int, int>;
static std::map<BenchmarkingCacheKey, TopKConfig> algorithm_cache;
static std::mutex cache_mutex;

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

TopKConfig BenchmarkAndGetBestAlgorithm(SamplingData* data, cudaStream_t stream, int vocab_size, int batch_size, int k) {
  assert(k <= 64);
  BenchmarkingCacheKey key = {vocab_size, batch_size, k};
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (auto it = algorithm_cache.find(key); it != algorithm_cache.end()) return it->second;

  auto d_rand_scores_owner = CudaMallocArray<float>(vocab_size * batch_size);
  auto d_rand_scores = d_rand_scores_owner.get();
  auto d_rand_scores_copy_owner = CudaMallocArray<float>(vocab_size * batch_size);
  auto d_rand_scores_copy = d_rand_scores_copy_owner.get();
  auto d_rand_indices = CudaMallocArray<int>(k * batch_size);
  auto d_rand_out = CudaMallocArray<float>(k * batch_size);
  RandomTopkInput(stream, d_rand_scores, data->curand_states.get(), vocab_size * batch_size, batch_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  struct Result {
    TopKConfig config;
    float time;
  };
  std::vector<Result> results;

  auto benchmark_algorithm = [&](TopKConfig config, auto func, bool copy_input) {
    if (copy_input) {
        cudaMemcpy(d_rand_scores_copy, d_rand_scores, sizeof(float) * vocab_size * batch_size, cudaMemcpyDeviceToDevice);
    }
    for (int i = 0; i < 2; ++i) func(); // Warmup
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 5; ++i) func(); // Timed
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    results.push_back({config, ms / 5});
  };

  benchmark_algorithm({TopKAlgorithm::SELECTION_SORT}, [&]() { RunTopKViaSelectionSort(data, stream, d_rand_scores_copy, d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, 1.0f); }, true);

  for (int sort_size : {512, 1024, 2048, 4096}) {
    for (int num_partitions : {32, 64, 128, 256}) {
      if (vocab_size <= sort_size * num_partitions && vocab_size > sort_size * num_partitions / 2) {
        benchmark_algorithm({TopKAlgorithm::BITONIC_SORT, num_partitions, 256, sort_size}, [&]() {
          RunTopKViaMapReduceBitonicSort(data, stream, d_rand_scores, d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, 1.0f, num_partitions, sort_size);
        }, false);
      }
    }
  }

  benchmark_algorithm({TopKAlgorithm::FULL_SORT}, [&]() { RunTopKViaFullSort(data, stream, d_rand_scores, d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, 1.0f); }, false);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  auto best_it = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
  return algorithm_cache[key] = best_it->config;
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  TopKConfig chosen_config;
  if (k <= 8) {
    chosen_config.algorithm = TopKAlgorithm::SELECTION_SORT;
  } else if (k > 64) {
    chosen_config.algorithm = TopKAlgorithm::FULL_SORT;
  } else {
    chosen_config = BenchmarkAndGetBestAlgorithm(data, stream, vocab_size, batch_size, k);
  }

  // Since SelectionSort modifies the input scores, we must make a copy before calling it.
  // The other algorithms do not modify the input.
  if (chosen_config.algorithm == TopKAlgorithm::SELECTION_SORT) {
      cudaMemcpyAsync(data->scores_temp.get(), scores_in, sizeof(float) * vocab_size * batch_size, cudaMemcpyDeviceToDevice, stream);
      RunTopKViaSelectionSort(data, stream, data->scores_temp.get(), scores_out, indices_out, vocab_size, batch_size, k, temperature);
  } else {
      switch (chosen_config.algorithm) {
          case TopKAlgorithm::BITONIC_SORT:
              RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, chosen_config.num_partitions, chosen_config.sort_size);
              break;
          default: // FULL_SORT
              RunTopKViaFullSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
              break;
      }
  }
}

}  // namespace cuda
}  // namespace Generators



