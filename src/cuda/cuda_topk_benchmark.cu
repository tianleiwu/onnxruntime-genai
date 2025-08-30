// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_helper.h"
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>

namespace Generators {
namespace cuda {

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

const char* AlgorithmToString(TopKAlgorithm algo) {
  switch (algo) {
    case TopKAlgorithm::SELECTION_SORT:
      return "SELECTION_SORT";
    case TopKAlgorithm::BITONIC_SORT:
      return "BITONIC_SORT";
    case TopKAlgorithm::FULL_SORT:
      return "FULL_SORT";
    default:
      return "UNKNOWN";
  }
}

#define CUDA_CHECK_WITH_CONFIG(call, config)                                    \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "\n--- CUDA Error during benchmark ---\n");               \
      fprintf(stderr, "Algorithm:  %s\n", AlgorithmToString(config.algorithm)); \
      fprintf(stderr, "Partitions: %d\n", config.num_partitions);               \
      fprintf(stderr, "SortSize:  %d\n", config.sort_size);                    \
      fprintf(stderr, "Block Size: %d\n", config.block_size);                   \
      fprintf(stderr, "Error:      %s\n", cudaGetErrorString(err));             \
      fprintf(stderr, "Location:   %s:%d\n", __FILE__, __LINE__);               \
      fprintf(stderr, "-------------------------------------\n");               \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

// Performs a one-time benchmark to find the fastest Top-K algorithm for a given configuration.
TopKConfig BenchmarkAndGetBestAlgorithm(SamplingData* data, cudaStream_t stream, int vocab_size, int batch_size, int k) {
  assert(k <= 64);
  BenchmarkingCacheKey key = {vocab_size, batch_size, k};
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = algorithm_cache.find(key);
  if (it != algorithm_cache.end()) return it->second;

  auto d_rand_scores = CudaMallocArray<float>(vocab_size * batch_size);
  auto d_rand_indices = CudaMallocArray<int>(k * batch_size);
  auto d_rand_out = CudaMallocArray<float>(k * batch_size);
  int total_size = vocab_size * batch_size;
  RandomTopkInput(stream, d_rand_scores.get(), data->curand_states.get(), total_size, batch_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  constexpr int warmup_runs = 2, timing_runs = 5;
  float temperature = 1.0f;

  struct Result {
    TopKConfig config;
    float time;
  };

  std::vector<Result> results;

  auto benchmark_algorithm = [&](TopKConfig config, auto func) {
    for (int i = 0; i < warmup_runs; ++i) {
      func();
    }
    // Check for any async errors from the warmup runs. cudaGetLastError resets the error state.
    CUDA_CHECK_WITH_CONFIG(cudaGetLastError(), config);

    CUDA_CHECK_WITH_CONFIG(cudaEventRecord(start, stream), config);
    for (int i = 0; i < timing_runs; ++i) {
      func();
    }
    // Check for async errors from the timed runs before stopping the timer.
    CUDA_CHECK_WITH_CONFIG(cudaGetLastError(), config);

    CUDA_CHECK_WITH_CONFIG(cudaEventRecord(stop, stream), config);
    CUDA_CHECK_WITH_CONFIG(cudaEventSynchronize(stop), config);
    float ms = 0.0f;
    CUDA_CHECK_WITH_CONFIG(cudaEventElapsedTime(&ms, start, stop), config);
    results.push_back({config, ms / timing_runs});
  };

  benchmark_algorithm({TopKAlgorithm::SELECTION_SORT}, [&]() { RunTopKViaSelectionSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature); });

  for (int sort_size : {512, 1024, 2048, 4096}) {
    for (int num_partitions : {32, 64, 128, 256}) {
      assert(num_partitions <= kBitonicSortMaxPartitions);
      // Check if the partition size is valid for the given sort_size
      if (vocab_size <= sort_size * num_partitions && vocab_size > sort_size * num_partitions / 2) {
        benchmark_algorithm({TopKAlgorithm::BITONIC_SORT, num_partitions, 256, sort_size}, [&]() {
          RunTopKViaMapReduceBitonicSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature, num_partitions, sort_size);
        });
      }
    }
  }

  benchmark_algorithm({TopKAlgorithm::FULL_SORT}, [&]() { RunTopKViaFullSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature); });

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  auto best_it = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
  TopKConfig winner = best_it->config;
  algorithm_cache[key] = winner;
  return winner;
}

}  // namespace cuda
}  // namespace Generators