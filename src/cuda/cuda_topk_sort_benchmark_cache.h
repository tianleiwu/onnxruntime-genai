// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>
#include <iomanip>
#include <map>

#include "cuda_topk.h"
#include "cuda_topk_warp_sort_helper.cuh"

namespace Generators {
namespace cuda {

// A struct to hold the results of the one-time sort micro-benchmark.
struct SortBenchmarkResults : public ISortAlgoPicker {
  std::vector<int> sort_sizes;
  std::vector<std::vector<float>> latencies;  // [SortAlgo][sort_size_index]
  std::vector<SortAlgo> best_algos;           // [sort_size_index]

  SortBenchmarkResults() : latencies(static_cast<int>(SortAlgo::COUNT)) {}

  // Computes and caches the best algorithm for each benchmarked sort size.
  void ComputeBestAlgos() {
    if (sort_sizes.empty()) {
      return;
    }
    best_algos.resize(sort_sizes.size());
    for (size_t i = 0; i < sort_sizes.size(); ++i) {
      float min_latency = std::numeric_limits<float>::max();
      // Default to a robust, general-purpose algorithm.
      SortAlgo best_algo_for_size = SortAlgo::CUB_BLOCK_MERGE;
      int current_sort_size = sort_sizes[i];

      for (int j = 0; j < static_cast<int>(SortAlgo::COUNT); ++j) {
        bool is_supported = false;
        SortAlgo current_algo = static_cast<SortAlgo>(j);

        // Apply hard architectural constraints for each algorithm.
        switch (current_algo) {
          case SortAlgo::WARP_BITONIC:
            is_supported = (current_sort_size <= 32);
            break;
          case SortAlgo::CUB_WARP_MERGE:
            is_supported = (current_sort_size >= 32 && current_sort_size <= 256);
            break;
          case SortAlgo::CUB_BLOCK_MERGE:
          case SortAlgo::CUB_BLOCK_RADIX:
            is_supported = (current_sort_size >= 64);
            break;
          default:
            is_supported = false;
        }

        if (is_supported && !latencies[j].empty() && latencies[j][i] > 0 && latencies[j][i] < min_latency) {
          min_latency = latencies[j][i];
          best_algo_for_size = current_algo;
        }
      }
      best_algos[i] = best_algo_for_size;
    }
  }

  // Finds the fastest algorithm for a given sort size using the pre-computed cache.
  SortAlgo GetBestAlgo(int sort_size) const override {
    if (best_algos.empty()) {
      // Fallback for safety, though this should not be called on an empty result set.
      return sort_size <= 2048 ? SortAlgo::CUB_BLOCK_MERGE : SortAlgo::CUB_BLOCK_RADIX;
    }

    auto it = std::lower_bound(sort_sizes.begin(), sort_sizes.end(), sort_size);
    if (it == sort_sizes.end()) {
      // If sort_size is larger than any benchmarked size, use the results for the largest size.
      return best_algos.back();
    }
    size_t index = std::distance(sort_sizes.begin(), it);
    return best_algos[index];
  }

  // Gets the latency for a given algo and sort size using the nearest benchmarked point.
  float GetLatency(SortAlgo algo, int sort_size) const override{
    if (sort_sizes.empty() || algo >= SortAlgo::COUNT) {
      return std::numeric_limits<float>::max();
    }

    auto it = std::lower_bound(sort_sizes.begin(), sort_sizes.end(), sort_size);
    if (it == sort_sizes.end()) {
      return std::numeric_limits<float>::max();
    }

    size_t index = std::distance(sort_sizes.begin(), it);
    if (latencies[static_cast<int>(algo)].empty()) {
      return std::numeric_limits<float>::max();
    }

    return latencies[static_cast<int>(algo)][index];
  }
};

/**
 * @brief Measures the average execution time of a CUDA kernel over several runs.
 * This is a lightweight version for online benchmarking, using fewer iterations
 * than an offline profiler to minimize runtime overhead.
 * @param stream The CUDA stream to run the kernel on.
 * @param kernel_func A lambda function that launches the kernel.
 * @return The average execution time in milliseconds.
 */
float TimeKernel(cudaStream_t stream, std::function<void()> kernel_func, int warm_up_runs = 2, int total_runs = 5) {
  cuda_event_holder start_event, stop_event;

  // Warm-up runs to handle any one-time kernel loading costs or JIT compilation.
  for (int i = 0; i < warm_up_runs; ++i) {
    kernel_func();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timed runs.
  CUDA_CHECK(cudaEventRecord(start_event, stream));
  for (int i = 0; i < total_runs; ++i) {
    kernel_func();
  }
  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

  return ms / total_runs;
}

// --- Singleton Cache Manager ---
namespace {  // Anonymous namespace for internal benchmark kernels
#define ENABLE_SORT_BENCHMARK 1
#if ENABLE_SORT_BENCHMARK
__global__ void warpBitonicSortKernel_b(const float* scores_in, float* scores_out, int k) {
  if (threadIdx.x >= 32) return;
  float my_score = scores_in[threadIdx.x];
  int my_index = threadIdx.x;  // Index doesn't matter for latency benchmark
  Generators::cuda::topk_common::WarpBitonicSort(my_score, my_index);
  if (threadIdx.x < k) {
    scores_out[threadIdx.x] = my_score;
  }
}

template <int SORT_SIZE>
__global__ void cubWarpMergeSortKernel_b(const float* scores_in, float* scores_out, int k) {
  constexpr int SORT_SIZE_PO2 = Generators::cuda::topk_common::NextPowerOfTwo(SORT_SIZE);
  union SharedStorage {
    struct {
      float scores[SORT_SIZE_PO2];
      int indices[SORT_SIZE_PO2];
    } sort_data;
    typename cub::WarpMergeSort<float, (SORT_SIZE_PO2 + 31) / 32, 32, int>::TempStorage cub_storage;
  };
  __shared__ SharedStorage smem;

  for (int i = threadIdx.x; i < SORT_SIZE; i += blockDim.x) {
    smem.sort_data.scores[i] = scores_in[i];
    smem.sort_data.indices[i] = i;
  }
  for (int i = SORT_SIZE + threadIdx.x; i < SORT_SIZE_PO2; i += blockDim.x) {
    smem.sort_data.scores[i] = -FLT_MAX;
    smem.sort_data.indices[i] = INT_MAX;
  }
  __syncthreads();

  Generators::cuda::topk_common::WarpMergeSort<SORT_SIZE_PO2>(
      smem.sort_data.scores, smem.sort_data.indices, &smem.cub_storage, SORT_SIZE);
  __syncthreads();

  if (threadIdx.x < k) {
    scores_out[threadIdx.x] = smem.sort_data.scores[threadIdx.x];
  }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockMergeSortKernel_b(const float* scores_in, float* scores_out, int n, int k) {
  using BlockMergeSort = cub::BlockMergeSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockMergeSort::TempStorage temp_storage;
  float thread_scores[ITEMS_PER_THREAD];
  int thread_indices[ITEMS_PER_THREAD];
  cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
  // Create dummy indices for sorting
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) thread_indices[i] = threadIdx.x * ITEMS_PER_THREAD + i;

  BlockMergeSort(temp_storage).Sort(thread_scores, thread_indices, topk_common::DescendingOp());
  cub::StoreDirectBlocked(threadIdx.x, scores_out, thread_scores, k);
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockRadixSortKernel_b(const float* scores_in, float* scores_out, int n, int k) {
  using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  float thread_scores[ITEMS_PER_THREAD];
  int thread_indices[ITEMS_PER_THREAD];
  cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
  // Create dummy indices for sorting
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) thread_indices[i] = threadIdx.x * ITEMS_PER_THREAD + i;

  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_scores, thread_indices);
  if (threadIdx.x < k) {
    scores_out[threadIdx.x] = thread_scores[0];
  }
}

// Helper to convert SortAlgo enum to its string representation for printing.
static const char* SortAlgoToString(SortAlgo algo) {
  switch (algo) {
    case SortAlgo::WARP_BITONIC:
      return "Warp Bitonic";
    case SortAlgo::CUB_WARP_MERGE:
      return "CUB Warp Merge";
    case SortAlgo::CUB_BLOCK_MERGE:
      return "CUB Block Merge";
    case SortAlgo::CUB_BLOCK_RADIX:
      return "CUB Block Radix";
    default:
      return "Unknown";
  }
}

void RunAndCacheSortBenchmark(int device_id, cudaStream_t stream, SortBenchmarkResults& results) {
  std::cout << "Running one-time micro-benchmark for internal sorting primitives on device " << device_id << "..." << std::endl;

  results.sort_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  for (auto& lat_vec : results.latencies) {
    lat_vec.assign(results.sort_sizes.size(), -1.0f);
  }

  const int block_size = 256;

  const int max_n = results.sort_sizes.back();
  auto d_scores_in = CudaMallocArray<float>(max_n);
  auto d_scores_out = CudaMallocArray<float>(max_n);
  CUDA_CHECK(cudaMemsetAsync(d_scores_in.get(), 0, max_n * sizeof(float), stream));

  constexpr int warm_up_runs = 5;
  constexpr int total_runs = 1000;

  for (size_t i = 0; i < results.sort_sizes.size(); ++i) {
    int n = results.sort_sizes[i];
    int k = std::min(n, 64);

    try {
      // --- 1. Warp Bitonic Sort ---
      if (n <= 32) {
        results.latencies[static_cast<int>(SortAlgo::WARP_BITONIC)][i] =
            TimeKernel(stream, [&]() { warpBitonicSortKernel_b<<<1, 32, 0, stream>>>(d_scores_in.get(), d_scores_out.get(), k); }, warm_up_runs, total_runs);
      }

      // --- 2. CUB Warp Merge Sort ---
      if (n >= 32 && n <= 256) {
        auto launch = [&](auto n_const) {
          results.latencies[static_cast<int>(SortAlgo::CUB_WARP_MERGE)][i] =
              TimeKernel(stream, [&]() { cubWarpMergeSortKernel_b<n_const.value><<<1, 64, 0, stream>>>(d_scores_in.get(), d_scores_out.get(), k); }, warm_up_runs, total_runs);
        };
        if (n <= 64)
          launch(std::integral_constant<int, 64>());
        else if (n <= 128)
          launch(std::integral_constant<int, 128>());
        else if (n <= 256)
          launch(std::integral_constant<int, 256>());
      }

      // --- 3. CUB Block Merge Sort ---
      if (n >= 64) {
        const int items_per_thread = (n + block_size - 1) / block_size;
        auto launch = [&](auto ipt_const) {
          results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_MERGE)][i] =
              TimeKernel(stream, [&]() { cubBlockMergeSortKernel_b<block_size, ipt_const.value><<<1, block_size, 0, stream>>>(d_scores_in.get(), d_scores_out.get(), n, k); }, warm_up_runs, total_runs);
        };
        if (items_per_thread <= 1)
          launch(std::integral_constant<int, 1>());
        else if (items_per_thread <= 2)
          launch(std::integral_constant<int, 2>());
        else if (items_per_thread <= 4)
          launch(std::integral_constant<int, 4>());
        else if (items_per_thread <= 8)
          launch(std::integral_constant<int, 8>());
        else if (items_per_thread <= 16)
          launch(std::integral_constant<int, 16>());
        else
          launch(std::integral_constant<int, 32>());
      }

      // --- 4. CUB Block Radix Sort ---
      if (n >= 64) {
        const int items_per_thread = (n + block_size - 1) / block_size;
        auto launch = [&](auto ipt_const) {
          results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_RADIX)][i] =
              TimeKernel(stream, [&]() { cubBlockRadixSortKernel_b<block_size, ipt_const.value><<<1, block_size, 0, stream>>>(d_scores_in.get(), d_scores_out.get(), n, k); }, warm_up_runs, total_runs);
        };
        if (items_per_thread <= 1)
          launch(std::integral_constant<int, 1>());
        else if (items_per_thread <= 2)
          launch(std::integral_constant<int, 2>());
        else if (items_per_thread <= 4)
          launch(std::integral_constant<int, 4>());
        else if (items_per_thread <= 8)
          launch(std::integral_constant<int, 8>());
        else if (items_per_thread <= 16)
          launch(std::integral_constant<int, 16>());
        else
          launch(std::integral_constant<int, 32>());
      }
    } catch (const std::exception& e) {
      printf("[DEBUG] Exception caught while benchmarking N = %d. Error: %s\n", n, e.what());
      fflush(stdout);
      // Re-throw to allow the test framework to catch it
      throw;
    }
  }

  printf("[DEBUG] Benchmark finished successfully.\n");
  fflush(stdout);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "--- Internal Sort Micro-Benchmark Results (us) ---\n";
  std::cout << std::setw(8) << "N";
  for (int i = 0; i < (int)SortAlgo::COUNT; ++i) std::cout << std::setw(18) << SortAlgoToString((SortAlgo)i);
  std::cout << "\n-------------------------------------------------------------------------------------------------------------------\n";
  for (size_t i = 0; i < results.sort_sizes.size(); ++i) {
    std::cout << std::setw(8) << results.sort_sizes[i];
    for (int j = 0; j < (int)SortAlgo::COUNT; ++j) {
      std::cout << std::setw(18);
      if (results.latencies[j][i] < 0.0f)
        std::cout << "N/A";
      else
        std::cout << results.latencies[j][i] * 1000.0f;
    }
    std::cout << "\n";
  }
  std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

  // Pre-compute the best algorithm for each sort size.
  results.ComputeBestAlgos();
}
#else
void RunAndCacheSortBenchmark(int /*device_id*/, cudaStream_t /*stream*/, SortBenchmarkResults& results) {
  /*
    Store the following benchmark result from an NVIDIA RTX 4090 GPU to cache.
    These results are a fallback for when the online benchmark is disabled.
    CUDA Version: 12.8, Driver Version: 580.88

      N       Warp Bitonic      CUB Warp Merge    CUB Block Merge   CUB Block Radix
-------------------------------------------------------------------------------------------------------------------
      32              5.338 *          5.622            N/A               N/A
      64               N/A             6.287*           6.545              8.164
     128               N/A             7.848            6.569 *            8.039
     256               N/A             7.141            6.953 *            7.234
     512               N/A               N/A            6.971 *            7.196
    1024               N/A               N/A            7.537 *            7.789
    2048               N/A               N/A            10.112 *           10.254
    4096               N/A               N/A            15.002             14.561 *
    8192               N/A               N/A            77.912             20.857 *
-------------------------------------------------------------------------------------------------------------------

  */
  results.sort_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  results.latencies[static_cast<int>(SortAlgo::WARP_BITONIC)] = {4.636f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
  results.latencies[static_cast<int>(SortAlgo::CUB_WARP_MERGE)] = {5.728f, 5.071f, 6.338f, 7.419f, -1.f, -1.f, -1.f, -1.f, -1.f};
  results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_MERGE)] = {-1.f, 6.474f, 6.285f, 6.505f, 7.235f, 7.574f, 11.388f, 24.312f, 77.912f};
  results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_RADIX)] = {-1.f, 6.939f, 7.100f, 6.491f, 6.536f, 9.695f, 10.900f, 13.896f, 20.857f};

  // Pre-compute the best algorithm for each sort size.
  results.ComputeBestAlgos();
}
#endif

class SortBenchmarkCacheManager {
 public:
  // Runs the benchmark if it hasn't been run yet for this device, then returns the results.
  const SortBenchmarkResults& GetOrRun(int device_id, cudaStream_t stream) {
    std::call_once(once_flag_, [&] {
      results_ = std::make_unique<SortBenchmarkResults>();
      RunAndCacheSortBenchmark(device_id, stream, *results_);
      is_initialized_ = true;
    });
    return *results_;
  }

  // Returns benchmark results without a stream. Assumes GetOrRun has been called previously.
  const SortBenchmarkResults& Get() const {
    if (!is_initialized_) {
      throw std::runtime_error("Sort benchmark cache has not been initialized. Call GetOrRunSortBenchmark first.");
    }
    return *results_;
  }

 private:
  // We only have one set of benchmark results even though multiple devices may exist,
  // because the sorting algorithms are not device-specific, and the benchmark result can be
  // reused across devices of the same architecture.
  std::unique_ptr<SortBenchmarkResults> results_;
  std::once_flag once_flag_;
  std::atomic<bool> is_initialized_{false};
};

// Singleton instance provider
SortBenchmarkCacheManager& GetSortCache() {
  static SortBenchmarkCacheManager g_sort_benchmark_cache;
  return g_sort_benchmark_cache;
}
}  // namespace

// Public-facing functions to access the global sort benchmark cache.
inline const SortBenchmarkResults& GetOrRunSortBenchmark(cudaStream_t stream) {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return GetSortCache().GetOrRun(device_id, stream);
}


/**
 * @brief Gets previously cached benchmark results for the current device.
 * @return A const reference to the benchmark results.
 * @throws std::runtime_error if the benchmark has not yet been run for the current device.
 */
inline const SortBenchmarkResults& GetSortBenchmarkResults() {
  return GetSortCache().Get();
}

}  // namespace cuda
}  // namespace Generators
