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

// A struct to hold the results of offline sort benchmark.
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

// --- Singleton Cache Manager ---
namespace {  // Anonymous namespace for internal benchmark kernels

void CacheSortBenchmark(SortBenchmarkResults& results) {
  /*
    Store the following benchmark result from NVIDIA H200 GPU to cache.
    CUDA Version: 12.8, Driver Version:  570.133.20
-------------------------------------------------------------------------------------------------------------------
       N       K   Warp Bitonic Sort      CUB Warp Merge     CUB Block Merge     CUB Block Radix
-------------------------------------------------------------------------------------------------------------------
      32      32               2.313               2.830              -1.000              -1.000
      64      64              -1.000               3.409               4.077               5.445
     128      64              -1.000               4.177               4.098 rf              5.430
     256      64              -1.000               5.718               4.183               5.452
     512      64              -1.000              -1.000               4.937               5.697
    1024      64              -1.000              -1.000               6.363               6.649
    2048      64              -1.000              -1.000               9.606               9.391
    4096      64              -1.000              -1.000              19.547              14.504
    8192      64              -1.000              -1.000              48.835              30.836
-------------------------------------------------------------------------------------------------------------------
  */
  results.sort_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  results.latencies[static_cast<int>(SortAlgo::WARP_BITONIC)] = {2.313f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
  results.latencies[static_cast<int>(SortAlgo::CUB_WARP_MERGE)] = {2.830f, 3.409f, 4.177f, 5.718f, -1.f, -1.f, -1.f, -1.f, -1.f};
  results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_MERGE)] = {-1.f, 4.077f, 4.098f, 4.183f, 4.937f, 6.363f, 9.606f, 19.547f, 48.835f};
  results.latencies[static_cast<int>(SortAlgo::CUB_BLOCK_RADIX)] = {-1.f, 5.445f, 5.430f, 5.452f, 5.697f, 6.649f, 9.391f, 14.504f, 30.836f};

  // Pre-compute the best algorithm for each sort size.
  results.ComputeBestAlgos();
}

class SortBenchmarkCacheManager {
 public:
  SortBenchmarkCacheManager() {
    results_ = std::make_unique<SortBenchmarkResults>();
    CacheSortBenchmark(*results_);
  }

  const SortBenchmarkResults& Get() const {
    return *results_;
  }

 private:
  // We only have one set of benchmark results even though multiple devices may exist,
  // because the sorting algorithms are not device-specific, and the benchmark result can be
  // reused across devices of the same architecture.
  std::unique_ptr<SortBenchmarkResults> results_;
};

// Singleton instance provider
SortBenchmarkCacheManager& GetSortCache() {
  static SortBenchmarkCacheManager g_sort_benchmark_cache;
  return g_sort_benchmark_cache;
}
}  // namespace

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
