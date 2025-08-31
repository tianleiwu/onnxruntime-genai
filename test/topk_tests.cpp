// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <limits>
#include <gtest/gtest.h>

#include "cuda_runtime.h"
#include "smartptrs.h"
#include "../src/cuda/cuda_topk.h"

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

// A struct to hold the parameters for a benchmark configuration
struct BenchmarkParams {
  int batch_size;
  int vocab_size;
  int k;
};

// A struct to hold the results of a single benchmark run
struct BenchmarkResult {
  BenchmarkParams params;
  std::string algo_name;
  int num_partitions;
  int block_size;
  float latency_ms;
};

// Global mutex to serialize benchmark tests
static std::mutex benchmark_mutex;

// Function to compare results between a test algorithm and a reference
bool CompareResults(const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::string& algo_name) {
  bool match = true;
  const float epsilon = 1e-5f;

  for (size_t i = 0; i < reference_scores.size(); ++i) {
    if (reference_indices[i] != actual_indices[i] || std::abs(reference_scores[i] - actual_scores[i]) > epsilon) {
      std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch at position " << i
                << ". Expected: (" << reference_indices[i] << ", " << std::fixed << std::setprecision(6) << reference_scores[i]
                << "), Got: (" << actual_indices[i] << ", " << actual_scores[i] << ")" << std::endl;
      match = false;
      break;
    }
  }
  return match;
}

// Function to run parity tests for all algorithms against a reference implementation
void RunParityTests(const BenchmarkParams& params, float temperature) {
  std::cout << "\n--- Running Parity Tests with batch_size="
            << params.batch_size << ", vocab_size=" << params.vocab_size << ", k=" << params.k << ", temperature=" << temperature
            << "---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto sampling_data = std::make_unique<Generators::cuda::SamplingData>(1234, params.batch_size, params.vocab_size, stream);
  auto scores_in_d = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);
  auto scores_in_d_copy = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);

  // Use a fixed seed for reproducibility
  std::mt19937 gen(3407);
  std::uniform_real_distribution<float> dis(0.0f, 100.0f);
  std::vector<float> scores_in_h(params.batch_size * params.vocab_size);
  for (auto& val : scores_in_h) {
    val = dis(gen);
  }
  CUDA_CHECK(cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(scores_in_d_copy.get(), scores_in_d.get(), scores_in_h.size() * sizeof(float), cudaMemcpyDeviceToDevice));

  // --- Get Reference Result using Full Sort ---
  auto ref_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
  auto ref_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
  Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in_d.get(), ref_scores_d.get(), ref_indices_d.get(), params.vocab_size, params.batch_size, params.k, temperature);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> ref_scores_h(params.batch_size * params.k);
  std::vector<int> ref_indices_h(params.batch_size * params.k);
  CUDA_CHECK(cudaMemcpy(ref_scores_h.data(), ref_scores_d.get(), ref_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_indices_h.data(), ref_indices_d.get(), ref_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

  // --- Test Other Algorithms ---
  auto test_algo = [&](const std::string& name, auto func) {
    auto actual_scores_d = Generators::CudaMallocArray<float>(params.batch_size * params.k);
    auto actual_indices_d = Generators::CudaMallocArray<int>(params.batch_size * params.k);
    func(actual_scores_d.get(), actual_indices_d.get());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> actual_scores_h(params.batch_size * params.k);
    std::vector<int> actual_indices_h(params.batch_size * params.k);
    CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), actual_scores_d.get(), actual_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), actual_indices_d.get(), actual_indices_h.size() * sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_TRUE(CompareResults(ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, name));
    std::cout << "  [PASS] " << name << std::endl;
  };

  if (params.k <= 64) {
    test_algo("SELECTION_SORT", [&](float* s_d, int* i_d) {
      // Selection Sort kernel will change scores_in inplace so we use a copy here.
      Generators::cuda::RunTopKViaSelectionSort(sampling_data.get(), stream, scores_in_d_copy.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature);
    });

    test_algo("SELECTION_SORT", [&](float* s_d, int* i_d) {
      // Selection Sort kernel will change scores_in inplace so we use a copy here.
      Generators::cuda::baseline::RunTopKViaBaseline(sampling_data.get(), stream, scores_in_d_copy.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature);
    });


    for (int sort_size : {256, 512, 1024, 2048, 4096}) {
      for (int num_partitions : {16, 32, 64, 128, 256, 512, 1024}) {
        assert(num_partitions <= Generators::cuda::kBitonicSortMaxPartitions);
        if (params.vocab_size <= sort_size * num_partitions && params.vocab_size >= sort_size * num_partitions / 2) {
          // std::string algo_name_v0 = "BITONIC_V0 (s=" + std::to_string(sort_size) + ",p=" + std::to_string(num_partitions) + ")";
          // test_algo(algo_name_v0, [&](float* s_d, int* i_d) {
          //   Generators::cuda::v0::RunTopKViaMapReduceBitonicSort_v0(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, num_partitions, sort_size);
          // });

          std::string algo_name = "BITONIC (s=" + std::to_string(sort_size) + ",p=" + std::to_string(num_partitions) + ")";
          test_algo(algo_name, [&](float* s_d, int* i_d) {
            Generators::cuda::RunTopKViaMapReduceBitonicSort(sampling_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size, params.batch_size, params.k, temperature, num_partitions, sort_size);
          });
        }
      }
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// Function to print a summary of benchmark results in CSV format
void PrintSummary(const std::vector<BenchmarkResult>& all_results) {
  // This struct holds the aggregated data for a unique key (batch_size, vocab_size, k)
  struct SummaryData {
    std::string best_algo_full_name;
    float min_latency = std::numeric_limits<float>::max();
    float latency_full_sort = -1.0f;
    float latency_selection_sort = -1.0f;
    float latency_bitonic_sort = -1.0f;
  };

  // Use a map to group results by their parameters
  std::map<std::tuple<int, int, int>, SummaryData> summary_map;

  // Iterate over all detailed results to populate the summary map
  for (const auto& result : all_results) {
    auto key = std::make_tuple(result.params.batch_size, result.params.vocab_size, result.params.k);

    // Create a more descriptive name for map-reduce algorithms
    std::string full_algo_name = result.algo_name;

    // Check if this is the best performing algorithm for this key so far
    if (result.latency_ms < summary_map[key].min_latency) {
      summary_map[key].min_latency = result.latency_ms;
      summary_map[key].best_algo_full_name = full_algo_name;
    }

    // Store the latencies for the specific baseline algorithms
    if (result.algo_name == "FULL_SORT") {
      summary_map[key].latency_full_sort = result.latency_ms;
    } else if (result.algo_name == "SELECTION_SORT") {
      summary_map[key].latency_selection_sort = result.latency_ms;
    } else {
      summary_map[key].latency_bitonic_sort = summary_map[key].latency_bitonic_sort > 0 ? std::min(result.latency_ms, summary_map[key].latency_bitonic_sort) : result.latency_ms;
    }
  }

  // Print the CSV header
  std::cout << "\n--- Benchmark Summary (CSV) ---\n";
  std::cout << "batch_size,vocab_size,k,full_sort,selection_sort,bitonic_sort,full/selection,full/bitonic,selection/bitonic,best\n";

  // Print each row of the summary table
  for (const auto& pair : summary_map) {
    const auto& key = pair.first;
    const auto& data = pair.second;

    // Calculate performance ratios against the baselines
    float ratio_0 = (data.latency_full_sort > 0 && data.latency_selection_sort > 0) ? data.latency_full_sort / data.latency_selection_sort : 0.0f;
    float ratio_1 = (data.latency_bitonic_sort > 0 && data.latency_full_sort > 0) ? data.latency_full_sort / data.latency_bitonic_sort : 0.0f;
    float ratio_2 = (data.latency_bitonic_sort > 0 && data.latency_selection_sort > 0) ? data.latency_selection_sort / data.latency_bitonic_sort : 0.0f;

    std::cout << std::get<0>(key) << ","
              << std::get<1>(key) << ","
              << std::get<2>(key) << ","
              << std::fixed << std::setprecision(4)
              << (data.latency_full_sort > 0 ? std::to_string(data.latency_full_sort) : "N/A") << ","
              << (data.latency_selection_sort > 0 ? std::to_string(data.latency_selection_sort) : "N/A") << ","
              << (data.latency_bitonic_sort > 0 ? std::to_string(data.latency_bitonic_sort) : "N/A") << ","
              << std::fixed << std::setprecision(2) << ratio_0 << "," << ratio_1 << "," << ratio_2 << ","
              << "\"" << data.best_algo_full_name << "\"" << "\n";
  }
}

// Main benchmark function
void RunBenchmarks() {
  // --- Define Benchmark Configurations ---
  std::vector<int> batch_sizes = {1};
  std::vector<int> vocab_sizes = {201088, 32000};  // GPT-OSS 201088, LLAMA2 32000, LLAMA3 128256, DeepSeek 102400, QWen3 1516465
  std::vector<int> ks = {50, 1, 8, 16, 32, 64};

  // Enable this to to find heuristics for kernel selection.
  // It lists vocabulary sizes within the range of 16K to 512K (32 data points using step of 16K).
  constexpr bool comprehensive = false;
  if constexpr (comprehensive) {
    vocab_sizes.clear();
    for (int v = 16 * 1024; v <= 512 * 1024; v += 16 * 1024) {
      vocab_sizes.push_back(v);
    }
     for (int b = 2; b <= 8; b *= 2) {
      batch_sizes.push_back(b);
    }
  }

  std::vector<BenchmarkParams> configs;
  for (int batch_size : batch_sizes) {
    for (int vocab_size : vocab_sizes) {
      for (int k : ks) {
        configs.push_back({batch_size, vocab_size, k});
      }
    }
  }

  constexpr int warmup_runs = 5;
  constexpr int timing_runs = comprehensive ? 1000 : 10000;
  constexpr float temperature = 0.9f;

  std::vector<BenchmarkResult> all_results;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (const auto& params : configs) {
    std::cout << "\nRunning benchmark for: batch_size=" << params.batch_size
              << ", vocab_size=" << params.vocab_size
              << ", k=" << params.k << "..." << std::endl;

    // --- Setup ---
    unsigned long long random_seed = 1234;
    auto sampling_data = std::make_unique<Generators::cuda::SamplingData>(random_seed, params.batch_size, params.vocab_size, stream);

    auto scores_in = Generators::CudaMallocArray<float>(params.batch_size * params.vocab_size);
    auto scores_out = Generators::CudaMallocArray<float>(params.batch_size * params.k);
    auto indices_out = Generators::CudaMallocArray<int>(params.batch_size * params.k);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int total_size = params.batch_size * params.vocab_size;

    auto measure_latency = [&](const std::string& name, int num_partitions, int block_size, auto func) {
      // Warmup
      for (int i = 0; i < warmup_runs; ++i) {
        // Regenerate data for each warmup run as well to ensure caches are not misleading
        Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);
        func();
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Timing
      float total_ms = 0.0f;
      for (int i = 0; i < timing_runs; ++i) {
        // Regenerate random data before each timed run to bust caches
        Generators::cuda::RandomTopkInput(stream, scores_in.get(), sampling_data->curand_states.get(), total_size, params.batch_size);

        CUDA_CHECK(cudaEventRecord(start, stream));
        func();
        CUDA_CHECK(cudaEventRecord(stop, stream));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
      }
      all_results.push_back({params, name, num_partitions, block_size, total_ms / timing_runs});
    };

    // --- Run Benchmarks for each algorithm ---

    if (params.k <= 64) {
      measure_latency("SELECTION_SORT", 0, 256, [&]() {
        Generators::cuda::RunTopKViaSelectionSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
      });

      measure_latency("BASELINE", 0, 256, [&]() {
        Generators::cuda::baseline::RunTopKViaBaseline(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
      });

      // This supports from vocabulary size in the range of (256 * 32 / 2, 4096 * 256], that is from 4K (exclusive) to 1M (inclusive).
      for (int sort_size : {256, 512, 1024, 2048, 4096}) {
        for (int num_partitions : {16, 32, 64, 128, 256, 512, 1024}) {
          assert(num_partitions <= Generators::cuda::kBitonicSortMaxPartitions);
          if (params.vocab_size <= sort_size * num_partitions && params.vocab_size >= sort_size * num_partitions / 2) {
            std::string algo_name = "BITONIC (s=" + std::to_string(sort_size) + ",p=" + std::to_string(num_partitions) + ")";
            measure_latency(algo_name, num_partitions, 256, [&, sort_size, num_partitions]() {
              Generators::cuda::RunTopKViaMapReduceBitonicSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions, sort_size);
            });

            // std::string algo_name_v0 = "BITONIC_V0 (s=" + std::to_string(sort_size) + ",p=" + std::to_string(num_partitions) + ")";
            // measure_latency(algo_name_v0, num_partitions, 256, [&, sort_size, num_partitions]() {
            //   Generators::cuda::v0::RunTopKViaMapReduceBitonicSort_v0(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature, num_partitions, sort_size);
            // });
          }
        }
      }
    }

    measure_latency("FULL_SORT", 0, 256, [&]() {
      Generators::cuda::RunTopKViaFullSort(sampling_data.get(), stream, scores_in.get(), scores_out.get(), indices_out.get(), params.vocab_size, params.batch_size, params.k, temperature);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  // --- Print Detailed Results ---
  std::cout << "\n--- Benchmark Results ---\n";
  std::cout << std::left << std::setw(12) << "Batch Size"
            << std::setw(12) << "Vocab Size"
            << std::setw(5) << "K"
            << std::setw(28) << "Algorithm"
            << "Latency (ms)\n";
  std::cout << std::string(97, '-') << "\n";

  for (const auto& result : all_results) {
    std::cout << std::left << std::setw(12) << result.params.batch_size
              << std::setw(12) << result.params.vocab_size
              << std::setw(5) << result.params.k
              << std::setw(28) << result.algo_name
              << std::fixed << std::setprecision(4) << result.latency_ms << "\n";
  }

  // --- Print CSV Summary ---
  PrintSummary(all_results);
}

TEST(TopKTests, ParityTests) {
  std::lock_guard<std::mutex> lock(benchmark_mutex);
  std::vector<BenchmarkParams> test_cases = {
      {1, 10000, 50}, {2, 10000, 64}, {1, 32000, 1}, {1, 32000, 50}, {1, 512000, 50}  // Test case for bitonic sort
  };

  for (const auto& params : test_cases) {
    RunParityTests(params, 1.0f);
    RunParityTests(params, 0.7f);
  }
}

TEST(TopKTests, BenchmarkTests) {
  std::lock_guard<std::mutex> lock(benchmark_mutex);
  RunBenchmarks();
}
