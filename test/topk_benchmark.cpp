// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if USE_CUDA
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../src/cuda/cuda_topk.h"
#include "statistics_helper.h"

namespace {

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
  int partition_size;

  float latency_ms;
  float latency_ms_stdev;
  float latency_ms_95_percentile;
};

// A struct to hold the aggregated results for the final CSV summary.
struct CsvSummaryResult {
  BenchmarkParams params;

  // Latency for each algorithm. A negative value indicates it was not run.
  float full_sort_latency = -1.0f;
  float radix_sort_latency = -1.0f;
  float selection_sort_latency = -1.0f;
  float hybrid_sort_1024_latency = -1.0f;
  float hybrid_sort_2048_latency = -1.0f;
  float hybrid_sort_4096_latency = -1.0f;
  float hybrid_sort_8192_latency = -1.0f;

  std::string best_algorithm = "NA";
  float best_latency = std::numeric_limits<float>::max();

  // Fields for hybrid sort partition size accuracy
  int estimated_partition_size = 0;
  float latency_of_estimated_partition_size = -1.0f;
  int best_partition_size = 0;
  float latency_of_best_partition_size = -1.0f;
  float diff_ratio = 0.0f;
};

void PrintSummary(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n--- TopK Cuda Kernel Benchmark Summary ---\n";
  std::cout << std::left << std::setw(12) << "Batch Size" << std::setw(12) << "Vocab Size" << std::setw(5) << "K"
            << std::setw(28) << "Algorithm" << std::setw(12) << "Latency(us)" << std::setw(12) << "Stdev(us)"
            << std::setw(12) << "P95(us)" << "\n";
  std::cout << std::string(97, '-') << "\n";

  for (const auto& result : results) {
    std::string full_algo_name = result.algo_name;
    if (result.partition_size > 0) {
      full_algo_name += " (p=" + std::to_string(result.partition_size) + ")";
    }

    std::cout << std::left << std::setw(12) << result.params.batch_size << std::setw(12) << result.params.vocab_size
              << std::setw(5) << result.params.k << std::setw(28) << full_algo_name << std::fixed
              << std::setprecision(2) << std::setw(12) << result.latency_ms * 1000.0f << std::setw(12)
              << result.latency_ms_stdev * 1000.0f << std::setw(12) << result.latency_ms_95_percentile * 1000.0f
              << "\n";
  }
}

void PrintCsvSummary(const std::vector<CsvSummaryResult>& results) {
  if (results.empty()) {
    return;
  }
  std::cout << "\n--- TopK Benchmark CSV Summary ---\n";
  std::cout << "batch_size,vocab_size,k,full_sort,radix_sort,selection_sort,hybrid_sort_1024,hybrid_sort_2048,hybrid_sort_4096,hybrid_sort_8192,best_algorithm,best_latency,estimated_partition_size,latency_of_estimated_partition_size,best_partition_size,latency_of_best_partition_size,diff_ratio\n";

  for (const auto& result : results) {
    std::cout << result.params.batch_size << ","
              << result.params.vocab_size << ","
              << result.params.k << ",";

    auto print_latency = [](float latency) {
      if (latency < 0.0f) {
        std::cout << "NA";
      } else {
        std::cout << std::fixed << std::setprecision(4) << latency;
      }
    };

    print_latency(result.full_sort_latency);
    std::cout << ",";
    print_latency(result.radix_sort_latency);
    std::cout << ",";
    print_latency(result.selection_sort_latency);
    std::cout << ",";
    print_latency(result.hybrid_sort_1024_latency);
    std::cout << ",";
    print_latency(result.hybrid_sort_2048_latency);
    std::cout << ",";
    print_latency(result.hybrid_sort_4096_latency);
    std::cout << ",";
    print_latency(result.hybrid_sort_8192_latency);
    std::cout << ",";

    std::cout << result.best_algorithm << ",";

    if (result.best_latency == std::numeric_limits<float>::max()) {
      std::cout << "NA";
    } else {
      std::cout << std::fixed << std::setprecision(4) << result.best_latency;
    }

    std::cout << "," << result.estimated_partition_size << ",";
    print_latency(result.latency_of_estimated_partition_size);
    std::cout << "," << result.best_partition_size << ",";
    print_latency(result.latency_of_best_partition_size);
    std::cout << "," << std::fixed << std::setprecision(4) << result.diff_ratio << "\n";
  }
}

void RunBenchmarks(const BenchmarkParams& params, std::vector<CsvSummaryResult>& csv_results) {
  std::cout << "\n--- Running Benchmarks with batch_size=" << params.batch_size << ", vocab_size=" << params.vocab_size
            << ", k=" << params.k << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto scores_in_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.vocab_size);

  auto bench_algo = [&](auto func) {
    const int warm_up_runs = 5;
    const int total_runs = 1000;
    std::vector<double> latencies;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warm_up_runs + total_runs; ++i) {
      CUDA_CHECK(cudaEventRecord(start, stream));
      func();
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      if (i >= warm_up_runs) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        latencies.push_back(ms);
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return std::make_tuple(static_cast<float>(mean(latencies)), static_cast<float>(stdev(latencies)),
                           static_cast<float>(percentile(latencies, 95.0)));
  };

  std::vector<BenchmarkResult> all_results;
  CsvSummaryResult current_csv_result;
  current_csv_result.params = params;
  std::map<std::string, float> algo_latencies;

  auto data = std::make_unique<Generators::cuda::TopkData>(params.batch_size, params.vocab_size, stream);
  // Benchmark Full Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::RunTopKViaFullSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                           params.batch_size, params.k);
    });
    all_results.push_back({params, "FULL_SORT", 0, mean_ms, stdev_ms, p95_ms});
    current_csv_result.full_sort_latency = mean_ms;
    algo_latencies["FULL_SORT"] = mean_ms;
  }

  // Benchmark Selection Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::RunTopKViaSelectionSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                                params.batch_size, params.k);
    });
    all_results.push_back({params, "SELECTION_SORT", 0, mean_ms, stdev_ms, p95_ms});
    current_csv_result.selection_sort_latency = mean_ms;
    algo_latencies["SELECTION_SORT"] = mean_ms;
  }

  // Benchmark Radix Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::RunTopKViaRadixSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                            params.batch_size, params.k);
    });
    all_results.push_back({params, "RADIX_SORT", 0, mean_ms, stdev_ms, p95_ms});
    current_csv_result.radix_sort_latency = mean_ms;
    algo_latencies["RADIX_SORT"] = mean_ms;
  }

  if (params.k <= Generators::cuda::kHybridSortMaxK) {
    // Benchmark Hybrid Sort
    current_csv_result.estimated_partition_size = data->hybrid_sort_partition_size;

    float best_hybrid_latency_ms = std::numeric_limits<float>::max();
    int best_hybrid_partition_size = 0;

    for (int partition_size : {1024, 2048, 4096, 8192}) {
      if (partition_size > 1024 && partition_size > params.vocab_size * 2) {
        continue;
      }

      data->hybrid_sort_partition_size = partition_size;
      auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
        Generators::cuda::RunTopKViaHybridSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                               params.batch_size, params.k);
      });

      if (mean_ms < best_hybrid_latency_ms) {
        best_hybrid_latency_ms = mean_ms;
        best_hybrid_partition_size = partition_size;
      }
      if (partition_size == current_csv_result.estimated_partition_size) {
        current_csv_result.latency_of_estimated_partition_size = mean_ms;
      }

      std::string algo_name = "HYBRID_SORT_" + std::to_string(partition_size);
      algo_latencies[algo_name] = mean_ms;

      if (partition_size == 1024) current_csv_result.hybrid_sort_1024_latency = mean_ms;
      if (partition_size == 2048) current_csv_result.hybrid_sort_2048_latency = mean_ms;
      if (partition_size == 4096) current_csv_result.hybrid_sort_4096_latency = mean_ms;
      if (partition_size == 8192) current_csv_result.hybrid_sort_8192_latency = mean_ms;

      all_results.push_back({params, "HYBRID_SORT", partition_size, mean_ms, stdev_ms, p95_ms});
    }

    if (best_hybrid_partition_size > 0 && current_csv_result.latency_of_estimated_partition_size >= 0.0f) {
      current_csv_result.best_partition_size = best_hybrid_partition_size;
      current_csv_result.latency_of_best_partition_size = best_hybrid_latency_ms;
      current_csv_result.diff_ratio = (current_csv_result.latency_of_estimated_partition_size - best_hybrid_latency_ms) / best_hybrid_latency_ms;
    }
  }

  // Find the best algorithm overall for this configuration
  for (const auto& pair : algo_latencies) {
    if (pair.second < current_csv_result.best_latency) {
      current_csv_result.best_latency = pair.second;
      current_csv_result.best_algorithm = pair.first;
    }
  }

  // If the best algorithm is a hybrid sort, just call it "HYBRID"
  if (current_csv_result.best_algorithm.rfind("HYBRID_SORT", 0) == 0) {
    current_csv_result.best_algorithm = "HYBRID";
  }

  csv_results.push_back(current_csv_result);

  PrintSummary(all_results);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace

TEST(TopKBenchmarks, PerformanceTests) {
  std::vector<CsvSummaryResult> csv_summary_results;
  
  // Test different K
  {
    std::vector<int> batch_sizes = {1};
    std::vector<int> vocab_sizes = {201088};
    std::vector<int> ks = {1, 2, 4, 8, 10, 16, 32, 50, Generators::cuda::kHybridSortMaxK};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }
    for (const auto& params : test_cases) {
      RunBenchmarks(params, csv_summary_results);
    }
  }

  constexpr bool is_build_pipeline = false;  // Change it false to trigger more runs in local machine.

  // Test small vocab_sizes.
  // if constexpr (!is_build_pipeline) {
  //   std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
  //   std::vector<int> vocab_sizes = {256};
  //   std::vector<int> ks = {50};

  //   std::vector<BenchmarkParams> test_cases;
  //   for (int batch_size : batch_sizes) {
  //     for (int vocab_size : vocab_sizes) {
  //       for (int k : ks) {
  //         test_cases.push_back({batch_size, vocab_size, k});
  //       }
  //     }
  //   }
  //   for (const auto& params : test_cases) {
  //     RunBenchmarks(params, csv_summary_results);
  //   }
  // }

  // Test medium vocab_sizes.
  // if constexpr (!is_build_pipeline) {
  //   std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
  //   std::vector<int> vocab_sizes = {32000};
  //   std::vector<int> ks = {50};

  //   std::vector<BenchmarkParams> test_cases;
  //   for (int batch_size : batch_sizes) {
  //     for (int vocab_size : vocab_sizes) {
  //       for (int k : ks) {
  //         test_cases.push_back({batch_size, vocab_size, k});
  //       }
  //     }
  //   }
  //   for (const auto& params : test_cases) {
  //     RunBenchmarks(params, csv_summary_results);
  //   }
  // }

  // Test batch sizes.
  // if constexpr (!is_build_pipeline) {
  //   std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
  //   std::vector<int> vocab_sizes = {201088};
  //   std::vector<int> ks = {50};

  //   std::vector<BenchmarkParams> test_cases;
  //   for (int batch_size : batch_sizes) {
  //     for (int vocab_size : vocab_sizes) {
  //       for (int k : ks) {
  //         test_cases.push_back({batch_size, vocab_size, k});
  //       }
  //     }
  //   }

  //   for (const auto& params : test_cases) {
  //     RunBenchmarks(params, csv_summary_results);
  //   }
  // }

  // Test vocab_sizes.
  // if constexpr (!is_build_pipeline) {
  //   // std::vector<int> batch_sizes = {1};
  //   // std::vector<int> vocab_sizes = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  //   // std::vector<int> ks = {1, 4, 8, 50};
  //   std::vector<int> batch_sizes = {4};
  //   std::vector<int> vocab_sizes = {16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  //   std::vector<int> ks = {50};

  //   std::vector<BenchmarkParams> test_cases;
  //   for (int batch_size : batch_sizes) {
  //     for (int vocab_size : vocab_sizes) {
  //       for (int k : ks) {
  //         test_cases.push_back({batch_size, vocab_size, k});
  //       }
  //     }
  //   }

  //   for (const auto& params : test_cases) {
  //     RunBenchmarks(params, csv_summary_results);
  //   }
  // }

  if constexpr (!is_build_pipeline) {
    std::vector<int> batch_sizes = {1, 2, 4, 8};
    std::vector<int> vocab_sizes = {512, 1024, 2048, 4096};
    for (int v = 8 * 1024; v < 64 * 1024; v += 8 * 1024) {
      vocab_sizes.push_back(v);
    }
    for (int v = 64 * 1024; v <= 256 * 1024; v += 16 * 1024) {
      vocab_sizes.push_back(v);
    }
    std::vector<int> ks = {1, 2, 4, 8, 10, 16, 32, 50, 64, 128};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }

    for (const auto& params : test_cases) {
      RunBenchmarks(params, csv_summary_results);
    }
  }

  PrintCsvSummary(csv_summary_results);
}

#endif
