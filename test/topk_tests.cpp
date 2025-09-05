// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

// A struct to hold the parameters for a test configuration
struct TopKTestParams {
  int batch_size;
  int vocab_size;
  int k;
};

// Function to compare final raw scores and indices
bool CompareResults(const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::string& algo_name, int batch_size, int k) {
  bool match = true;
  const float epsilon = 1e-4f;

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < k; ++i) {
      size_t idx = static_cast<size_t>(b) * k + i;
      if (reference_indices[idx] != actual_indices[idx] ||
          std::abs(reference_scores[idx] - actual_scores[idx]) > epsilon) {
        std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch in batch " << b << " at position " << i
                  << ". Expected: (" << reference_indices[idx] << ", " << std::fixed << std::setprecision(6)
                  << reference_scores[idx] << "), Got: (" << actual_indices[idx] << ", " << actual_scores[idx] << ")"
                  << std::endl;
        match = false;
        goto end_loops;  // Exit both loops
      }
    }
  }

end_loops:
  return match;
}

// Function to run parity tests for all algorithms against a reference implementation (Full Sort)
void RunParityTests(const TopKTestParams& params) {
  std::cout << "\n--- Running Parity Tests with batch_size=" << params.batch_size
            << ", vocab_size=" << params.vocab_size << ", k=" << params.k << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_vocab_size = static_cast<size_t>(params.batch_size) * params.vocab_size;
  size_t topk_size = static_cast<size_t>(params.batch_size) * params.k;

  auto scores_in_d = Generators::CudaMallocArray<float>(total_vocab_size);
  auto scores_in_d_copy = Generators::CudaMallocArray<float>(total_vocab_size);

  // Use a fixed seed for reproducibility
  std::mt19937 gen(3407);
  std::uniform_real_distribution<float> dis(0.0f, 100.0f);
  std::vector<float> scores_in_h(total_vocab_size);
  for (auto& val : scores_in_h) {
    val = dis(gen);
  }
  CUDA_CHECK(
      cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(scores_in_d_copy.get(), scores_in_d.get(), scores_in_h.size() * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  // --- Get Reference Result using Full Sort ---
  // The refactored Top-K functions now output raw scores directly.
  auto ref_scores_d = Generators::CudaMallocArray<float>(topk_size);
  auto ref_indices_d = Generators::CudaMallocArray<int>(topk_size);

  auto topk_data = std::make_unique<Generators::cuda::TopkData>(params.batch_size, params.vocab_size, stream);
  Generators::cuda::RunTopKViaFullSort(topk_data.get(), stream, scores_in_d.get(), ref_scores_d.get(),
                                       ref_indices_d.get(), params.vocab_size, params.batch_size, params.k);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> ref_scores_h(topk_size);
  std::vector<int> ref_indices_h(topk_size);
  CUDA_CHECK(
      cudaMemcpy(ref_scores_h.data(), ref_scores_d.get(), ref_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_indices_h.data(), ref_indices_d.get(), ref_indices_h.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // --- Test Other Algorithms ---
  auto test_algo = [&](const std::string& name, auto func) {
    auto actual_scores_d = Generators::CudaMallocArray<float>(topk_size);
    auto actual_indices_d = Generators::CudaMallocArray<int>(topk_size);
    func(actual_scores_d.get(), actual_indices_d.get());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> actual_scores_h(topk_size);
    std::vector<int> actual_indices_h(topk_size);
    CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), actual_scores_d.get(), actual_scores_h.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), actual_indices_d.get(), actual_indices_h.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    ASSERT_TRUE(CompareResults(ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, name, params.batch_size,
                               params.k));
    std::cout << "  [PASS] " << name << " (Raw Scores & Indices)" << std::endl;
  };

  if (params.k <= 64) {
    test_algo("SELECTION_SORT", [&](float* s_d, int* i_d) {
      // Selection sort modifies the input in place, so we use a copy.
      CUDA_CHECK(cudaMemcpy(scores_in_d_copy.get(), scores_in_d.get(), scores_in_h.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      Generators::cuda::RunTopKViaSelectionSort(topk_data.get(), stream, scores_in_d_copy.get(), s_d, i_d,
                                                params.vocab_size, params.batch_size, params.k);
    });

    for (int partition_size : {1024, 2048, 4096, 8192}) {
      if (partition_size > params.vocab_size) {
        continue;
      }
      std::string algo_name = "HYBRID (" + std::to_string(partition_size) + ")";
      test_algo(algo_name, [&](float* s_d, int* i_d) {
        Generators::cuda::RunTopKViaHybridSort(topk_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size,
                                               params.batch_size, params.k, partition_size);
      });
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(TopKTests, ParityTests) {
  std::vector<TopKTestParams> test_cases = {{1, 10000, 50}, {2, 10000, 64},  {1, 32000, 1},
                                            {1, 32000, 16}, {1, 512000, 50}, {1, 1024, 18}};

  for (const auto& params : test_cases) {
    RunParityTests(params);
  }
}
