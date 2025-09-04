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

// A struct to hold the parameters for a benchmark configuration
struct TopKTestParams {
  int batch_size;
  int vocab_size;
  int k;
  float temperature;
};

// Function to compare final probabilities and indices
bool CompareResults(const std::vector<float>& reference_scores, const std::vector<int>& reference_indices,
                    const std::vector<float>& actual_scores, const std::vector<int>& actual_indices,
                    const std::string& algo_name) {
  bool match = true;
  const float epsilon = 1e-5f;

  for (size_t i = 0; i < reference_scores.size(); ++i) {
    if (reference_indices[i] != actual_indices[i] || std::abs(reference_scores[i] - actual_scores[i]) > epsilon) {
      std::cerr << "Parity Test Failed for " << algo_name << ": Mismatch at position " << i << ". Expected: ("
                << reference_indices[i] << ", " << std::fixed << std::setprecision(6) << reference_scores[i]
                << "), Got: (" << actual_indices[i] << ", " << actual_scores[i] << ")" << std::endl;
      match = false;
      break;
    }
  }
  return match;
}

// Function to compare raw scores before softmax
bool CompareRawScores(const std::vector<float>& reference_scores, const std::vector<float>& actual_scores,
                      const std::string& algo_name) {
  bool match = true;
  const float epsilon = 1e-6f;  // Use a smaller epsilon for raw scores

  for (size_t i = 0; i < reference_scores.size(); ++i) {
    if (std::abs(reference_scores[i] - actual_scores[i]) > epsilon) {
      std::cerr << "Raw Score Verification Failed for " << algo_name << ": Mismatch at position " << i
                << ". Expected: " << std::fixed << std::setprecision(6) << reference_scores[i]
                << ", Got: " << actual_scores[i] << std::endl;
      match = false;
      break;
    }
  }
  return match;
}

// Function to run parity tests for all algorithms against a reference
// implementation
void RunParityTests(const TopKTestParams& params) {
  std::cout << "\n--- Running Parity Tests with batch_size=" << params.batch_size
            << ", vocab_size=" << params.vocab_size << ", k=" << params.k << ", temperature=" << params.temperature
            << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto scores_in_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.vocab_size);
  auto scores_in_d_copy =
      Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.vocab_size);

  // Use a fixed seed for reproducibility
  std::mt19937 gen(3407);
  std::uniform_real_distribution<float> dis(0.0f, 100.0f);
  std::vector<float> scores_in_h(static_cast<size_t>(params.batch_size) * params.vocab_size);
  for (auto& val : scores_in_h) {
    val = dis(gen);
  }
  CUDA_CHECK(
      cudaMemcpy(scores_in_d.get(), scores_in_h.data(), scores_in_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(scores_in_d_copy.get(), scores_in_d.get(), scores_in_h.size() * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  // --- Get Reference Result using Full Sort ---
  auto ref_scores_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.k);
  auto ref_indices_d = Generators::CudaMallocArray<int>(static_cast<size_t>(params.batch_size) * params.k);

  float temperature = params.temperature;
  auto topk_data = std::make_unique<Generators::cuda::TopkData>(params.batch_size, params.vocab_size, stream);
  Generators::cuda::RunTopKViaFullSort(topk_data.get(), stream, scores_in_d.get(), ref_scores_d.get(),
                                       ref_indices_d.get(), params.vocab_size, params.batch_size, params.k,
                                       temperature);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> ref_scores_h(static_cast<size_t>(params.batch_size) * params.k);
  std::vector<int> ref_indices_h(static_cast<size_t>(params.batch_size) * params.k);
  CUDA_CHECK(
      cudaMemcpy(ref_scores_h.data(), ref_scores_d.get(), ref_scores_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_indices_h.data(), ref_indices_d.get(), ref_indices_h.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // --- Get Reference Raw Scores from Full Sort ---
  std::vector<float> ref_raw_topk_scores_h(static_cast<size_t>(params.batch_size) * params.k);
  CUDA_CHECK(cudaMemcpy2D(ref_raw_topk_scores_h.data(),                            // dst
                          static_cast<size_t>(params.k) * sizeof(float),           // dpitch
                          topk_data->intermediate_scores_1.get(),                  // src
                          static_cast<size_t>(params.vocab_size) * sizeof(float),  // spitch
                          static_cast<size_t>(params.k) * sizeof(float),           // width
                          params.batch_size,                                       // height
                          cudaMemcpyDeviceToHost));

  // --- Test Other Algorithms ---
  auto test_algo = [&](const std::string& name, auto func) {
    auto actual_scores_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.k);
    auto actual_indices_d = Generators::CudaMallocArray<int>(static_cast<size_t>(params.batch_size) * params.k);
    func(actual_scores_d.get(), actual_indices_d.get());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> actual_scores_h(static_cast<size_t>(params.batch_size) * params.k);
    std::vector<int> actual_indices_h(static_cast<size_t>(params.batch_size) * params.k);
    CUDA_CHECK(cudaMemcpy(actual_scores_h.data(), actual_scores_d.get(), actual_scores_h.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_indices_h.data(), actual_indices_d.get(), actual_indices_h.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    ASSERT_TRUE(CompareResults(ref_scores_h, ref_indices_h, actual_scores_h, actual_indices_h, name));
    std::cout << "  [PASS] " << name << " (Probabilities & Indices)" << std::endl;

    // --- Raw Score Verification for Hybrid Sort ---
    if (name.find("HYBRID") != std::string::npos) {
      std::vector<float> hybrid_raw_topk_scores_h(static_cast<size_t>(params.batch_size) * params.k);

      CUDA_CHECK(cudaMemcpy2D(hybrid_raw_topk_scores_h.data(),                    // dst
                              static_cast<size_t>(params.k) * sizeof(float),      // dpitch
                              topk_data->intermediate_scores_1.get(),             // src
                              Generators::cuda::kHybridSortMaxK * sizeof(float),  // spitch
                              static_cast<size_t>(params.k) * sizeof(float),      // width
                              params.batch_size,                                  // height
                              cudaMemcpyDeviceToHost));
      ASSERT_TRUE(CompareRawScores(ref_raw_topk_scores_h, hybrid_raw_topk_scores_h, name));
      std::cout << "  [PASS] " << name << " (Raw Scores)" << std::endl;
    }
  };

  if (params.k <= 64) {
    test_algo("SELECTION_SORT", [&](float* s_d, int* i_d) {
      // Selection sort modifies the input in place, so we use a copy.
      CUDA_CHECK(cudaMemcpy(scores_in_d_copy.get(), scores_in_d.get(), scores_in_h.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      Generators::cuda::RunTopKViaSelectionSort(topk_data.get(), stream, scores_in_d_copy.get(), s_d, i_d,
                                                params.vocab_size, params.batch_size, params.k, temperature);
    });

    for (int partition_size : {1024, 2048, 4096, 8192}) {
      if (partition_size > params.vocab_size) {
        continue;
      }
      std::string algo_name = "HYBRID (" + std::to_string(partition_size) + ")";
      test_algo(algo_name, [&](float* s_d, int* i_d) {
        Generators::cuda::RunTopKViaHybridSort(topk_data.get(), stream, scores_in_d.get(), s_d, i_d, params.vocab_size,
                                               params.batch_size, params.k, temperature, partition_size);
      });
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(TopKTests, ParityTests) {
  std::vector<TopKTestParams> test_cases = {{1, 10000, 50, 1.0f}, {2, 10000, 64, 1.0f},  {1, 32000, 1, 0.5f},
                                            {1, 32000, 16, 2.0f}, {1, 512000, 50, 1.0f}, {1, 1024, 18, 1.0f}};

  for (const auto& params : test_cases) {
    RunParityTests(params);
  }
}
