// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_sampling.h"
#include "cuda_topk_baseline.h" 

namespace Generators {
namespace cuda {

struct SamplingData;

// Finds the top-k scores and their indices from the input scores.
// This function internally benchmarks and selects the fastest algorithm (Selection, Bitonic, or Full Sort).
void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

// Fills a device buffer with random float data for testing purposes.
void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size);

void LaunchPopulateOffsets(int* offsets, int size, int batch_size, cudaStream_t stream);

// --- Functions related to benchmark
enum class TopKAlgorithm { SELECTION_SORT,
                           BITONIC_SORT,
                           FULL_SORT };

struct TopKConfig {
  TopKAlgorithm algorithm = TopKAlgorithm::FULL_SORT;
  int num_partitions = 0;
  int block_size = 256;
  int sort_size = 0;
};

TopKConfig BenchmarkAndGetBestAlgorithm(SamplingData* data, cudaStream_t stream, int vocab_size, int batch_size, int k);

// --- Functions below are exposed for testing/benchmarking purposes ---

// Runs the Top-K algorithm using an iterative selection sort approach.
void RunTopKViaSelectionSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

// Runs the Top-K algorithm by performing a full sort on the vocabulary.
void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

// Runs the Top-K algorithm using a map-reduce approach with a bitonic sort in shared memory.
void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size);

}  // namespace cuda
}  // namespace Generators
