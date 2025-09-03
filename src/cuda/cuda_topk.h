// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_sampling.h"

namespace Generators {
namespace cuda {

constexpr int kBitonicSortMaxPartitions = 256;
constexpr int kBitonicSortMaxK = 64;

struct SamplingData;

// Finds the top-k scores and their indices from the input scores.
// This function internally benchmarks and selects the fastest algorithm (Selection, Bitonic, or Full Sort).
void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

void LaunchPopulateOffsets(int* offsets, int size, int batch_size, cudaStream_t stream);

enum class TopKAlgorithm { SELECTION_SORT,
                           BITONIC_SORT,
                           HYBRID_SORT,
                           FULL_SORT };

// --- Functions below are exposed for testing/benchmarking purposes ---

// Fills a device buffer with random float data for testing purposes.
void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size);

// Runs the Top-K algorithm using an iterative selection sort approach.
void RunTopKViaSelectionSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

// Runs the Top-K algorithm by performing a full sort on the vocabulary.
void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

// Runs the Top-K algorithm using a map-reduce approach with bitonic sort in shared memory.
void RunTopKViaBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int partition_size);

// Runs the Top-K algorithm using a map-reduce approach with a hybrid: Radix sort in merge, and Bitonic sort in reduction. 
void RunTopKViaHybridSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int partition_size);

}  // namespace cuda
}  // namespace Generators
