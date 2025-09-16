// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace topk_common {

/**
 * @brief Reusable device function for Stage 1 of Top-K sorting algorithms.
 *
 * This function is called by a kernel to find the Top-K elements within a single
 * partition of the input data. It uses a two-phase stable sort to ensure correct
 * tie-breaking (score descending, index ascending) without the performance
 * overhead of packing data into 64-bit keys.
 *
 * How it works:
 * 1.  **Phase 1 (Sort by Index):** A key-value sort is performed where the inverted
 * index is the key and the score is the value. Sorting the inverted index in
 * descending order effectively sorts the original indices in ascending order.
 * The sort is performed "blocked-to-blocked" to prepare the data for the next phase.
 * 2.  **Phase 2 (Stable Sort by Score):** A second, stable key-value sort is
 * performed on the result of Phase 1. This time, the score is the key and
 * the index is the value. The stability of `cub::BlockRadixSort` guarantees
 * that for elements with equal scores, their relative order from Phase 1
 * (which was sorted by index) is preserved. This sort is "blocked-to-striped"
 * for efficient output.
 *
 * This approach correctly handles tie-breaking and is significantly faster than
 * using composite 64-bit keys.
 *
 * @tparam kBlockSize The number of threads in the thread block.
 * @tparam kPartitionSize The size of the data partition to sort.
 * @tparam K The number of top elements to find.
 * @tparam SharedStorage The type of the shared memory storage union from the calling kernel.
 * @param scores_in Pointer to the input scores in global memory.
 * @param intermediate_indices Pointer to the intermediate buffer for indices.
 * @param intermediate_scores Pointer to the intermediate buffer for scores.
 * @param vocab_size The total vocabulary size.
 * @param num_partitions The total number of partitions.
 * @param smem A reference to the shared memory union from the calling kernel.
 */
template <int kBlockSize, int kPartitionSize, int K, typename SharedStorage>
__device__ void FindPartitionTopK(const float* __restrict__ scores_in,
                                 int* __restrict__ intermediate_indices,
                                 float* __restrict__ intermediate_scores,
                                 int vocab_size,
                                 int num_partitions,
                                 SharedStorage& smem) {
  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using BlockRadixSortScore = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;
  using BlockRadixSortIndex = cub::BlockRadixSort<unsigned int, kBlockSize, ItemsPerThread, float>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_scores[ItemsPerThread];
  int thread_indices[ItemsPerThread];
  unsigned int inverted_indices[ItemsPerThread];

  // Load scores and indices for the current partition
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size) {
      thread_scores[i] = batch_scores_in[global_idx];
      thread_indices[i] = global_idx;
    } else {
      thread_scores[i] = -FLT_MAX;
      thread_indices[i] = INT_MAX;
    }
    inverted_indices[i] = INT_MAX - thread_indices[i];
  }

  // Phase 1: Sort by index (ascending) by sorting inverted index (descending).
  // The output remains in a blocked arrangement for the next sort.
  BlockRadixSortIndex(
      reinterpret_cast<typename BlockRadixSortIndex::TempStorage&>(smem))
      .SortDescending(inverted_indices, thread_scores);

  __syncthreads();

  // Re-create indices from the sorted inverted_indices.
  // The data remains in a blocked arrangement.
  for (int i = 0; i < ItemsPerThread; ++i) {
    thread_indices[i] = INT_MAX - inverted_indices[i];
  }

  // Phase 2: Stable sort by score (descending).
  // The output is converted to a striped arrangement for efficient global memory writes.
  BlockRadixSortScore(
      reinterpret_cast<typename BlockRadixSortScore::TempStorage&>(smem))
      .SortDescendingBlockedToStriped(thread_scores, thread_indices);

  // Write out the top K results for this partition
  if (threadIdx.x < K) {
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = thread_scores[0];
    intermediate_indices[offset] = thread_indices[0];
  }
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators

