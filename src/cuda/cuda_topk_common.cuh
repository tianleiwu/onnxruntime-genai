// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace topk_common {

/**
 * @brief Optimized device function for Stage 1 of Top-K sorting using a
 * "rank-then-select" strategy.
 *
 * This function is called by a kernel to find the Top-K candidates within a
 * single partition. It is highly performant because it avoids sorting the
 * entire partition. It relies on the stability of `cub::BlockRadixRank` to
 * handle tie-breaking correctly without expensive key-packing operations.
 *
 * How it works:
 * 1.  **Key Creation**: For each element, the float score is converted to a
 * sortable `unsigned int` representation.
 * 2.  **Stable Ranking**: `cub::BlockRadixRank` is used to calculate the stable
 * rank of each score. Because the input data is loaded in a blocked
 * arrangement (i.e., in ascending index order), the stability of the rank
 * operation automatically ensures that for identical scores, the element with
 * the smaller original index gets a smaller rank. This correctly handles the
 * tie-breaking rule.
 * 3.  **Selective Scatter**: After ranking, each thread checks the ranks of its
 * local items. If an item's rank is less than K, the thread writes that item's
 * original score and index directly to the correct output slot in global memory.
 *
 * @tparam kBlockSize The number of threads in the thread block.
 * @tparam kPartitionSize The size of the data partition to sort.
 * @tparam K The number of top elements to find.
 * @tparam SharedStorage The type of the shared memory storage from the calling kernel.
 */
template <int kBlockSize, int kPartitionSize, int K, typename SharedStorage>
__device__ void FindPartitionTopK_Rank(const float* __restrict__ scores_in,
                                      int* __restrict__ intermediate_indices,
                                      float* __restrict__ intermediate_scores,
                                      int vocab_size,
                                      int num_partitions,
                                      SharedStorage& smem) {
  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  constexpr int RadixBits = 4;
  using BlockRadixRank = cub::BlockRadixRank<kBlockSize, RadixBits, true>; // true for descending

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_scores[ItemsPerThread];
  int thread_indices[ItemsPerThread];
  unsigned int sortable_scores[ItemsPerThread];

  // Load data and create sortable unsigned int keys from float scores
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size) {
      thread_scores[i] = batch_scores_in[global_idx];
      thread_indices[i] = global_idx;
    } else {
      thread_scores[i] = -FLT_MAX;
      thread_indices[i] = INT_MAX;
    }

    unsigned int score_bits = __float_as_uint(thread_scores[i]);
    sortable_scores[i] = (score_bits & 0x80000000) ? (~score_bits) : (score_bits | 0x80000000);
  }

  // Rank the sortable score keys
  int ranks[ItemsPerThread];
  cub::BFEDigitExtractor<unsigned int> digit_extractor(0, sizeof(unsigned int) * 8);
  BlockRadixRank(reinterpret_cast<typename BlockRadixRank::TempStorage&>(smem))
      .RankKeys(sortable_scores, ranks, digit_extractor);

  __syncthreads();

  // Selective scatter: write out only the top K elements
  for (int i = 0; i < ItemsPerThread; ++i) {
    if (ranks[i] < K) {
      size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + ranks[i];
      intermediate_scores[offset] = thread_scores[i];
      intermediate_indices[offset] = thread_indices[i];
    }
  }
}

/**
 * @brief Kernel to perform a final, stable sort on K candidates when there is
 * only one partition.
 *
 * This is used by sorting algorithms when the reduction phase is skipped
 * (i.e., num_partitions == 1) to ensure the final output is correctly sorted
 * with proper tie-breaking.
 *
 * @tparam kBlockSize The number of threads in the thread block.
 * @tparam K_padded The padded size of K, which must be a power of two for bitonic sort.
 * @param scores_in The unsorted Top-K candidate scores.
 * @param indices_in The unsorted Top-K candidate indices.
 * @param scores_out The final, sorted Top-K scores.
 * @param indices_out The final, sorted Top-K indices.
 * @param k_final The actual value of K (the number of elements to write).
 */
template <int kBlockSize, int K_padded>
__global__ void FinalSort(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                          float* __restrict__ scores_out, int* __restrict__ indices_out, int k_final) {
  __shared__ float smem_scores[K_padded];
  __shared__ int smem_indices[K_padded];

  const int batch_idx = blockIdx.y;
  const size_t in_offset = static_cast<size_t>(batch_idx) * K_padded;
  const size_t out_offset = static_cast<size_t>(batch_idx) * k_final;

  // Load the K_padded candidates into shared memory
  for (int i = threadIdx.x; i < K_padded; i += kBlockSize) {
    smem_scores[i] = scores_in[in_offset + i];
    smem_indices[i] = indices_in[in_offset + i];
  }
  __syncthreads();

  // Sort the K_padded candidates in shared memory
  bitonic_sort::SharedMemBitonicSort<kBlockSize, K_padded>(smem_scores, smem_indices);

  // Write out the final top-k_final results
  if (threadIdx.x < k_final) {
    scores_out[out_offset + threadIdx.x] = smem_scores[threadIdx.x];
    indices_out[out_offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators

