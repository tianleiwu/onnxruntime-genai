// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
// Stage 1 of Hybrid Sort: Find the top-k elements within large, contiguous partitions of the vocabulary.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CubRegisterSort(const float* __restrict__ scores_in,
                                              int* __restrict__ intermediate_indices,
                                              float* __restrict__ intermediate_scores, int vocab_size,
                                              int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + batch_idx * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

  // Coalesced load from global memory into per-thread registers.
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
      thread_keys[i] = batch_scores_in[global_idx];
      thread_values[i] = global_idx;
    } else {
      thread_keys[i] = -FLT_MAX;
      thread_values[i] = -1;
    }
  }

  // Sort the keys and values held in registers across the entire block.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  // The first K threads now hold the top K elements for this partition. Write them out.
  if (threadIdx.x < K) {
    int offset = (batch_idx * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

// Helper to calculate the size of intermediate buffers needed by hybrid sort.
inline size_t GetHybridSortIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  return static_cast<size_t>(batch_size) * num_partitions * kHybridSortMaxK;
}

void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int max_k = kHybridSortMaxK;
  constexpr int block_size = 256;
  static_assert(kHybridSortMaxK <= block_size);

  int partition_size = data->hybrid_sort_partition_size;

  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  // Stage 1: Find Top-K within partitions.
  // The results are written to intermediate buffers.
  switch (partition_size) {
    case 1024:
      FindBlockTopK_CubRegisterSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 2048:
      FindBlockTopK_CubRegisterSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 4096:
      FindBlockTopK_CubRegisterSort<block_size, 4096, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 8192:
      FindBlockTopK_CubRegisterSort<block_size, 8192, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    default:
      assert(false && "Unsupported partition_size");
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Iteratively reduce the candidates from each partition until only one partition remains.
  // This uses a ping-pong buffer scheme for scores and indices.
  int current_num_partitions = num_partitions;
  float* input_scores = data->intermediate_scores_1.get();
  float* output_scores = data->intermediate_scores_2.get();
  int* input_indices = data->intermediate_indices_1.get();
  int* output_indices = data->intermediate_indices_2.get();

  while (current_num_partitions > 1) {
    constexpr int partitions_per_block = 8;
    int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);
    bitonic::reduction::BlockReduceTopK<block_size, max_k, partitions_per_block>
        <<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices,
                                                   current_num_partitions);
    CUDA_CHECK(cudaGetLastError());
    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);
    current_num_partitions = num_blocks;
  }

  // After reduction, input_scores and input_indices point to the device buffers containing the final top-`max_k` raw scores and indices.
  data->topk_scores = input_scores;
  data->topk_indices = input_indices;
  data->topk_stride = max_k;
  CUDA_CHECK(cudaGetLastError());
}


/**
 * @brief Estimates the best partition size for the hybrid Top-K sorting algorithm.
 *
 * This function uses a heuristic based on extensive benchmarking and analysis of the
 * hybrid sort's two-stage nature. The goal is to select a partition size that
 * creates an optimal amount of parallel work to saturate the GPU, while minimizing
 * the overhead of the reduction stage.
 *
 * The heuristic considers:
 * 1.  The trivial case where the entire vocabulary fits in a single partition.
 * 2.  The need to generate sufficient parallelism for small batch sizes.
 * 3.  The goal of matching the total number of workstreams to the GPU's number of
 * Streaming Multiprocessors (SMs) for larger batch sizes.
 *
 * @param batch_size The number of batches being processed.
 * @param vocab_size The size of the vocabulary to sort over.
 * @param device_prop The CUDA device properties, used to get the SM count.
 * @return The estimated optimal partition size (e.g., 1024, 2048, 4096, 8192).
 */
inline int EstimateHybridSortBestPartitionSize(int batch_size, int vocab_size, const cudaDeviceProp& /*device_prop*/) {
  const std::vector<int> available_partition_sizes = {1024, 2048, 4096, 8192};

  // --- Rule 1: Single Partition Dominance ---
  // If the entire vocabulary fits within one of the available partition sizes,
  // the smallest one that fits is always the most efficient choice. This avoids
  // the reduction stage (Stage 2) entirely.
  for (int p_size : available_partition_sizes) {
    if (vocab_size <= p_size) {
      return p_size;
    }
  }

  // Helper lambda to find the partition size that gets closest to a target number of partitions.
  auto find_closest = [&](int target_partitions) {
    long min_diff = -1;
    int best_p_size = available_partition_sizes.back();
    // Iterate in descending order to prefer smaller partition sizes in a tie-break,
    // as the heuristic targets are tuned for this behavior.
    for (size_t i = 0; i < available_partition_sizes.size(); ++i) {
      int p_size = available_partition_sizes[available_partition_sizes.size() - 1 - i];
      int num_partitions = (vocab_size + p_size - 1) / p_size;
      long diff = std::abs(num_partitions - target_partitions);

      if (min_diff == -1 || diff <= min_diff) {
        min_diff = diff;
        best_p_size = p_size;
      }
    }
    return best_p_size;
  };

  if (batch_size <= 2) {
    // --- Rule 2: Small Batch Size Heuristic (batch <= 2) ---
    // This logic is tuned based on empirical data for small batches where performance
    // can be non-monotonic and requires specific ranges.
    if (vocab_size > 40000 && vocab_size < 50000) return 8192; // Specific exception for this range
    if (vocab_size < 80000) return find_closest(8);
    // In this range, the optimal number of partitions dips.
    if (vocab_size >= 140000 && vocab_size < 180000) return find_closest(36);
    // For other ranges (80k-140k and >180k), a higher number of partitions is more efficient.
    return find_closest(56);
  } else { // batch_size > 2
    // --- Rule 3: Large Batch Size Heuristic (batch > 2) ---
    // With more batches, the optimal number of partitions is more stable and primarily
    // depends on creating enough work without excessive reduction overhead.
    if (vocab_size < 81920) return find_closest(8);
    if (vocab_size >= 524288) return find_closest(64);
    // The broad mid-range performs best when targeting ~32 partitions.
    return find_closest(32);
  }
}

}  // namespace cuda
}  // namespace Generators
