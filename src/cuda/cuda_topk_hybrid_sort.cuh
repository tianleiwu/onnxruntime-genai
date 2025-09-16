// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_common.cuh"

namespace Generators {
namespace cuda {
namespace hybrid_sort {
// Stage 1 of Hybrid Sort: Finds the top-k elements within partitions of the vocabulary.
// This single, unified kernel handles both padded and non-padded cases with the "on-the-fly" padding logic.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void HybridSort_Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                                     int* __restrict__ intermediate_indices,
                                                     float* __restrict__ intermediate_scores,
                                                     int vocab_size, int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using BlockRadixSortScore = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;
  using BlockRadixSortIndex = cub::BlockRadixSort<unsigned int, kBlockSize, ItemsPerThread, float>;

  union SharedStorage {
    typename BlockRadixSortScore::TempStorage stage1_storage_score;
    typename BlockRadixSortIndex::TempStorage stage1_storage_index;
  };
  __shared__ SharedStorage smem;

  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

// Helper to calculate the size of intermediate buffers needed by hybrid sort.
inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kHybridSortMaxK;
}

inline int GetPartitionsPerBlock(int num_partitions) {
  if (num_partitions <= 2) return 2;
  if (num_partitions <= 4) return 4;
  return 8;
}

// Kernel for the special case where the vocab fits in one partition. It takes the
// unsorted top-K candidates from Stage 1 and performs a final sort in shared
// memory to produce the true top-k result.
template <int kBlockSize, int K_padded>
__global__ void FinalSinglePartitionSort(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
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

// Stage 2 of Hybrid Sort: Iteratively reduces partitions to find the final top-K.
// This logic is shared between the standard and Flash versions.
template <int K>
void HybridSort_ReducePartitions(TopkData* data, cudaStream_t stream, int num_partitions, int batch_size, int k) {
  if (num_partitions == 1) {
    // Special case: Vocab fits in one partition. Stage 1 found the top K candidates,
    // but they are not fully sorted. We must perform a final, single-block sort on
    // these K candidates to get the true top k. This is more efficient than running
    // the multi-partition reduction loop for this scenario.
    constexpr int block_size = 256;
    dim3 grid(1, batch_size);
    dim3 block(block_size);

    FinalSinglePartitionSort<block_size, K><<<grid, block, 0, stream>>>(
        data->intermediate_scores_1, data->intermediate_indices_1,
        data->intermediate_scores_2, data->intermediate_indices_2,
        k);
    CUDA_CHECK(cudaGetLastError());

    // The final sorted data is in buffer 2. The output is compact with size k.
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
    data->topk_stride = k;
    return;
  }

  int current_num_partitions = num_partitions;
  float* input_scores = data->intermediate_scores_1;
  float* output_scores = data->intermediate_scores_2;
  int* input_indices = data->intermediate_indices_1;
  int* output_indices = data->intermediate_indices_2;
  constexpr int block_size = 256;

  while (current_num_partitions > 1) {
    const int partitions_per_block = GetPartitionsPerBlock(current_num_partitions);
    const int num_blocks = CeilDiv(current_num_partitions, partitions_per_block);
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);

    switch (partitions_per_block) {
      case 8:
        bitonic_sort::BlockReduceTopK<block_size, K, 8><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      case 4:
        bitonic_sort::BlockReduceTopK<block_size, K, 4><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      default:
        bitonic_sort::BlockReduceTopK<block_size, K, 2><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);
    current_num_partitions = num_blocks;
  }

  data->topk_scores = input_scores;
  data->topk_indices = input_indices;
  data->topk_stride = K;
  CUDA_CHECK(cudaGetLastError());
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(k <= kHybridSortMaxK);  // The caller shall ensure k does not exceed the maximum allowed for hybrid sort.
  constexpr int block_size = 256;

  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  auto launch_stage1_flash = [&](auto k_const) {
    constexpr int K = decltype(k_const)::value;
    switch (partition_size) {
      case 1024:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 1024, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
        break;
      case 2048:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 2048, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
        break;
      case 4096:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 4096, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
        break;
      // default:
      //   HybridSort_Stage1_FindPartitionsTopK<block_size, 8192, K><<<grid_stage1, block_stage1, 0, stream>>>(
      //       scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
      //   break;
    }
    CUDA_CHECK(cudaGetLastError());
    HybridSort_ReducePartitions<K>(data, stream, num_partitions, batch_size, k);
  };

  // This kernel is optimized for large vocab_size and large k since flash sort or LLM sort is preferred for smaller vocab_size and smaller k.
  if (k <= 64) {
    launch_stage1_flash(std::integral_constant<int, 64>());
    return;
  }

  if (k <= 128) {
    launch_stage1_flash(std::integral_constant<int, 128>());
    return;
  }

  static_assert(kHybridSortMaxK == 128 || kHybridSortMaxK == 256);
  if constexpr (kHybridSortMaxK > 128) {
    launch_stage1_flash(std::integral_constant<int, kHybridSortMaxK>());
  }
}

/**
 * @brief Estimates the best partition size for the hybrid Top-K sorting algorithm.
 */
inline int EstimateBestPartitionSize(int vocab_size) {
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  // if (vocab_size <= 4096) return 4096;
  // // TODO: This is tuned when reduction factor is 2.
  // // We need revisit it since we allow reduction factors 2, 4, 8 now.
  // return 8192;
  return 4096;
}

}  // namespace hybrid_sort
}  // namespace cuda
}  // namespace Generators

