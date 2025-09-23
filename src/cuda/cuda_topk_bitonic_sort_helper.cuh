// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include <math_constants.h>
#include "cuda_topk.h"
#include <cub/warp/warp_merge_sort.cuh>
#include "cuda_topk_stable_sort_helper.cuh"  // For Pack/Unpack functions

namespace Generators {
namespace cuda {
namespace bitonic_sort {

/**
 * @brief Performs an in-place bitonic sort on data in shared memory.
 * This specialized version is for when the number of threads (`kBlockSize`)
 * is greater than or equal to the number of items to sort (`SortSize`).
 * Each element is handled by a dedicated thread, leading to high parallelism.
 */
template <int kBlockSize, int SortSize>
__device__ void _SharedMemBitonicSort_Small(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize >= SortSize);

  // This implementation uses one thread per element for the sort.
  const int ix = threadIdx.x;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      if (ix < SortSize) {
        int paired_ix = ix ^ j;
        if (paired_ix > ix) {
          bool ascending = ((ix & k) == 0);

#ifdef STABLE_TOPK
          // For stable sort, include tie-breaking logic (smaller index wins).
          bool is_ix_greater = (smem_scores[ix] > smem_scores[paired_ix]) ||
                               (smem_scores[ix] == smem_scores[paired_ix] && smem_indices[ix] < smem_indices[paired_ix]);
#else
          // For unstable sort, no tie-breaking is needed for performance.
          bool is_ix_greater = smem_scores[ix] > smem_scores[paired_ix];
#endif
          // For a descending sort, swap if the greater element is NOT in the correct position
          // according to the bitonic sequence direction.
          if (is_ix_greater != ascending) {
            float temp_score = smem_scores[ix];
            smem_scores[ix] = smem_scores[paired_ix];
            smem_scores[paired_ix] = temp_score;

            int temp_index = smem_indices[ix];
            smem_indices[ix] = smem_indices[paired_ix];
            smem_indices[paired_ix] = temp_index;
          }
        }
      }
    }
  }
}

/**
 * @brief A generic, in-place bitonic sort on data in shared memory.
 * This version handles cases where there are fewer threads than elements to sort.
 * Threads loop to cover all necessary comparisons in the sort network.
 */
template <int kBlockSize, int SortSize>
__device__ void _SharedMemBitonicSort_Big(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0, "SortSize must be power of two");

  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < N; i += kBlockSize) {
        const int ixj = i ^ j;
        if (ixj > i) {
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          bool ascending = ((i & k) == 0);

#if STABLE_TOPK
          bool is_i_greater = (a_i > a_j) || (a_i == a_j && idx_i < idx_j);
#else
          bool is_i_greater = a_i > a_j;
#endif

          // For a descending sort, swap if the greater element is NOT in the correct position
          // according to the bitonic sequence direction.
          if (is_i_greater != ascending) {
            smem_scores[i] = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i] = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }
}

/**
 * @brief A dispatch wrapper for shared memory bitonic sort.
 * At compile time, it selects the optimal implementation (`_Small` or `_Big`)
 * based on the relationship between block size and sort size.
 * The caller might need call __syncthreads() before and after this function.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(float* smem_scores, int* smem_indices) {
  if constexpr (kBlockSize >= SortSize) {
    _SharedMemBitonicSort_Small<kBlockSize, SortSize>(smem_scores, smem_indices);
  } else {
    _SharedMemBitonicSort_Big<kBlockSize, SortSize>(smem_scores, smem_indices);
  }
}

/**
 * @brief Performs an in-place, warp-wide bitonic sort on data held entirely in registers.
 *
 * This function sorts `warpSize` (typically 32) score/index pairs distributed across the
 * threads of a single warp. It uses `__shfl_sync` instructions for extremely fast
 * data exchange between threads in the same warp, avoiding shared memory latency entirely.
 * This is highly effective for the reduction phase of algorithms like `iterative_sort` when `k` is small.
 */
__device__ inline void WarpBitonicSort(float& score, int& index) {
  const int lane_id = threadIdx.x % warpSize;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= warpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      // Exchange data with a paired lane using a warp shuffle instruction.
      int paired_lane = lane_id ^ j;
      float paired_score = __shfl_sync(0xFFFFFFFF, score, paired_lane);
      int paired_index = __shfl_sync(0xFFFFFFFF, index, paired_lane);

      // A standard bitonic network sorts ascending with `(lane_id & k) == 0`.
      // The swap condition is inverted as needed to produce an overall descending sort.
      bool direction = ((lane_id & k) == 0);

#ifdef STABLE_TOPK
      // For stable sort, include tie-breaking logic (smaller index wins for equal scores).
      bool is_mine_greater = (score > paired_score) || (score == paired_score && index < paired_index);
#else
      bool is_mine_greater = score > paired_score;
#endif

      // In-register min/max calculation.
      float s_max = is_mine_greater ? score : paired_score;
      int i_max = is_mine_greater ? index : paired_index;
      float s_min = is_mine_greater ? paired_score : score;
      int i_min = is_mine_greater ? paired_index : index;

      // Redistribute the min/max values based on the sort direction for this stage.
      if (direction) {
        score = (lane_id < paired_lane) ? s_max : s_min;
        index = (lane_id < paired_lane) ? i_max : i_min;
      } else {
        score = (lane_id < paired_lane) ? s_min : s_max;
        index = (lane_id < paired_lane) ? i_min : i_max;
      }
    }
  }
}

// Functor for descending sort comparison.
template <typename T>
struct Greater {
  __device__ __host__ __forceinline__ bool operator()(const T& a, const T& b) const {
    return a > b;
  }
};

/**
 * @brief Performs an in-place, warp-wide merge sort on data in shared memory using CUB.
 *
 * This function uses a single warp (32 threads) to sort up to 256 key-value pairs.
 * Each thread in the warp manages `kItemsPerThread` items in its registers. This approach
 * is highly efficient for small sort sizes that fit within a single warp's processing capacity.
 * It is designed to be called by a full thread block, where only the first warp performs work.
 * The caller might need call __syncthreads() before and after this function.
 */
template <int BufferSize>
__device__ void WarpMergeSort(float* smem_scores, int* smem_indices, void* temp_storage_ptr, int num_valid_items) {
  static_assert(BufferSize <= 256, "BufferSize must be less than or equal to 256");
  constexpr int kThreadsInWarp = 32;

  // This sort is performed entirely by the first warp in the block.
  if (threadIdx.x >= kThreadsInWarp) {
    return;
  }

  constexpr int kItemsPerThread = (BufferSize + kThreadsInWarp - 1) / kThreadsInWarp;

#ifdef STABLE_TOPK
  // --- Stable Sort Path: Pack into uint64_t and sort keys-only ---
  using WarpSortT = cub::WarpMergeSort<uint64_t, kItemsPerThread, kThreadsInWarp, cub::NullType>;
  typename WarpSortT::TempStorage& temp_storage = *static_cast<typename WarpSortT::TempStorage*>(temp_storage_ptr);

  uint64_t thread_keys[kItemsPerThread];

  // Load, pack, and pad based on the number of valid items.
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = threadIdx.x + i * kThreadsInWarp;
    if (idx < num_valid_items) {
      thread_keys[i] = topk_common::PackStableSortKey(smem_scores[idx], smem_indices[idx]);
    } else {
      thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
    }
  }

  // Sort descending using a "greater than" comparator.
  WarpSortT(temp_storage).Sort(thread_keys, Greater<uint64_t>());

  // Unpack and write back to the full buffer.
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = threadIdx.x * kItemsPerThread + i;
    if (idx < BufferSize) {
      smem_scores[idx] = topk_common::UnpackStableSortScore(thread_keys[i]);
      smem_indices[idx] = topk_common::UnpackStableSortIndex(thread_keys[i]);
    }
  }
#else
  // --- Unstable Sort Path: Sort key-value pairs ---
  using WarpSortT = cub::WarpMergeSort<float, kItemsPerThread, kThreadsInWarp, int>;
  typename WarpSortT::TempStorage& temp_storage = *static_cast<typename WarpSortT::TempStorage*>(temp_storage_ptr);

  float thread_scores[kItemsPerThread];
  int thread_indices[kItemsPerThread];

  // Load and pad based on the number of valid items.
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = threadIdx.x + i * kThreadsInWarp;
    if (idx < num_valid_items) {
      thread_scores[i] = smem_scores[idx];
      thread_indices[i] = smem_indices[idx];
    } else {
      thread_scores[i] = -FLT_MAX;
      thread_indices[i] = INT_MAX;
    }
  }

  // Sort descending using a "greater than" comparator.
  WarpSortT(temp_storage).Sort(thread_scores, thread_indices, Greater<float>());

  // Write back to the full buffer.
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = threadIdx.x * kItemsPerThread + i;
    if (idx < BufferSize) {
      smem_scores[idx] = thread_scores[i];
      smem_indices[idx] = thread_indices[i];
    }
  }
#endif
}

}  // namespace bitonic_sort
}  // namespace cuda
}  // namespace Generators
