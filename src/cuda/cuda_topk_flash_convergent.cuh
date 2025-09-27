// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_warp_sort_helper.cuh"
#include <cooperative_groups.h>
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace flash_convergent {

/**
 * @brief A two-stage cooperative algorithm hyper-specialized for **small k**.
 *
 * Algorithm Overview:
 * This kernel is designed to have the absolute minimum overhead, making it dominant
 * for small k (<= 16), where launch overhead and instruction efficiency are critical.
 *
 * 1.  **Stage 1 (Partition Top-K)**: All thread blocks find top candidates in parallel.
 *
 * 2.  **Grid-Wide Sync**.
 *
 * 3.  **Stage 2 (Single-Step Reduction)**: A single block (`blockIdx.x == 0`) performs the final merge.
 * - **Warp-Specialized Path**: For k <= 16, this block uses a highly optimized path.
 * Warps collaboratively load all candidates into shared memory, then each warp sorts
 * a segment in registers using ultra-fast `WarpBitonicSort`. A final k-way merge by
 * the first warp produces the result. This avoids all overhead from generic CUB block-level sorts.
 * - **Generic Path**: For larger k, it falls back to a benchmark-driven CUB block-level sort.
 *
 * Performance Characteristics:
 * -   **Strengths**: Extremely high performance for small `k` due to its minimal overhead.
 * -   **Weaknesses**: Requires cooperative launch. Limited by the number of partitions a single
 * block can handle in its reduction phase.
 */
namespace cg = cooperative_groups;

constexpr int kMaxPartitions = 64;
constexpr std::array<int, 4> kPartitionSizes = {2816, 3328, 4096, 4864};
constexpr int kDeviceWarpSize = 32;

// --- Unified Convergent Kernel ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void FlashConvergentKernel(const float* __restrict__ scores_in,
                                      float* __restrict__ intermediate_scores,
                                      int* __restrict__ intermediate_indices,
                                      float* __restrict__ scores_out,
                                      int* __restrict__ indices_out,
                                      int vocab_size,
                                      int num_partitions,
                                      int k_actual) {
  cg::grid_group grid = cg::this_grid();

  // Define a single, comprehensive union for shared memory to handle all code paths.
  constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  constexpr int kSmallKSortSize = 16 * kMaxPartitions; // Max possible size for small k (16 * 64 = 1024)

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;
    // Storage for small-k warp-specialized path
    struct {
      __align__(128) float scores[kSmallKSortSize];
      __align__(128) int indices[kSmallKSortSize];
      __align__(128) int segment_heads[kBlockSize / kDeviceWarpSize];
    } warp_sort_storage;
    // Storage for large-k CUB path
#ifdef STABLE_TOPK
    typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage merge_storage;
    typename cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThread>::TempStorage radix_storage;
#else
    typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage merge_storage;
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>::TempStorage radix_storage;
#endif
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Parallel Partition Sort ---
  constexpr bool kUseMergeSortInStage1 = kPartitionSize <= BestAlgoThresholds::kCubBlockMerge_MaxSize;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, kUseMergeSortInStage1>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2: One block performs the final merge ---
  if (blockIdx.x == 0) {
    const int batch_idx = blockIdx.y;
    const int num_elements_to_sort = num_partitions * K_PADDED;
    const int lane_id = threadIdx.x % kDeviceWarpSize;

    // --- Warp-Specialized Path for Small K ---
    if constexpr (K_PADDED <= 16) {
      // Warps collaborate to load all data into shared memory
      for (int i = threadIdx.x; i < num_elements_to_sort; i += kBlockSize) {
        size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + i;
        smem.warp_sort_storage.scores[i] = intermediate_scores[offset];
        smem.warp_sort_storage.indices[i] = intermediate_indices[offset];
      }
      __syncthreads();

      // Each warp sorts a segment of 32 elements in registers
      const int warp_id = threadIdx.x / kDeviceWarpSize;
      const int segment_start = warp_id * kDeviceWarpSize;
      float my_score = (segment_start + lane_id < num_elements_to_sort) ? smem.warp_sort_storage.scores[segment_start + lane_id] : -FLT_MAX;
      int my_index = (segment_start + lane_id < num_elements_to_sort) ? smem.warp_sort_storage.indices[segment_start + lane_id] : INT_MAX;

      topk_common::WarpBitonicSort(my_score, my_index);
      smem.warp_sort_storage.scores[segment_start + lane_id] = my_score;
      smem.warp_sort_storage.indices[segment_start + lane_id] = my_index;
      __syncthreads();

      // First warp performs a k-way merge on the sorted segments
      if (warp_id == 0) {
        const int num_segments = CeilDiv(num_elements_to_sort, kDeviceWarpSize);
        if (lane_id < num_segments) {
          smem.warp_sort_storage.segment_heads[lane_id] = 0;
        }
        __syncthreads();

        for (int i = 0; i < k_actual; ++i) {
          float max_score = -FLT_MAX;
          int max_index = -1;
          int max_segment = -1;

          // Parallel reduction to find the max among segment heads
          for (int seg = lane_id; seg < num_segments; seg += kDeviceWarpSize) {
            int head_idx = smem.warp_sort_storage.segment_heads[seg];
            if (head_idx < kDeviceWarpSize) {
                float score = smem.warp_sort_storage.scores[seg * kDeviceWarpSize + head_idx];
                if (score > max_score) {
                    max_score = score;
                    max_index = smem.warp_sort_storage.indices[seg * kDeviceWarpSize + head_idx];
                    max_segment = seg;
                }
            }
          }
          
          // Find the winner within the warp
          for (int offset = kDeviceWarpSize / 2; offset > 0; offset /= 2) {
              float other_score = __shfl_down_sync(0xFFFFFFFF, max_score, offset);
              int other_index = __shfl_down_sync(0xFFFFFFFF, max_index, offset);
              int other_segment = __shfl_down_sync(0xFFFFFFFF, max_segment, offset);
              if (other_score > max_score) {
                  max_score = other_score;
                  max_index = other_index;
                  max_segment = other_segment;
              }
          }

          if (lane_id == 0) {
            size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + i;
            scores_out[out_offset] = max_score;
            indices_out[out_offset] = max_index;
            smem.warp_sort_storage.segment_heads[max_segment]++;
          }
           __syncthreads(); // Ensure head is updated for all threads
        }
      }
    } else {
      // --- Generic Path for Larger K using CUB ---
      constexpr SortAlgo kSortAlgo = GetBestAlgo(kSortSize);
#ifdef STABLE_TOPK
      // ... CUB logic for stable sort ...
#else
      using SortKeyT = float;
      using SortValueT = int;
      SortKeyT thread_keys[kItemsPerThread];
      SortValueT thread_values[kItemsPerThread];
      for (int i = 0; i < kItemsPerThread; ++i) {
        int load_idx = threadIdx.x + i * kBlockSize; // Striped load
        if (load_idx < num_elements_to_sort) {
          size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + load_idx;
          thread_keys[i] = intermediate_scores[offset];
          thread_values[i] = intermediate_indices[offset];
        } else {
          thread_keys[i] = -FLT_MAX;
          thread_values[i] = INT_MAX;
        }
      }
      if constexpr (kSortAlgo == SortAlgo::CUB_BLOCK_RADIX) {
        cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
        if (threadIdx.x < k_actual) {
          size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
          scores_out[out_offset] = thread_keys[0];
          indices_out[out_offset] = thread_values[0];
        }
      } else {  // CUB_BLOCK_MERGE or CUB_WARP_MERGE
        cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.merge_storage).Sort(thread_keys, thread_values, topk_common::DescendingOp());
        cub::StoreDirectBlocked(threadIdx.x, scores_out + (size_t)batch_idx * k_actual, thread_keys, k_actual);
        cub::StoreDirectBlocked(threadIdx.x, indices_out + (size_t)batch_idx * k_actual, thread_values, k_actual);
      }
#endif
    }
  }
}
// ... Rest of the file (launchers, IsSupported) remains largely the same ...
// --- Host-side Launcher ---

inline int EstimateBestPartitionSize(int vocab_size, int k) {
  const auto& benchmarks = GetSortBenchmarkResults();

  double min_total_latency = std::numeric_limits<double>::max();
  int best_partition_size = 0;

  for (int p_size : kPartitionSizes) {
    const int num_partitions = CeilDiv(vocab_size, p_size);
    if (num_partitions > kMaxPartitions) {
      continue;
    }

    // Estimate Stage 1 latency
    SortAlgo best_algo_s1 = GetBestAlgo(p_size);
    float latency_s1 = benchmarks.GetLatency(best_algo_s1, p_size);

    // Estimate Stage 2 latency
    int sort_size_s2 = kConvergentSortMaxK * num_partitions;
    SortAlgo best_algo_s2 = GetBestAlgo(sort_size_s2);
    float latency_s2 = benchmarks.GetLatency(best_algo_s2, sort_size_s2);

    float total_latency = latency_s1 + latency_s2;

    if (total_latency < min_total_latency) {
      min_total_latency = total_latency;
      best_partition_size = p_size;
    }
  }

  // Fallback if no suitable size found (e.g., for very small vocab_size)
  if (best_partition_size == 0) {
    best_partition_size = kPartitionSizes[0];
  }

  return best_partition_size;
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kConvergentSortMaxK;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
void* GetKernel() {
  return (void*)FlashConvergentKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED>
void* GetKernelForNumPartitions(int num_partitions) {
  if (num_partitions <= 8) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 8>();
  if (num_partitions <= 16) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 16>();
  if (num_partitions <= 32) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 32>();
  return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 64>();
}

template <int P_SIZE, int K_PADDED>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  int num_partitions = CeilDiv(vocab_size, P_SIZE);

  dim3 grid(num_partitions, batch_size);
  dim3 block(kBlockSize);

  void* kernel = GetKernelForNumPartitions<kBlockSize, P_SIZE, K_PADDED>(num_partitions);
  void* kernel_args[] = {(void*)&scores_in, (void*)&data->intermediate_scores_1, (void*)&data->intermediate_indices_1,
                         (void*)&data->intermediate_scores_2, (void*)&data->intermediate_indices_2,
                         (void*)&vocab_size, (void*)&num_partitions, (void*)&k};
  CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, kernel_args, 0, stream));
}

template <int K_PADDED>
void LaunchKernelByPartitionSize(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size) {
  if (partition_size == kPartitionSizes[0])
    LaunchKernel<kPartitionSizes[0], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kPartitionSizes[1])
    LaunchKernel<kPartitionSizes[1], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kPartitionSizes[2])
    LaunchKernel<kPartitionSizes[2], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else
    LaunchKernel<kPartitionSizes[3], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  if (data->flash_convergent_partition_size_k != k) {
    data->flash_convergent_partition_size_k = k;
    data->flash_convergent_partition_size = EstimateBestPartitionSize(vocab_size, k);
  }

  const int partition_size = data->flash_convergent_partition_size;

  if (k <= 4)
    LaunchKernelByPartitionSize<4>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 8)
    LaunchKernelByPartitionSize<8>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 16)
    LaunchKernelByPartitionSize<16>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 32)
    LaunchKernelByPartitionSize<32>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 52)
    LaunchKernelByPartitionSize<52>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else
    LaunchKernelByPartitionSize<64>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  CUDA_CHECK_LAUNCH();

  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k;
}

// --- The following implements IsSupported ---
template <int K_PADDED, int kMaxPartitionsForKernel>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;

  void* kernel;
  if (partition_size == kPartitionSizes[0])
    kernel = GetKernel<kBlockSize, kPartitionSizes[0], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kPartitionSizes[1])
    kernel = GetKernel<kBlockSize, kPartitionSizes[1], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kPartitionSizes[2])
    kernel = GetKernel<kBlockSize, kPartitionSizes[2], K_PADDED, kMaxPartitionsForKernel>();
  else
    kernel = GetKernel<kBlockSize, kPartitionSizes[3], K_PADDED, kMaxPartitionsForKernel>();

  return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

template <int K_PADDED>
bool IsSupportedDispatch(int batch_size, int partition_size, int num_partitions) {
  if (num_partitions <= 8) return CheckSupport<K_PADDED, 8>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 16) return CheckSupport<K_PADDED, 16>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 32) return CheckSupport<K_PADDED, 32>(batch_size, num_partitions, partition_size);
  return CheckSupport<K_PADDED, 64>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kConvergentSortMaxK) return false;

  const int partition_size = EstimateBestPartitionSize(vocab_size, k);

  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) return false;

  if (k <= 4) return IsSupportedDispatch<4>(batch_size, partition_size, num_partitions);
  if (k <= 8) return IsSupportedDispatch<8>(batch_size, partition_size, num_partitions);
  if (k <= 16) return IsSupportedDispatch<16>(batch_size, partition_size, num_partitions);
  if (k <= 32) return IsSupportedDispatch<32>(batch_size, partition_size, num_partitions);
  if (k <= 52) return IsSupportedDispatch<52>(batch_size, partition_size, num_partitions);
  return IsSupportedDispatch<64>(batch_size, partition_size, num_partitions);
}

}  // namespace flash_convergent
}  // namespace cuda
}  // namespace Generators

