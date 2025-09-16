// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>  // For FLT_MAX
#include <math_constants.h> // For CUDART_INF_F

namespace Generators {
namespace cuda {
namespace bitonic_sort {

// This helper always evaluates to true, but encodes the value into the type
template<int...> struct debug_always_true : std::true_type {};

#ifdef DEBUG_INSTANTIATION_ENABLE
  #define DEBUG_INSTANTIATION(kBlockSize, SortSize);                                 \
    static_assert(debug_always_true<kBlockSize, SortSize>::value,                   \
      "SharedMemBitonicSort slow path instantiated with kBlockSize and SortSize")
#else
  #define DEBUG_INSTANTIATION(kBlockSize, SortSize); do {} while (0)
#endif

/*
// Generic implementation for bitonic sort in shared memory.
// operating on separate score and index arrays (Struct of Arrays layout).
// IMPORTANT NOTE: This implementation contains a latent bug that manifests as a race
// condition when the number of threads (`kBlockSize`) is exactly equal to the number
// of elements being sorted (`SortSize`). This function should only be called when
// that condition is not met. The static_assert below enforces this.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_SoA(float* smem_scores, int* smem_indices) {
  // Enforce the constraint that kBlockSize must not be equal to SortSize to avoid a latent bug.
  static_assert(kBlockSize != SortSize, "SharedMemBitonicSort_SoA has a bug when kBlockSize == SortSize");
  // Stage 1: Build bitonic sequences
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          float score_i = smem_scores[i];
          float score_ixj = smem_scores[ixj];
          int index_i = smem_indices[i];
          int index_ixj = smem_indices[ixj];

          bool is_greater = (score_i > score_ixj) || (score_i == score_ixj && index_i < index_ixj);
          if (is_greater != ascending) {
            smem_scores[i] = score_ixj;
            smem_scores[ixj] = score_i;
            smem_indices[i] = index_ixj;
            smem_indices[ixj] = index_i;
          }
        }
      }
      __syncthreads();
    }
  }

  // Stage 2: Final merge to sort descending
  for (int j = SortSize >> 1; j > 0; j >>= 1) {
    for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
      int ixj = i ^ j;
      if (ixj > i) {
        float score_i = smem_scores[i];
        float score_ixj = smem_scores[ixj];
        int index_i = smem_indices[i];
        int index_ixj = smem_indices[ixj];

        if ((score_i < score_ixj) || (score_i == score_ixj && index_i > index_ixj)) {
          smem_scores[i] = score_ixj;
          smem_scores[ixj] = score_i;
          smem_indices[i] = index_ixj;
          smem_indices[ixj] = index_i;
        }
      }
    }
    __syncthreads();
  }
}


// Optimized implementation for finding top K elements in shared memory.
// Uses bitonic sort for small K, and heap-based approach for larger arrays.
template <int kBlockSize, int SortSize, int K>
__device__ void SharedMemBitonicTopK_SoA(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && K > 0 && K <= SortSize,
                "Invalid SortSize or K parameters");
  static_assert(kBlockSize > 0, "kBlockSize must be positive");

  const int tid = threadIdx.x;

  // Helper lambda for comparing and swapping elements with tie-breaking
  auto compareAndSwap = [&](int i, int j, bool ascending) {
    bool should_swap;
    if (smem_scores[i] != smem_scores[j]) {
      should_swap = (smem_scores[i] > smem_scores[j]) == ascending;
    } else {
      should_swap = (smem_indices[i] > smem_indices[j]) == ascending;
    }

    if (should_swap) {
      float temp_score = smem_scores[i];
      smem_scores[i] = smem_scores[j];
      smem_scores[j] = temp_score;

      int temp_index = smem_indices[i];
      smem_indices[i] = smem_indices[j];
      smem_indices[j] = temp_index;
    }
  };

  // Optimized partial sorting approach

  // Phase 1: Each thread finds local top elements
  const int elements_per_thread = (SortSize + kBlockSize - 1) / kBlockSize;
  const int start_idx = tid * elements_per_thread;
  const int end_idx = min(start_idx + elements_per_thread, SortSize);

  // Local partial sort - only need to identify top candidates
  for (int i = start_idx; i < end_idx - 1; ++i) {
    for (int j = i + 1; j < end_idx; ++j) {
      compareAndSwap(i, j, false);  // Descending order
    }
  }

  __syncthreads();

  // Phase 2: Use bitonic network to merge and find top K
  int current_k = min(K, SortSize);

  // Gradually reduce the working set size
  for (int working_size = 1; working_size < current_k; working_size <<= 1) {
    int next_size = min(working_size << 1, current_k);

    // Bitonic merge for the working portion
    for (int stride = working_size; stride > 0; stride >>= 1) {
      if (tid < (next_size + 1) / 2) {
        int partner = tid ^ stride;
        if (partner < next_size && partner != tid) {
          bool ascending = ((tid / (working_size << 1)) % 2) != 0;
          compareAndSwap(tid, partner, ascending);
        }
      }
      __syncthreads();
    }
  }

  // Final cleanup - ensure top K are in descending order
  for (int stride = (current_k >> 1); stride > 0; stride >>= 1) {
    if (tid < current_k && tid < stride) {
      int partner = tid + stride;
      if (partner < current_k) {
        compareAndSwap(tid, partner, false);
      }
    }
    __syncthreads();
  }
}

// Full bitonic sort implementation (for larger K values)
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Large(float* smem_scores, int* smem_indices) {
  DEBUG_INSTANTIATION(kBlockSize, SortSize);
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize > 0 && (kBlockSize & (kBlockSize - 1)) == 0,
                "kBlockSize must be a power of 2");
  static_assert(SortSize >= kBlockSize);

  const int tid = threadIdx.x;
  constexpr int elements_per_thread = SortSize / kBlockSize;

  auto compareAndSwap = [&](int i, int j, bool ascending) {
    bool should_swap;
    if (smem_scores[i] != smem_scores[j]) {
      should_swap = (smem_scores[i] > smem_scores[j]) == ascending;
    } else {
      should_swap = (smem_indices[i] > smem_indices[j]) == ascending;
    }

    if (should_swap) {
      float temp_score = smem_scores[i];
      smem_scores[i] = smem_scores[j];
      smem_scores[j] = temp_score;

      int temp_index = smem_indices[i];
      smem_indices[i] = smem_indices[j];
      smem_indices[j] = temp_index;
    }
  };

  // Phase 1: Sort local elements within each thread (descending order)
  if constexpr (elements_per_thread > 1) {
    for (int t = 0; t < elements_per_thread; ++t) {
      int base_idx = tid * elements_per_thread;
      for (int i = base_idx + 1; i < base_idx + elements_per_thread; ++i) {
        for (int j = i; j > base_idx; --j) {
          bool should_swap;
          if (smem_scores[j - 1] != smem_scores[j]) {
            should_swap = smem_scores[j - 1] < smem_scores[j];
          } else {
            should_swap = smem_indices[j - 1] > smem_indices[j];
          }

          if (!should_swap) break;
          compareAndSwap(j - 1, j, false);
        }
      }
    }
    __syncthreads();
  }

  // Phase 2: Bitonic merge phases
  for (int size = 2 * elements_per_thread; size <= SortSize; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      if constexpr (elements_per_thread > 1) {
        for (int t = 0; t < elements_per_thread; ++t) {
          int idx = tid * elements_per_thread + t;
          int partner = idx ^ stride;

          if (partner > idx && partner < SortSize) {
            bool ascending = ((idx / size) % 2) != 0;
            compareAndSwap(idx, partner, ascending);
          }
        }
      } else {  // one element per thread
        int partner = tid ^ stride;
        if (partner > tid) {
          // Determine sort direction based on bitonic sequence pattern
          bool ascending = ((tid / size) % 2) != 0;
          compareAndSwap(tid, partner, ascending);
        }
      }

      __syncthreads();
    }
  }

  // Phase 3: Final merge to create fully sorted sequence in descending order
  for (int stride = SortSize >> 1; stride > 0; stride >>= 1) {
    if constexpr (elements_per_thread > 1) {
      for (int t = 0; t < elements_per_thread; ++t) {
        int idx = tid * elements_per_thread + t;
        int partner = idx ^ stride;
        if (partner > idx && partner < SortSize) {
          compareAndSwap(idx, partner, false);
        }
      }
    } else { // one element per thread
      int partner = tid ^ stride;
      if (partner > tid) {
        compareAndSwap(tid, partner, false);  // Always descending
      }
    }
    __syncthreads();
  }
}
*/

/**
 * @brief Performs an in-place bitonic sort on data in shared memory.
 *
 * This function sorts an array of scores (`smem_scores`) in descending order
 * and simultaneously permutes an array of indices (`smem_indices`) to
 * maintain the original score-index correspondence. The sort is performed
 * entirely within shared memory and is synchronized to prevent race
 * conditions. When scores are equal, the element with the smaller original
 * index is placed first (tie-breaking).
 *
 * @tparam kBlockSize The number of threads in the CUDA thread block. This must be
 * greater than or equal to `SortSize`.
 * @tparam SortSize The number of elements to sort. This MUST be a power of two.
 * @param smem_scores A pointer to the shared memory array of floats (scores) to be sorted.
 * @param smem_indices A pointer to the shared memory array of ints (indices) to be permuted.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Small(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize >= SortSize);       
  DEBUG_INSTANTIATION(kBlockSize, SortSize);
    // This implementation uses one thread per element for the sort.
    const int ix = threadIdx.x;

    // The bitonic sort network is constructed in stages.
    // Stage 'k' creates sorted sequences of length 'k'.
    for (int k = 2; k <= SortSize; k <<= 1) {

        // Within each stage, we perform merge steps with decreasing comparison distances.
        // The comparison distance 'j' determines which elements are compared.
        for (int j = k >> 1; j > 0; j >>= 1) {

            // All threads must synchronize after each comparison step. This ensures that
            // the data from the previous step is correctly written to shared memory
            // before the next step begins. Failing to sync here would cause a race
            // condition, as threads could read stale data from an incomplete prior step.
            __syncthreads();

            // We only need SortSize threads to perform the sort. Any extra threads
            // in the block do not participate in the swap logic.
            if (ix < SortSize) {
                // Determine the other element in the pair for comparison.
                // The XOR indexing scheme is an efficient and elegant way to pair up
                // all necessary elements at each step of the network.
                int paired_ix = ix ^ j;

                // To prevent each pair from being processed twice (which would be a race condition)
                // and to define a clear ordering, we enforce that only the thread with the
                // lower index in the pair performs the comparison and potential swap.
                if (paired_ix > ix) {

                    // Determine the sorting direction (ascending or descending).
                    // The direction depends on which sub-sequence of size 'k' the thread is in.
                    // The expression `(ix & k) == 0` partitions the elements into blocks
                    // that are sorted in alternating directions. This is essential for building
                    // the bitonic sequences that are correctly merged in the next stage.
                    bool ascending = ((ix & k) == 0);

                    // Determine if a swap is needed for a descending sort.
                    bool should_swap = smem_scores[ix] < smem_scores[paired_ix];

                    // Handle tie-breaking: if scores are equal, the element with the
                    // smaller original index should be considered "greater".
                    if (smem_scores[ix] == smem_scores[paired_ix] && smem_indices[ix] > smem_indices[paired_ix]) {
                        should_swap = true;
                    }

                    // Swap if the order is incorrect for the current sorting direction.
                    if (should_swap == ascending) {
                        // Swap score
                        float temp_score = smem_scores[ix];
                        smem_scores[ix] = smem_scores[paired_ix];
                        smem_scores[paired_ix] = temp_score;

                        // Swap index
                        int temp_index = smem_indices[ix];
                        smem_indices[ix] = smem_indices[paired_ix];
                        smem_indices[paired_ix] = temp_index;
                    }
                }
            }
        }
    }

    // A final synchronization is good practice to ensure that all shared memory writes
    // from the sort are complete before the calling kernel proceeds to use the sorted data.
    __syncthreads();
}

// Generic implementation for bitonic sort in shared memory.
// Operating on separate score and index arrays (SoA).
// - SortSize must be a power of two.
// - All threads in the block must call this function.
// - Result: sorted in-place by score descending, tie-breaker index ascending.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Big(float* smem_scores, int* smem_indices) {
  static_assert(SortSize >= 2, "SortSize must be >= 2");
  static_assert((SortSize & (SortSize - 1)) == 0, "SortSize must be power of two");
 DEBUG_INSTANTIATION(kBlockSize, SortSize);
  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  // Make sure any prior shared-memory writes are visible.
  __syncthreads();

  // Outer bitonic build/merge loops (k = block size of current bitonic sequence).
  for (int k = 2; k <= N; k <<= 1) {
    // Inner merge strides.
    for (int j = k >> 1; j > 0; j >>= 1) {
      // Each thread processes multiple indices: i = tid, tid + kBlockSize, ...
      for (int i = tid; i < N; i += kBlockSize) {
        const int ixj = i ^ j;  // pair index
        if (ixj > i) {          // ensure only one thread of the pair does the swap
          // load both elements (single read each)
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          bool do_swap = false;

          // We choose final global ordering = descending by score.
          // Standard bitonic comparator: direction flips by (i & k).
          if ((i & k) == 0) {
            // this pair should be in descending order: put larger score at i
            if (a_i < a_j || (a_i == a_j && idx_i > idx_j)) do_swap = true;
          } else {
            // opposite direction for this sub-sequence
            if (a_i > a_j || (a_i == a_j && idx_i < idx_j)) do_swap = true;
          }

          if (do_swap) {
            // single writer for this pair — safe
            smem_scores[i] = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i] = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }  // i loop

      // synchronize after completing all pair operations for this stride
      __syncthreads();
    }  // j loop
  }  // k loop

  // ensure final sorted data is visible to callers
  __syncthreads();
}

constexpr int NextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Generic implementation for bitonic sort in shared memory (SoA).
// Supports arbitrary SortSize by padding to next power of two.
// - All threads in the block must call this function.
// - Sorted in-place: descending by score, tie-breaker ascending index.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Pad(float* smem_scores, int* smem_indices) {
DEBUG_INSTANTIATION(kBlockSize, SortSize);
  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  // compute next power of two
  constexpr int Npad = NextPowerOfTwo(N);

  // If SortSize < Npad, fill sentinels.
  for (int i = tid; i < Npad; i += kBlockSize) {
    if (i >= N) {
      smem_scores[i] = -FLT_MAX;  // sentinel (very small value for descending sort)
      smem_indices[i] = INT_MAX;  // tie-breaker sentinel
    }
  }
  __syncthreads();

  // Bitonic network
  for (int k = 2; k <= Npad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < Npad; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          bool do_swap = false;
          if ((i & k) == 0) {
            // descending region
            if (a_i < a_j || (a_i == a_j && idx_i > idx_j)) do_swap = true;
          } else {
            // ascending region
            if (a_i > a_j || (a_i == a_j && idx_i < idx_j)) do_swap = true;
          }

          if (do_swap) {
            smem_scores[i]   = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i]   = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }

  // After sort, valid results are in [0..N-1].  
  // Elements [N..Npad-1] contain sentinels and can be ignored.
  __syncthreads();
}

template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_SoA(float* smem_scores, int* smem_indices) {
  if constexpr ((SortSize & (SortSize - 1)) != 0) {
    SharedMemBitonicSort_Pad<kBlockSize, SortSize>(smem_scores, smem_indices);
  } else {
    if constexpr (kBlockSize >= SortSize) {
      SharedMemBitonicSort_Small<kBlockSize, SortSize>(smem_scores, smem_indices);
    } else {
      SharedMemBitonicSort_Big<kBlockSize, SortSize>(smem_scores, smem_indices);
    }
  }
}

template <int kBlockSize, int SortSize, int K>
__device__ void SharedMemBitonicTopK(float* smem_scores, int* smem_indices) {
  if constexpr ((SortSize & (SortSize - 1)) != 0) {
    SharedMemBitonicSort_Pad<kBlockSize, SortSize>(smem_scores, smem_indices);
  } else if constexpr (kBlockSize >= SortSize) {
    SharedMemBitonicSort_Small<kBlockSize, SortSize>(smem_scores, smem_indices);
  } /* else if constexpr (K <= SortSize / 4) {
    // For small K relative to SortSize, use partial bitonic sort
    SharedMemBitonicTopK_SoA<kBlockSize, SortSize, K>(smem_scores, smem_indices);
  }*/ else {
    // For larger K, use optimized full bitonic sort
    // SharedMemBitonicSort_Large<kBlockSize, SortSize>(smem_scores, smem_indices);
    // Below is less optimized version:
    SharedMemBitonicSort_Big<kBlockSize, SortSize>(smem_scores, smem_indices);
  }
}

template <int N>
__device__ void RegisterBitonicSort(float scores[N], int indices[N]) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          bool is_greater = (scores[i] > scores[ixj]) || (scores[i] == scores[ixj] && indices[i] < indices[ixj]);
          if (is_greater != ascending) {
            float temp_s = scores[i];
            scores[i] = scores[ixj];
            scores[ixj] = temp_s;
            int temp_i = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = temp_i;
          }
        }
      }
    }
  }
  for (int j = N >> 1; j > 0; j >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int ixj = i ^ j;
      if (ixj > i) {
        if ((scores[i] < scores[ixj]) || (scores[i] == scores[ixj] && indices[i] > indices[ixj])) {
          float temp_s = scores[i];
          scores[i] = scores[ixj];
          scores[ixj] = temp_s;
          int temp_i = indices[i];
          indices[i] = indices[ixj];
          indices[ixj] = temp_i;
        }
      }
    }
  }
}

template <int kBlockSize, int K, int PartitionsPerBlock>
__global__ void BlockReduceTopK_SoA(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                    float* __restrict__ scores_out, int* __restrict__ indices_out, int num_partitions_in) {
  constexpr int SortSize = K * PartitionsPerBlock;
  __shared__ float smem_scores[SortSize];
  __shared__ int smem_indices[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const int in_base_offset = batch_idx * num_partitions_in * K;
  const int out_base_offset = (batch_idx * gridDim.x + blockIdx.x) * K;

  // Load data from global memory into shared memory using an SoA layout
  for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
    if (i < K * num_partitions_to_process) {
      int partition_idx = i / K;
      int element_idx = i % K;
      int global_offset = in_base_offset + (block_start_partition + partition_idx) * K + element_idx;
      smem_scores[i] = scores_in[global_offset];
      smem_indices[i] = indices_in[global_offset];
    } else {
      smem_scores[i] = -FLT_MAX;
      smem_indices[i] = INT_MAX;
    }
  }
  __syncthreads();

  // Perform the sort on the SoA data in shared memory.
  SharedMemBitonicSort_SoA<kBlockSize, SortSize>(smem_scores, smem_indices);

  // Write the top K results back to global memory
  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_indices[threadIdx.x];
    scores_out[out_base_offset + threadIdx.x] = smem_scores[threadIdx.x];
  }
}

}  // namespace bitonic_sort
}  // namespace cuda
}  // namespace Generators
