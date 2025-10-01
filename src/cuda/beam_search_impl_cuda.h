// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <limits>

namespace Generators {
namespace cuda {

// A struct to hold a single candidate for sorting.
// It includes the score, the token ID, and the index of the beam it came from.
template <typename T>
struct BeamCandidate {
  T score;
  int32_t token_id;
  int32_t beam_index;
};

// CUB-style comparison operator for sorting BeamCandidate structs in descending order.
template <typename T>
struct BeamCandidateComparator {
  __device__ __forceinline__ bool operator()(const BeamCandidate<T>& a, const BeamCandidate<T>& b) const {
    return a.score > b.score;
  }
};

// Kernel for the second stage of beam search Top-K.
// Each block processes one item in the batch. It gathers the Top-K candidates from all beams,
// sorts them, and writes out the global Top-K results for the next beam search step.
template <typename T, int kBlockSize, int kNumBeams, int kVocabTopK>
__global__ void BatchTopKKernel(
    const T* topk_scores,      // Input scores from vocabulary Top-K (strided)
    const int* topk_tokens,    // Input token IDs from vocabulary Top-K (strided)
    int stride,                // Stride to access the input arrays
    int32_t* next_indices,     // Output: The source beam index for each winning candidate
    int32_t* next_tokens,      // Output: The token ID for each winning candidate
    T* next_scores,            // Output: The score for each winning candidate
    int32_t k_final) {         // The final number of candidates to produce (should be num_beams)
  const int batch_id = blockIdx.x;

  constexpr int kTotalCandidates = kNumBeams * kVocabTopK;
  constexpr int kItemsPerThread = (kTotalCandidates + kBlockSize - 1) / kBlockSize;

  using BlockMergeSort = cub::BlockMergeSort<BeamCandidate<T>, kBlockSize, kItemsPerThread>;

  __shared__ union {
    typename BlockMergeSort::TempStorage temp_storage;
    BeamCandidate<T> candidates[kTotalCandidates];
  } smem;

  // Step 1: All threads cooperate to load candidates from global to shared memory.
  for (int i = threadIdx.x; i < kTotalCandidates; i += kBlockSize) {
    int beam_idx = i / kVocabTopK;
    int local_item_idx = i % kVocabTopK;
    int batch_beam_idx = batch_id * kNumBeams + beam_idx;
    size_t global_offset = static_cast<size_t>(batch_beam_idx) * stride + local_item_idx;

    smem.candidates[i].score = topk_scores[global_offset];
    smem.candidates[i].token_id = topk_tokens[global_offset];
    smem.candidates[i].beam_index = beam_idx;
  }
  __syncthreads();

  // Step 2: Load candidates from shared memory into registers.
  BeamCandidate<T> thread_candidates[kItemsPerThread];
  cub::LoadDirectBlocked(threadIdx.x, smem.candidates, thread_candidates);
  __syncthreads();

  // Step 3: Sort the candidates in registers.
  BlockMergeSort(smem.temp_storage).Sort(thread_candidates, BeamCandidateComparator<T>());
  __syncthreads();

  // Step 4: Store sorted candidates from registers back to shared memory.
  cub::StoreDirectBlocked(threadIdx.x, smem.candidates, thread_candidates);
  __syncthreads();

  // Step 5: Have the first k_final threads write out the top results.
  if (threadIdx.x < k_final) {
    size_t out_offset = static_cast<size_t>(batch_id) * k_final + threadIdx.x;
    next_scores[out_offset] = smem.candidates[threadIdx.x].score;
    next_tokens[out_offset] = smem.candidates[threadIdx.x].token_id;
    next_indices[out_offset] = smem.candidates[threadIdx.x].beam_index;
  }
}

// Host-side launcher for the BatchTopKKernel.
// It uses template metaprogramming to instantiate the correct kernel based on the beam size.
template <typename T>
void LaunchBatchTopKKernel(const T* topk_scores,
                           const int* topk_tokens,
                           int stride,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           T* next_scores,
                           int32_t batch_size,
                           int32_t num_beams,
                           cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int k_final = num_beams;

#define LAUNCH_KERNEL(BEAM_SIZE)                                                                                                    \
  BatchTopKKernel<T, kBlockSize, BEAM_SIZE, 2 * BEAM_SIZE><<<batch_size, kBlockSize, 0, stream>>>(topk_scores,                        \
                                                                                                 topk_tokens,                         \
                                                                                                 stride,                              \
                                                                                                 next_indices,                        \
                                                                                                 next_tokens,                         \
                                                                                                 next_scores,                         \
                                                                                                 k_final);

  // This dispatch is needed because template arguments must be compile-time constants.
  if (num_beams <= 4) {
    LAUNCH_KERNEL(4);
  } else if (num_beams <= 8) {
    LAUNCH_KERNEL(8);
  } else if (num_beams <= 16) {
    LAUNCH_KERNEL(16);
  } else {
    // Add more cases if you support more beams, or assert.
    assert(false);
  }
#undef LAUNCH_KERNEL
}

}  // namespace cuda
}  // namespace Generators

