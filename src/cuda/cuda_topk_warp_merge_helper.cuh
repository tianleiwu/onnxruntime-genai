// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace warp_merge {

/**
 * @brief Performs an in-place, warp-wide bitonic sort on data held in registers.
 * Sorts `warpSize` elements distributed across the threads of a single warp.
 * The result is sorted in descending order by score.
 */
__device__ inline void WarpBitonicSort(float& score, int& index) {
  const int lane_id = threadIdx.x % warpSize;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= warpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int paired_lane = lane_id ^ j;
      float paired_score = __shfl_sync(0xFFFFFFFF, score, paired_lane);
      int paired_index = __shfl_sync(0xFFFFFFFF, index, paired_lane);

      if (paired_lane > lane_id) {
        // A standard bitonic network sorts ascending with `(lane_id & k) == 0`.
        // The swap condition is inverted to produce a descending sort.
        bool ascending = ((lane_id & k) == 0);

#ifdef STABLE_TOPK
        // For stable sort, include tie-breaking logic (smaller index wins).
        bool is_greater = (score > paired_score) || (score == paired_score && index < paired_index);
#else
        // For unstable sort, no tie-breaking is needed for performance.
        bool is_greater = score > paired_score;
#endif

        if (is_greater != ascending) {
          score = paired_score;
          index = paired_index;
        }
      }
    }
  }
}

}  // namespace warp_merge
}  // namespace cuda
}  // namespace Generators
