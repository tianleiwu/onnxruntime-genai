// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace Generators {
namespace cuda {

struct TopkData;

// This function orchestrates the two main parts of beam search Top-K:
// 1. Vocabulary Top-K: Uses the high-performance `RunTopK` to find the top `2 * num_beams` candidates from the vocabulary for each beam.
// 2. Beam-wise Top-K: Reduces the `batch_size * num_beams * (2 * num_beams)` candidates down to the final `batch_size * num_beams` candidates for the next step.
void BeamSearchTopK(
    TopkData* topk_data,
    const float* input,
    int batch_size,
    int num_beams,
    int vocab_size,
    float* tmp_scores,         // Temporary buffer for vocab Top-K scores
    int32_t* tmp_tokens,       // Temporary buffer for vocab Top-K tokens
    float* output_values,      // Final output scores
    int32_t* output_tokens,    // Final output tokens
    int32_t* output_indices,   // Final output beam indices
    cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators
