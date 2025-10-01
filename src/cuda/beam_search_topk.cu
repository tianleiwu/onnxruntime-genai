// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "beam_search_topk.h"
#include "beam_search_impl_cuda.h"
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

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
    cudaStream_t stream) {
  // --- 1. Vocabulary Top-K ---
  // Find the top `2 * num_beams` candidates from the entire vocabulary for each beam.
  const int k_vocab_top_k = 2 * num_beams;
  RunTopK(topk_data, stream, input, vocab_size, batch_size * num_beams, k_vocab_top_k);

  topk_data->GetCompactOutput(tmp_scores, tmp_tokens, batch_size * num_beams, k_vocab_top_k, stream);

  // --- 2. Beam-Wise Top-K ---
  // From the `num_beams * (2 * num_beams)` candidates for each batch item,
  // select the final `num_beams` best candidates.
  LaunchBatchTopKKernel(tmp_scores,
                        tmp_tokens,
                        k_vocab_top_k, /* stride */
                        output_indices,
                        output_tokens,
                        output_values,
                        batch_size,
                        num_beams,
                        stream);
}

}  // namespace cuda
}  // namespace Generators

