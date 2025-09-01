// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_helper.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>

namespace Generators {
namespace cuda {
namespace bitonic_v7 { // coalesced access
static const char* kAlgoDescription = "Bitonic v7 (same as v4)";

inline void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  bitonic_v4::RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, num_partitions, sort_size);
}

} // namespace bitonic_v7
}  // namespace cuda
}  // namespace Generators