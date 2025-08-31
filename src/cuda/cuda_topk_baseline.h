// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_sampling.h"

namespace Generators {
namespace cuda {
namespace baseline {

void RunTopKViaBaseline(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature);

} // namespace baseline
} // namespace cuda
} // namespace Generators
