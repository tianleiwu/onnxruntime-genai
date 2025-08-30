// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_sampling.h"

namespace Generators {
namespace cuda {
namespace v0 {

// Declares the v0 baseline function for use in the test files.
void RunTopKViaMapReduceBitonicSort_v0(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size);

} // namespace v0
} // namespace cuda
} // namespace Generators
