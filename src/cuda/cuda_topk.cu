// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_bitonic_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"

namespace Generators {
namespace cuda {

__global__ void FillRandom(float* array, curandState* states, int n, int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int batch_idx = (static_cast<long long>(i) * batch_size) / n;
    array[i] = curand_uniform(&states[batch_idx]);
  }
}

void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size) {
  FillRandom<<<(total_size + 255) / 256, 256, 0, stream>>>(data, batch_state, total_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

TopKAlgorithm ChooseTopkAlgorithm(int vocab_size, int batch_size, int k, int &partition_size) {
  if (k > 64) {
    return TopKAlgorithm::FULL_SORT;
  }

  if (k <= 4) {
    return TopKAlgorithm::SELECTION_SORT;
  }     
  
  if (vocab_size >= 147456) {
    partition_size = (vocab_size > 256 * 1024) ? 8192 : 4096;
    return TopKAlgorithm::HYBRID_SORT;
  }
  else {
    if (k <= 8)
      return TopKAlgorithm::SELECTION_SORT;

    if (vocab_size >= 65536 || batch_size >= 4 && vocab_size > 49152) {
      partition_size = 2048;
      return TopKAlgorithm::HYBRID_SORT;
    }
  }

  partition_size = 1024;
  if (vocab_size <= partition_size) {
    return TopKAlgorithm::SELECTION_SORT;
  }

  return TopKAlgorithm::HYBRID_SORT;
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  int partition_size = 0;

  auto algo = ChooseTopkAlgorithm(vocab_size, batch_size, k, partition_size);
  switch (algo) {
    case TopKAlgorithm::SELECTION_SORT:
      RunTopKViaSelectionSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;      
    case TopKAlgorithm::BITONIC_SORT:
      RunTopKViaBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, partition_size);
      break;
    case TopKAlgorithm::HYBRID_SORT:
      RunTopKViaHybridSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, partition_size);
      break;      
    default:
      RunTopKViaFullSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
  }
}

}  // namespace cuda
}  // namespace Generators

