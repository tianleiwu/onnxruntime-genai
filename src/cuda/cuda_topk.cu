// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "cuda_topk_bitonic_sort_v0.cuh"
// #include "cuda_topk_bitonic_sort_v3.cuh"
// #include "cuda_topk_bitonic_sort_v4.cuh"
// #include "cuda_topk_bitonic_sort_v6.cuh"
// #include "cuda_topk_bitonic_sort_v7.cuh"
// #include "cuda_topk_bitonic_sort_v8.cuh"
// #include "cuda_topk_bitonic_sort_v9.cuh"
#include "cuda_topk_bitonic_sort_v10.cuh"
// #include "cuda_topk_bitonic_sort_v11.cuh"
// #include "cuda_topk_bitonic_sort_v12.cuh"
// #include "cuda_topk_bitonic_sort_v13.cuh"
// #include "cuda_topk_bitonic_sort_v17.cuh"
#include "cuda_topk_bitonic_sort_v19.cuh"
// #include "cuda_topk_bitonic_sort_v21.cuh"
// #include "cuda_topk_bitonic_sort_v22.cuh"
#include "cuda_topk_bitonic_sort_v23.cuh"
#include "cuda_topk_bitonic_sort_v24.cuh"

namespace Generators {
namespace cuda {

const char* GetBitonicBaselineDescription() {
  return bitonic_v24::kAlgoDescription;
}
void RunTopKViaHybridSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  bitonic_v24::RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, num_partitions, sort_size);
}

const char* GetBitonicTreatmentDescription(){
  return bitonic_v19::kAlgoDescription;
}
void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  bitonic_v19::RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, num_partitions, sort_size);
}

int GetBestSortSize(int vocab_size, int batch_size, int k) {
      if (k <= 4)
          return 0;

      if (vocal_size > 256 * 1024)
        return 8192;

      if (vocab_size >= 147456)
          return 4096;
      else if (vocab_size < 49152) {
        if (k <= 8)
          return 0;
      }
      else {
        if (k < 8)
          return 0;
        if (vocab_size >= 65536 || batch_size >= 4)
            return 2048;
      }
          
      return 1024;
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  TopKConfig chosen_config;
  if (k <= 8) {
    chosen_config.algorithm = TopKAlgorithm::SELECTION_SORT;
  } else if (k <= 64) {
    chosen_config = BenchmarkAndGetBestAlgorithm(data, stream, vocab_size, batch_size, k);
  }

  switch (chosen_config.algorithm) {
    case TopKAlgorithm::SELECTION_SORT:
      RunTopKViaSelectionSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
    case TopKAlgorithm::BITONIC_SORT:
      RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, chosen_config.num_partitions, chosen_config.sort_size);
      break;
    default:
      RunTopKViaFullSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
  }
}

}  // namespace cuda
}  // namespace Generators

