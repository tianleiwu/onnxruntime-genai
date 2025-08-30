// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_topk_helper.h"
#include "cuda_sampling.h"

namespace Generators {
namespace cuda {

// START of improved Top-K kernel (Selection Sort approach)
struct TopK_2 {
  int p = INT_MAX;
  float u = -FLT_MAX;

  __device__ __forceinline__ void insert(float elem, int elem_id) {
    if (elem > u || (elem == u && elem_id < p)) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -FLT_MAX;
    p = -1;
  }
};

__device__ __forceinline__ TopK_2 reduce_topk_op_2(TopK_2 const& a, TopK_2 const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p < b.p) ? a
                                                   : b;
}

template <int kBlockSize>
__global__ void GetTopKKernel(int* indices_out, float* scores_in, float* scores_out, int batch_size, int vocab_size, int k, float temperature) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_2 partial;

  float const MAX_T_VAL = FLT_MAX;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
    for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
      float elem = scores_in[elemId + batch * vocab_size];
      partial.insert(elem, elemId);
    }
    // reduce in thread block
    typedef cub::BlockReduce<TopK_2, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_2 top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2);

    if (tid == 0) {
      scores_out[ite + batch * k] = top_k_sequence.u / temperature;
      indices_out[ite + batch * k] = top_k_sequence.p;

      // set the max value to -MAX_T_VAL so that the value doesn't get picked again
      scores_in[batch * vocab_size + top_k_sequence.p] = -MAX_T_VAL;
    }

    // Attention: Add a block-level memory fence here.
    // This ensures that the write to global memory by thread 0 is visible
    // to all other threads in the block before the next iteration begins.
    // Without this, other threads might read the old (pre-modification)
    // score in the next iteration, leading to the same item being picked again.
    __threadfence_block();

    __syncthreads();
  }
}

// Launcher for the improved Top-K kernel.
void LaunchImprovedGetTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  dim3 grid(batch_size, 1, 1);
  // Use a larger block size for better hardware utilization, as in the improved file.
  dim3 block(1024, 1, 1);
  GetTopKKernel<1024><<<grid, block, 0, stream>>>(indices_out, scores_in, scores_out, batch_size, vocab_size, k, temperature);
  CUDA_CHECK(cudaGetLastError());
}

void RunTopKViaSelectionSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  // The output of the kernel will be the temperature-scaled scores. We'll store
  // these in another pre-allocated buffer, `scores_buffer`.
  float* scaled_scores = data->scores_buffer.get();

  // Attention: The kernel modifies the `scores_in` tensor in-place.
  // This might have unintended side effects on the original `scores_in` tensor if it is used elsewhere later.
  LaunchImprovedGetTopK(stream, scores_in, scaled_scores, indices_out, vocab_size, batch_size, k, temperature);

  // Finally, apply softmax to the scaled scores to get the final probabilities,
  // writing the result to the `scores_out` buffer.
  DispatchBlockwiseSoftmaxForward<false>(stream, scores_out, const_cast<const float*>(scaled_scores), k, k, k, batch_size);
}

}  // namespace cuda
}  // namespace Generators
