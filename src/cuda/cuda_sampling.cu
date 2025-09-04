// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <math.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>

#include "cuda_sampling.h"
#include "cuda_topk.h"
#include "smartptrs.h"
#include "span.h"

namespace Generators {
namespace cuda {

constexpr int kMaxThreads = 1024;
constexpr int kGPUWarpSize = 32;

__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size)
    return;

  curand_init(seed, index, 0, &states[index]);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream) : TopkData(batch_size, vocab_size, stream) {
  const size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  prefix_sums = CudaMallocArray<float>(vocab_batch_size);
  scores_adjusted = CudaMallocArray<float>(vocab_batch_size);
  prefix_sums_adjusted = CudaMallocArray<float>(vocab_batch_size);

  thresholds = CudaMallocArray<float>(batch_size);
  curand_states = CudaMallocArray<curandState>(batch_size);

  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Softmax Kernels and Launchers

template <typename T, typename AccumT>
struct MaxFloat {
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T, typename AccumT>
struct SumExpFloat {
  __device__ __forceinline__ SumExpFloat(AccumT v)
      : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + exp((AccumT)v - max_k);
  }

  const AccumT max_k;
};

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

// aligned vector generates vectorized load/store on CUDA
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <template <typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT IlpReduce(int shift, T* data, int size, const Reduction<T, AccumT>& r, AccumT defaultVal) {
  using LoadT = aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;
  // shift and do 1
  if (shift > 0) {
    data -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  if (size <= 0) return threadVal;
  int last = size % (ILP * blockDim.x);
  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }
  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);
  return threadVal;
}

template <template <typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT SoftmaxReduce(AccumT* smem, AccumT val, const Reduction<AccumT>& r, AccumT defaultVal) {
  // To avoid RaW races from chaining SoftmaxReduce calls together, we need a sync here
  __syncthreads();
  smem[threadIdx.x] = val;
  __syncthreads();
  AccumT warpVal = defaultVal;
  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < kGPUWarpSize) {
    int warps_per_block = blockDim.x / kGPUWarpSize;
    for (int i = 0; i < warps_per_block; ++i) {
      warpVal = r(warpVal, smem[i * kGPUWarpSize + threadIdx.x]);
    }
    smem[threadIdx.x] = warpVal;
  }
  __syncthreads();
  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < kGPUWarpSize; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }
  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

dim3 SoftmaxGetBlockSize(int ILP, uint64_t size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = min(size / ILP, static_cast<uint64_t>(kMaxThreads));
  // In the vectorized case we want to trade off allowing more of the buffers to be accessed
  // in a vectorized way against wanting a larger block size to get better utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = max(block_size, static_cast<uint64_t>(kGPUWarpSize));
  return dim3(static_cast<unsigned int>(block_size));
}

template <typename T, typename AccumT, typename OutT>
struct LogSoftmaxForwardEpilogue {
  __device__ __forceinline__ LogSoftmaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), logsum(log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>((AccumT)input - max_input - logsum);
  }

  const AccumT max_input;
  const AccumT logsum;
};

template <typename T, typename AccumT, typename OutT>
struct SoftmaxForwardEpilogue {
  __device__ __forceinline__ SoftmaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(exp((AccumT)input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResultsVectorized(int size,
                                                            const int shift,
                                                            scalar_t* input,
                                                            outscalar_t* output,
                                                            Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using LoadT = aligned_vector<scalar_t, ILP>;
  using StoreT = aligned_vector<outscalar_t, ILP>;
  int offset = threadIdx.x;
  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    input -= shift;
    output -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }
  if (size <= 0) return;
  const int last = size % (ILP * blockDim.x);
  scalar_t in_v[ILP];
  LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);
  outscalar_t out_v[ILP];
  StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<LoadT*>(input)[offset];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }
    reinterpret_cast<StoreT*>(output)[offset] = *out_value;
  }
  offset = size - last + threadIdx.x;
  // handle the tail
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResults(int classes,
                                                  scalar_t* input,
                                                  outscalar_t* output,
                                                  Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  // Main bulk of loop with ILP
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }
  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
          template <typename, typename, typename> class Epilogue>
__global__ void SoftmaxBlockForward(outscalar_t* output, scalar_t* input, int classes,
                                    int input_stride, int output_stride) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  const int input_align_bytes = ILP * sizeof(scalar_t);
  const int output_align_bytes = ILP * sizeof(outscalar_t);
  const int shift = ((uint64_t)input) % input_align_bytes / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % output_align_bytes / sizeof(outscalar_t);
  // find the max
  accscalar_t threadMax = IlpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = SoftmaxReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  // reduce all values
  accscalar_t threadExp = IlpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = SoftmaxReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));
  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(float);
  dim3 block = SoftmaxGetBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    SoftmaxBlockForward<ILP, float, float, float, LogSoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride);
  } else {
    SoftmaxBlockForward<ILP, float, float, float, SoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride);
  }
  CUDA_CHECK(cudaGetLastError());
}
template void DispatchBlockwiseSoftmaxForward<true>(cudaStream_t, float*, const float*, int, int, int, int);
template void DispatchBlockwiseSoftmaxForward<false>(cudaStream_t, float*, const float*, int, int, int, int);

// --- Sampling Kernels ---
template <int kBlockSize>
__global__ void PrefixSumKernel(float* scores, float* prefix_sums, int sample_range, int batch_size) {
  int batch = blockIdx.x;
  float prefix_sum = 0.0f;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = 0; i < sample_range; i += blockDim.x) {
    int global_index = threadIdx.x + i + batch * sample_range;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;
    float sum = score;
    BlockScan(temp_storage).InclusiveSum(sum, sum);
    prefix_sum += sum;
    __syncthreads();
    if (local_index < sample_range) {
      prefix_sums[global_index] = prefix_sum;
    }
  }
}

template <int kBlockSize>
__global__ void FilterOnTopP(float* scores, float* prefix_sums, float* scores_temp, float* actual_values, int sample_range, int batch_size, float p) {
  int batch = blockIdx.x;
  float prefix_sum = 0.0f;
  float saferNegative = std::numeric_limits<float>::lowest() / 1000.0f;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = 0; i < sample_range; i += blockDim.x) {
    int global_index = threadIdx.x + i + batch * sample_range;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;
    float sum = score;
    BlockScan(temp_storage).InclusiveSum(sum, sum);
    prefix_sum += sum;
    __syncthreads();
    if (local_index < sample_range) {
      scores_temp[global_index] = prefix_sum;
    }
    __syncthreads();
    if (local_index == 0) {
      prefix_sums[global_index] = actual_values[global_index];
    } else if (local_index < sample_range) {
      if (scores_temp[global_index - 1] < p) {
        prefix_sums[global_index] = actual_values[global_index];
      } else {
        prefix_sums[global_index] = saferNegative;
      }
    }
  }
}

__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size) {
    // For numerical stability, we use 0.9999999f not 1.0f to avoid zero probabilities.
    thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
  }
}

template <int kBlockSize>
__global__ void SampleKernel(float* prefix_sums, int* indices, int* index_out, int sample_range, int indices_stride, float* thresholds) {
  int batch = blockIdx.x;
  int index = threadIdx.x;

  __shared__ int first_index;
  if (threadIdx.x == 0) {
    first_index = sample_range - 1;
  }
  __syncthreads();

  for (; index < sample_range - 1; index += blockDim.x) {
    float sum = prefix_sums[batch * sample_range + index];
    // TOP P or K
    if (sum >= thresholds[batch]) {
      atomicMin(&first_index, index);
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_out[batch] = indices[batch * indices_stride + first_index];
  }
}

void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, float* scores, int* indices, int* index_out, int sample_range, int batch_size, int indices_stride, float p, int k, float temperature) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);

  // The `FilterOnTopP` kernel reads from `scores` (which contains probabilities) and writes the filtered results to `prefix_sums`.
  // Values that do not meet the Top-P criteria are set to a large negative number.
  FilterOnTopP<256><<<grid, block, 0, stream>>>(scores, data->prefix_sums.get(), data->scores_temp.get(), data->scores_buffer.get(), sample_range, batch_size, p);
  CUDA_CHECK(cudaGetLastError());

  // After Top-P filtering, the remaining probabilities must be re-normalized.
  DispatchBlockwiseSoftmaxForward<false>(stream, data->scores_adjusted.get(), const_cast<const float*>(data->prefix_sums.get()), k, indices_stride, k, batch_size);
  CUDA_CHECK(cudaGetLastError());

  PrefixSumKernel<256><<<grid, block, 0, stream>>>(data->scores_adjusted.get(), data->prefix_sums_adjusted.get(), sample_range, batch_size);
  CUDA_CHECK(cudaGetLastError());

  RandomThresholdKernel<<<int((batch_size + 127) / 128), 128, 0, stream>>>(data->curand_states.get(), data->thresholds.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());

  SampleKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), indices, index_out, sample_range, indices_stride, data->thresholds.get());
  CUDA_CHECK(cudaGetLastError());
}

// Main sampling entry point
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size, int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // Stage 1: Get Top K candidates.
  TopkData* topk_data = data;
  GetTopKSubset(topk_data, stream, scores_in, data->scores_sorted.get(), data->indices_sorted.get(), vocab_size, batch_size, k, temperature);

  // Stage 2: Sample from the top k candidates.
  int sample_range = k;
  int indices_stride = k;
  LaunchSampleKernel(data, stream, data->scores_sorted.get(), data->indices_sorted.get(), next_token_out, sample_range, batch_size, indices_stride, p, k, temperature);
}
}  // namespace cuda
}  // namespace Generators
