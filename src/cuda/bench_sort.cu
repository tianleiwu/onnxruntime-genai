// Usage:
// /usr/local/cuda/bin/nvcc -o bench_sort bench_sort.cu -O3 -arch=sm_89 --extended-lambda -std=c++20 --expt-relaxed-constexpr
// ./bench_sort
#if 0
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <numeric>

// --- User-Provided Headers ---
// Note: These must be in the same directory or in the include path.
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_stable_sort_helper.cuh"
#include "cuda_topk_common.cuh"

// A simple greater-than comparator for descending sort
struct DescendingOp {
    template <typename T>
    __device__ __host__ bool operator()(const T& a, const T& b) const {
        return a > b;
    }
};

// ====================================================================================
// KERNELS FOR BENCHMARKING
// ====================================================================================

// --- 1. Warp Bitonic Sort ---
__global__ void warpBitonicSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int k) {
    if (threadIdx.x >= 32) return;
    float my_score = scores_in[threadIdx.x];
    int my_index = indices_in[threadIdx.x];
    Generators::cuda::bitonic_sort::WarpBitonicSort(my_score, my_index);
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = my_score;
        indices_out[threadIdx.x] = my_index;
    }
}

// --- 2. CUB Warp Merge Sort ---
template <int SORT_SIZE>
__global__ void cubWarpMergeSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int k) {
    constexpr int SORT_SIZE_PO2 = Generators::cuda::topk_common::NextPowerOfTwo(SORT_SIZE);
    union SharedStorage {
        struct {
            float scores[SORT_SIZE_PO2];
            int indices[SORT_SIZE_PO2];
        } sort_data;
        typename cub::WarpMergeSort<float, (SORT_SIZE_PO2 + 31) / 32, 32, int>::TempStorage cub_storage;
    };
    __shared__ SharedStorage smem;

    for (int i = threadIdx.x; i < SORT_SIZE; i += blockDim.x) {
        smem.sort_data.scores[i] = scores_in[i];
        smem.sort_data.indices[i] = indices_in[i];
    }
    for (int i = threadIdx.x + SORT_SIZE; i < SORT_SIZE_PO2; i += blockDim.x) {
         smem.sort_data.scores[i] = -FLT_MAX;
         smem.sort_data.indices[i] = INT_MAX;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
       Generators::cuda::bitonic_sort::WarpMergeSort<SORT_SIZE_PO2>(
           smem.sort_data.scores, smem.sort_data.indices, &smem.cub_storage, SORT_SIZE);
    }
    __syncthreads();

    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = smem.sort_data.scores[threadIdx.x];
        indices_out[threadIdx.x] = smem.sort_data.indices[threadIdx.x];
    }
}

// --- 3. Shared Memory Bitonic Sort ---
template <int BLOCK_SIZE, int SORT_SIZE>
__global__ void sharedMemBitonicSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int k) {
    constexpr int SORT_SIZE_PO2 = Generators::cuda::topk_common::NextPowerOfTwo(SORT_SIZE);
    __shared__ float smem_scores[SORT_SIZE_PO2];
    __shared__ int smem_indices[SORT_SIZE_PO2];

    for (int i = threadIdx.x; i < SORT_SIZE; i += blockDim.x) {
        smem_scores[i] = scores_in[i];
        smem_indices[i] = indices_in[i];
    }
    for (int i = threadIdx.x + SORT_SIZE; i < SORT_SIZE_PO2; i += blockDim.x) {
        smem_scores[i] = -FLT_MAX;
        smem_indices[i] = INT_MAX;
    }
    __syncthreads();

    Generators::cuda::bitonic_sort::SharedMemBitonicSort<BLOCK_SIZE, SORT_SIZE_PO2>(smem_scores, smem_indices);
    __syncthreads();

    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = smem_scores[threadIdx.x];
        indices_out[threadIdx.x] = smem_indices[threadIdx.x];
    }
}

// --- 4. CUB Block Merge Sort ---
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockMergeSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int n, int k) {
    using BlockMergeSort = cub::BlockMergeSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
    __shared__ typename BlockMergeSort::TempStorage temp_storage;
    float thread_scores[ITEMS_PER_THREAD];
    int thread_indices[ITEMS_PER_THREAD];
    cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
    cub::LoadDirectBlocked(threadIdx.x, indices_in, thread_indices, n, INT_MAX);
    BlockMergeSort(temp_storage).Sort(thread_scores, thread_indices, DescendingOp());
    cub::StoreDirectBlocked(threadIdx.x, scores_out, thread_scores, k);
    cub::StoreDirectBlocked(threadIdx.x, indices_out, thread_indices, k);
}

// --- 5. CUB Block Radix Sort ---
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockRadixSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int n, int k) {
    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    float thread_scores[ITEMS_PER_THREAD];
    int thread_indices[ITEMS_PER_THREAD];
    cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
    cub::LoadDirectBlocked(threadIdx.x, indices_in, thread_indices, n, INT_MAX);
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_scores, thread_indices);
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = thread_scores[0];
        indices_out[threadIdx.x] = thread_indices[0];
    }
}


// ====================================================================================
// VERIFICATION AND MAIN BENCHMARK LOGIC
// ====================================================================================

struct ScoreIndex {
    float score;
    int index;
    bool operator>(const ScoreIndex& other) const {
        if (score != other.score) return score > other.score;
        return index < other.index;
    }
};

void verifyTopK(const std::vector<float>& h_scores_out, const std::vector<int>& h_indices_out,
                const std::vector<float>& h_scores_in, const std::vector<int>& h_indices_in,
                int k, const std::string& algo_name) {
    std::vector<ScoreIndex> reference(h_scores_in.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        reference[i] = {h_scores_in[i], h_indices_in[i]};
    }
    std::stable_sort(reference.begin(), reference.end(), std::greater<ScoreIndex>());

    bool correct = true;
    for (int i = 0; i < k; ++i) {
        if (std::abs(h_scores_out[i] - reference[i].score) > 1e-5 || h_indices_out[i] != reference[i].index) {
            fprintf(stderr, "Verification FAILED for %s at k=%d!\n", algo_name.c_str(), i);
            fprintf(stderr, "  GPU: score=%.4f, index=%d\n", h_scores_out[i], h_indices_out[i]);
            fprintf(stderr, "  CPU: score=%.4f, index=%d\n", reference[i].score, reference[i].index);
            correct = false;
            break;
        }
    }
    if (!correct) {
        throw std::runtime_error("Verification failed for " + algo_name);
    }
}


void runBenchmarks() {
    // UPDATED N_values for more granular testing
    std::vector<int> N_values = {
        32, 64, 128, 256, 512, 1024, 2048,
        2304, 2560, 2816, 3072, 3328, 3584, 3840,
        4096
    };
    std::vector<int> k_values = {4, 8, 16, 32, 64};
    const int num_iterations = 2000;
    const int block_size = 256;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- Top-K Reduction Stage Sorting Benchmark ---\n";
    std::cout << "All times are average latency in microseconds (us).\n";
    std::cout << "Lower is better. -1.000 indicates a skipped test.\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::setw(8) << "N" << std::setw(8) << "K"
              << std::setw(20) << "Warp Bitonic Sort"
              << std::setw(20) << "CUB Warp Merge"
              << std::setw(20) << "SMEM Bitonic Sort"
              << std::setw(20) << "CUB Block Merge"
              << std::setw(20) << "CUB Block Radix"
              << "\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int n : N_values) {
        for (int k : k_values) {
            if (k > n) continue;
            std::cout << std::setw(8) << n << std::setw(8) << k;

            std::vector<float> h_scores_in(n), h_scores_out(k);
            std::vector<int> h_indices_in(n), h_indices_out(k);
            std::iota(h_indices_in.begin(), h_indices_in.end(), 0);
            for(int i = 0; i < n; ++i) h_scores_in[i] = dis(gen);

            float *d_scores_in, *d_scores_out;
            int *d_indices_in, *d_indices_out;
            cudaMalloc(&d_scores_in, n * sizeof(float));
            cudaMalloc(&d_indices_in, n * sizeof(int));
            cudaMalloc(&d_scores_out, k * sizeof(float));
            cudaMalloc(&d_indices_out, k * sizeof(int));
            cudaMemcpy(d_scores_in, h_scores_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices_in, h_indices_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float ms;

            // --- 1. Warp Bitonic Sort ---
            float time_bitonic = -1.0f;
            if (n == 32) {
                warpBitonicSortKernel<<<1, 32>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                cudaEventRecord(start);
                for(int i = 0; i < num_iterations; ++i) warpBitonicSortKernel<<<1, 32>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                time_bitonic = (ms * 1000.0f) / num_iterations;
                cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost);
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "WarpBitonicSort");
            }
            std::cout << std::setw(20) << time_bitonic;

            // --- 2. CUB Warp Merge Sort ---
            float time_warp_merge = -1.0f;
            if (n <= 256) {
                auto launch_warp_merge = [&](auto n_const) {
                    constexpr int SORT_SIZE = n_const.value;
                    cubWarpMergeSortKernel<SORT_SIZE><<<1, 64>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                    cudaEventRecord(start);
                    for (int i = 0; i < num_iterations; ++i) cubWarpMergeSortKernel<SORT_SIZE><<<1, 64>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                };
                if (n == 32) launch_warp_merge(std::integral_constant<int, 32>());
                else if (n == 64) launch_warp_merge(std::integral_constant<int, 64>());
                else if (n == 128) launch_warp_merge(std::integral_constant<int, 128>());
                else if (n == 256) launch_warp_merge(std::integral_constant<int, 256>());
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                time_warp_merge = (ms * 1000.0f) / num_iterations;
                cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost);
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "CubWarpMergeSort");
            }
            std::cout << std::setw(20) << time_warp_merge;

            // --- 3. SMEM Bitonic Sort ---
            float time_smem_bitonic = -1.0f;
            if (n >= 128 && n <= 512) { // Suitable for medium N
                auto launch_smem_bitonic = [&](auto n_const) {
                    constexpr int SORT_SIZE = n_const.value;
                    sharedMemBitonicSortKernel<block_size, SORT_SIZE><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k); // Warmup
                    cudaEventRecord(start);
                    for (int i = 0; i < num_iterations; ++i) {
                        sharedMemBitonicSortKernel<block_size, SORT_SIZE><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                    }
                };
                if (n == 128) launch_smem_bitonic(std::integral_constant<int, 128>());
                else if (n == 256) launch_smem_bitonic(std::integral_constant<int, 256>());
                else if (n == 512) launch_smem_bitonic(std::integral_constant<int, 512>());

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                time_smem_bitonic = (ms * 1000.0f) / num_iterations;
                cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost);
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "SmemBitonicSort");
            }
            std::cout << std::setw(20) << time_smem_bitonic;

            // --- 4. CUB Block Merge Sort ---
            float time_block_merge = -1.0f;
            if (n >= 64) {
                const int items_per_thread = (n + block_size - 1) / block_size;
                auto launch_block_merge = [&](auto ipt_const) {
                    constexpr int IPT = ipt_const.value;
                    cubBlockMergeSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    cudaEventRecord(start);
                    for(int i = 0; i < num_iterations; ++i) cubBlockMergeSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                };
                // UPDATED to use a switch for better scalability
                switch(items_per_thread) {
                    case 1: launch_block_merge(std::integral_constant<int, 1>()); break;
                    case 2: launch_block_merge(std::integral_constant<int, 2>()); break;
                    case 3: launch_block_merge(std::integral_constant<int, 3>()); break;
                    case 4: launch_block_merge(std::integral_constant<int, 4>()); break;
                    case 8: launch_block_merge(std::integral_constant<int, 8>()); break;
                    case 9: launch_block_merge(std::integral_constant<int, 9>()); break;
                    case 10: launch_block_merge(std::integral_constant<int, 10>()); break;
                    case 11: launch_block_merge(std::integral_constant<int, 11>()); break;
                    case 12: launch_block_merge(std::integral_constant<int, 12>()); break;
                    case 13: launch_block_merge(std::integral_constant<int, 13>()); break;
                    case 14: launch_block_merge(std::integral_constant<int, 14>()); break;
                    case 15: launch_block_merge(std::integral_constant<int, 15>()); break;
                    case 16: launch_block_merge(std::integral_constant<int, 16>()); break;
                    case 32: launch_block_merge(std::integral_constant<int, 32>()); break;
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                time_block_merge = (ms * 1000.0f) / num_iterations;
                cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost);
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "CubBlockMergeSort");
            }
            std::cout << std::setw(20) << time_block_merge;

            // --- 5. CUB Block Radix Sort ---
            float time_block_radix = -1.0f;
            if (n >= 64) {
                const int items_per_thread = (n + block_size - 1) / block_size;
                auto launch_block_radix = [&](auto ipt_const) {
                    constexpr int IPT = ipt_const.value;
                    cubBlockRadixSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    cudaEventRecord(start);
                    for(int i = 0; i < num_iterations; ++i) cubBlockRadixSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                };
                 // UPDATED to use a switch for better scalability
                switch(items_per_thread) {
                    case 1: launch_block_radix(std::integral_constant<int, 1>()); break;
                    case 2: launch_block_radix(std::integral_constant<int, 2>()); break;
                    case 3: launch_block_radix(std::integral_constant<int, 3>()); break;
                    case 4: launch_block_radix(std::integral_constant<int, 4>()); break;
                    case 8: launch_block_radix(std::integral_constant<int, 8>()); break;
                    case 9: launch_block_radix(std::integral_constant<int, 9>()); break;
                    case 10: launch_block_radix(std::integral_constant<int, 10>()); break;
                    case 11: launch_block_radix(std::integral_constant<int, 11>()); break;
                    case 12: launch_block_radix(std::integral_constant<int, 12>()); break;
                    case 13: launch_block_radix(std::integral_constant<int, 13>()); break;
                    case 14: launch_block_radix(std::integral_constant<int, 14>()); break;
                    case 15: launch_block_radix(std::integral_constant<int, 15>()); break;
                    case 16: launch_block_radix(std::integral_constant<int, 16>()); break;
                    case 32: launch_block_radix(std::integral_constant<int, 32>()); break;
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                time_block_radix = (ms * 1000.0f) / num_iterations;
                cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost);
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "CubBlockRadixSort");
            }
            std::cout << std::setw(20) << time_block_radix;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_scores_in);
            cudaFree(d_indices_in);
            cudaFree(d_scores_out);
            cudaFree(d_indices_out);
            std::cout << "\n";
        }
    }
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "Benchmark finished successfully.\n";
}


int main() {
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    std::cout << "Running on CUDA device: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    
    try {
        runBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/*
Example output:

Running on CUDA device: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

--- Top-K Reduction Stage Sorting Benchmark ---
All times are average latency in microseconds (us).
Lower is better. -1.000 indicates a skipped test.
-------------------------------------------------------------------------------------------------------------------
       N       K   Warp Bitonic Sort      CUB Warp Merge   SMEM Bitonic Sort     CUB Block Merge     CUB Block Radix
-------------------------------------------------------------------------------------------------------------------
      32       4               8.252               8.592              -1.000              -1.000              -1.000
      32       8               5.596               5.951              -1.000              -1.000              -1.000
      32      16               8.088               8.322              -1.000              -1.000              -1.000
      32      32               5.336               6.161              -1.000              -1.000              -1.000
      64       4              -1.000               7.430              -1.000               8.898               9.002
      64       8              -1.000               5.084              -1.000               6.892              10.131
      64      16              -1.000               5.179              -1.000               8.163               9.165
      64      32              -1.000               5.057              -1.000               6.513               6.485
      64      64              -1.000               6.066              -1.000               5.384               7.271
     128       4              -1.000               7.909               5.999               5.136               6.977
     128       8              -1.000               5.323               6.155               5.373               6.904
     128      16              -1.000               5.477               6.780               8.219              10.200
     128      32              -1.000               5.250               6.565               6.679               6.903
     128      64              -1.000               6.563               6.499               5.433               6.778
     256       4              -1.000               9.878               9.944               8.039               7.601
     256       8              -1.000               8.135               9.813               8.169               9.993
     256      16              -1.000               8.715               6.860               5.197               6.881
     256      32              -1.000               7.315               7.094               5.472               7.004
     256      64              -1.000               7.136               7.007               5.438               6.898
     512       4              -1.000              -1.000              11.099               7.110               8.135
     512       8              -1.000              -1.000              11.171               7.154               8.174
     512      16              -1.000              -1.000              11.514               6.404               8.354
     512      32              -1.000              -1.000              10.948               8.152               9.887
     512      64              -1.000              -1.000              10.877               6.926               8.422
    1024       4              -1.000              -1.000              -1.000               6.741               6.711
    1024       8              -1.000              -1.000              -1.000               6.869               6.737
    1024      16              -1.000              -1.000              -1.000               8.513               6.738
    1024      32              -1.000              -1.000              -1.000               7.402               6.715
    1024      64              -1.000              -1.000              -1.000               6.985               6.932
    2048       4              -1.000              -1.000              -1.000               8.120              10.537
    2048       8              -1.000              -1.000              -1.000               7.996               9.643
    2048      16              -1.000              -1.000              -1.000               8.832              10.723
    2048      32              -1.000              -1.000              -1.000               8.209              10.205
    2048      64              -1.000              -1.000              -1.000              11.587              10.914
    2304       4              -1.000              -1.000              -1.000              10.038              10.306
    2304       8              -1.000              -1.000              -1.000              14.395              13.661
    2304      16              -1.000              -1.000              -1.000              10.006              10.511
    2304      32              -1.000              -1.000              -1.000               9.570              10.126
    2304      64              -1.000              -1.000              -1.000               9.906               9.091
    2560       4              -1.000              -1.000              -1.000              10.101              10.537
    2560       8              -1.000              -1.000              -1.000              10.276              10.627
    2560      16              -1.000              -1.000              -1.000              10.290              10.336
    2560      32              -1.000              -1.000              -1.000               9.988              10.515
    2560      64              -1.000              -1.000              -1.000              10.114              10.493
    2816       4              -1.000              -1.000              -1.000              10.636              11.668
    2816       8              -1.000              -1.000              -1.000              10.870              12.065
    2816      16              -1.000              -1.000              -1.000              13.140              11.431
    2816      32              -1.000              -1.000              -1.000              10.693              11.589
    2816      64              -1.000              -1.000              -1.000              10.338              11.471
    3072       4              -1.000              -1.000              -1.000              10.771              11.813
    3072       8              -1.000              -1.000              -1.000              11.178              12.247
    3072      16              -1.000              -1.000              -1.000              11.599              12.081
    3072      32              -1.000              -1.000              -1.000              11.110              11.389
    3072      64              -1.000              -1.000              -1.000              10.870              11.927
    3328       4              -1.000              -1.000              -1.000              12.309              13.260
    3328       8              -1.000              -1.000              -1.000              12.041              11.398
    3328      16              -1.000              -1.000              -1.000              11.999              12.774
    3328      32              -1.000              -1.000              -1.000              12.159              11.351
    3328      64              -1.000              -1.000              -1.000              12.210              11.370
    3584       4              -1.000              -1.000              -1.000              12.275              13.899
    3584       8              -1.000              -1.000              -1.000              11.646              11.216
    3584      16              -1.000              -1.000              -1.000              11.606              11.317
    3584      32              -1.000              -1.000              -1.000              11.587              11.330
    3584      64              -1.000              -1.000              -1.000              11.763              11.169
    3840       4              -1.000              -1.000              -1.000              11.822              12.126
    3840       8              -1.000              -1.000              -1.000              12.193              13.216
    3840      16              -1.000              -1.000              -1.000              12.421              12.054
    3840      32              -1.000              -1.000              -1.000              12.346              12.114
    3840      64              -1.000              -1.000              -1.000              11.972              12.225
    4096       4              -1.000              -1.000              -1.000              14.810              13.647
    4096       8              -1.000              -1.000              -1.000              15.227              13.686
    4096      16              -1.000              -1.000              -1.000              15.782              14.332
    4096      32              -1.000              -1.000              -1.000              14.924              14.523
    4096      64              -1.000              -1.000              -1.000              14.927              13.648
-------------------------------------------------------------------------------------------------------------------

From the result, The best algo is:
32:       warp bitonic sort?
128:      cub warp merge
256~3072: cub block merge
>3072:    cub Block Radix
*/
#endif