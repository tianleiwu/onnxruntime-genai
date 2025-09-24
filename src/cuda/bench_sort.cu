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

// // A simple macro to wrap CUDA API calls and check for errors
// #define CUDA_CHECK(call)                                                                   \
//   do {                                                                                     \
//     cudaError_t err = call;                                                                \
//     if (err != cudaSuccess) {                                                              \
//       fprintf(stderr, "CUDA error in %s at %s:%d - %s\n", __FILE__, __func__, __LINE__,    \
//               cudaGetErrorString(err));                                                    \
//       throw std::runtime_error(cudaGetErrorString(err));                                   \
//     }                                                                                      \
//   } while (0)


// --- User-Provided Headers ---
// Note: These must be in the same directory or in the include path.
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_stable_sort_helper.cuh"

// The user did not provide this file, but it's a common pattern to have a
// utilities header. We'll define the required function here.
namespace Generators {
namespace cuda {
namespace topk_common {
// Returns the next power of two for a given integer.
constexpr int NextPowerOfTwo(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
}
}
}


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
// This kernel benchmarks the provided register-based warp bitonic sort.
// It's designed to sort exactly 32 elements. We launch one block with one warp.
__global__ void warpBitonicSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int k) {
    if (threadIdx.x >= 32) return;

    // Load data into registers
    float my_score = scores_in[threadIdx.x];
    int my_index = indices_in[threadIdx.x];

    // Perform the sort
    Generators::cuda::bitonic_sort::WarpBitonicSort(my_score, my_index);

    // Write top K results back to global memory
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = my_score;
        indices_out[threadIdx.x] = my_index;
    }
}

// --- 2. CUB Warp Merge Sort ---
// Benchmarks cub::WarpMergeSort for small N. Data is loaded to shared memory,
// sorted by the first warp, and then results are written back.
template <int SORT_SIZE>
__global__ void cubWarpMergeSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int k) {
    constexpr int SORT_SIZE_PO2 = Generators::cuda::topk_common::NextPowerOfTwo(SORT_SIZE);

    // Shared memory for sorting. We use a union to potentially alias storage
    // for different sort types if needed, which is good practice.
    union SharedStorage {
        struct {
            float scores[SORT_SIZE_PO2];
            int indices[SORT_SIZE_PO2];
        } sort_data;

        // CUB requires temporary storage for its operation.
        typename cub::WarpMergeSort<float, (SORT_SIZE_PO2 + 31) / 32, 32, int>::TempStorage cub_storage;
    };
    __shared__ SharedStorage smem;

    // Load data from global to shared memory
    for (int i = threadIdx.x; i < SORT_SIZE; i += blockDim.x) {
        smem.sort_data.scores[i] = scores_in[i];
        smem.sort_data.indices[i] = indices_in[i];
    }
    // Pad the rest of the shared memory with sentinel values
    for (int i = threadIdx.x + SORT_SIZE; i < SORT_SIZE_PO2; i += blockDim.x) {
         smem.sort_data.scores[i] = -FLT_MAX;
         smem.sort_data.indices[i] = INT_MAX;
    }
    __syncthreads();

    // Only the first warp performs the sort
    if (threadIdx.x < 32) {
       Generators::cuda::bitonic_sort::WarpMergeSort<SORT_SIZE_PO2>(
           smem.sort_data.scores,
           smem.sort_data.indices,
           &smem.cub_storage,
           SORT_SIZE);
    }
    __syncthreads();

    // Write top K results from shared memory back to global memory
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = smem.sort_data.scores[threadIdx.x];
        indices_out[threadIdx.x] = smem.sort_data.indices[threadIdx.x];
    }
}

// --- 3. Shared Memory Bitonic Sort ---
// Benchmarks the shared-memory-based bitonic sort.
template <int BLOCK_SIZE, int SORT_SIZE>
__global__ void sharedMemBitonicSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int n, int k) {
    // Shared memory for sorting. SORT_SIZE must be a power of two.
    __shared__ float smem_scores[SORT_SIZE];
    __shared__ int smem_indices[SORT_SIZE];

    // Load data from global to shared memory
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        smem_scores[i] = scores_in[i];
        smem_indices[i] = indices_in[i];
    }
    // Pad the rest of the shared memory with sentinel values
    for (int i = threadIdx.x + n; i < SORT_SIZE; i += BLOCK_SIZE) {
         smem_scores[i] = -FLT_MAX;
         smem_indices[i] = INT_MAX;
    }
    __syncthreads();

    // Perform the sort
    Generators::cuda::bitonic_sort::SharedMemBitonicSort<BLOCK_SIZE, SORT_SIZE>(smem_scores, smem_indices);
    __syncthreads();

    // Write top K results from shared memory back to global memory
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = smem_scores[threadIdx.x];
        indices_out[threadIdx.x] = smem_indices[threadIdx.x];
    }
}


// --- 4. CUB Block Radix Sort ---
// Benchmarks cub::BlockRadixSort for block-wide sorting.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockRadixSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int n, int k) {
    // CUB temporary storage in shared memory
    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Load data into per-thread register arrays in a blocked arrangement
    float thread_scores[ITEMS_PER_THREAD];
    int thread_indices[ITEMS_PER_THREAD];

    cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
    cub::LoadDirectBlocked(threadIdx.x, indices_in, thread_indices, n, INT_MAX);

    // Perform the block-wide sort. This takes blocked data and produces striped output.
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_scores, thread_indices);

    // Write top K results back to global memory.
    // The sorted elements are now striped across threads. The top K will be
    // in the first slot of the first K threads.
    if (threadIdx.x < k) {
        scores_out[threadIdx.x] = thread_scores[0];
        indices_out[threadIdx.x] = thread_indices[0];
    }
}

// --- 5. CUB Block Merge Sort ---
// Benchmarks cub::BlockMergeSort, a comparison-based block-wide sort.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void cubBlockMergeSortKernel(const float* scores_in, const int* indices_in, float* scores_out, int* indices_out, int n, int k) {
    // CUB temporary storage in shared memory
    using BlockMergeSort = cub::BlockMergeSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
    __shared__ typename BlockMergeSort::TempStorage temp_storage;

    // Load data into per-thread register arrays in a blocked arrangement
    float thread_scores[ITEMS_PER_THREAD];
    int thread_indices[ITEMS_PER_THREAD];

    cub::LoadDirectBlocked(threadIdx.x, scores_in, thread_scores, n, -FLT_MAX);
    cub::LoadDirectBlocked(threadIdx.x, indices_in, thread_indices, n, INT_MAX);

    // Perform the block-wide sort using a descending comparator
    BlockMergeSort(temp_storage).Sort(thread_scores, thread_indices, DescendingOp());

    // Write top K results back to global memory.
    // After sorting, data remains in a blocked layout.
    cub::StoreDirectBlocked(threadIdx.x, scores_out, thread_scores, k);
    cub::StoreDirectBlocked(threadIdx.x, indices_out, thread_indices, k);
}


// ====================================================================================
// VERIFICATION AND MAIN BENCHMARK LOGIC
// ====================================================================================

// Host-side struct for verification
struct ScoreIndex {
    float score;
    int index;

    bool operator>(const ScoreIndex& other) const {
        if (score != other.score) {
            return score > other.score;
        }
        // Stable sort tie-breaking: smaller index wins
        return index < other.index;
    }
};

// Verifies the GPU output against a CPU-sorted reference
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
    std::vector<int> N_values = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> k_values = {4, 8, 16, 32, 64};

    const int num_iterations = 2000;
    const int block_size = 256; // Common block size for block-wide sorts

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- Top-K Reduction Stage Sorting Benchmark ---\n";
    std::cout << "All times are average latency in microseconds (us).\n";
    std::cout << "Lower is better. -1.000 indicates a skipped test.\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(20) << "Warp Bitonic Sort"
              << std::setw(20) << "CUB Warp Merge"
              << std::setw(20) << "SMEM Bitonic Sort"
              << std::setw(20) << "CUB Block Merge"
              << std::setw(20) << "CUB Block Radix"
              << "\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

    // Random data generation
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int n : N_values) {
        for (int k : k_values) {
            if (k > n) continue;

            std::cout << std::setw(8) << n << std::setw(8) << k;

            // --- Allocate memory ---
            std::vector<float> h_scores_in(n), h_scores_out(k);
            std::vector<int> h_indices_in(n), h_indices_out(k);
            for(int i = 0; i < n; ++i) {
                h_scores_in[i] = dis(gen);
                h_indices_in[i] = i;
            }

            float *d_scores_in, *d_scores_out;
            int *d_indices_in, *d_indices_out;
            CUDA_CHECK(cudaMalloc(&d_scores_in, n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_indices_in, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_scores_out, k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_indices_out, k * sizeof(int)));

            CUDA_CHECK(cudaMemcpy(d_scores_in, h_scores_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_indices_in, h_indices_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            float ms;

            // --- 1. Warp Bitonic Sort ---
            float time_bitonic = -1.0f;
            if (n == 32) {
                warpBitonicSortKernel<<<1, 32>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k); // Warmup
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaEventRecord(start));
                for(int i = 0; i < num_iterations; ++i) {
                    warpBitonicSortKernel<<<1, 32>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                }
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                time_bitonic = (ms * 1000.0f) / num_iterations;
                CUDA_CHECK(cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost));
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "WarpBitonicSort");
            }
            std::cout << std::setw(20) << time_bitonic;

            // --- 2. CUB Warp Merge Sort ---
            float time_warp_merge = -1.0f;
            if (n <= 256) { // Suitable for smaller N
                // Dispatch to the correct kernel template instance
                auto launch_warp_merge = [&](auto n_const) {
                    constexpr int SORT_SIZE = n_const.value;
                    cubWarpMergeSortKernel<SORT_SIZE><<<1, 64>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k); // Warmup
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaEventRecord(start));
                    for (int i = 0; i < num_iterations; ++i) {
                        cubWarpMergeSortKernel<SORT_SIZE><<<1, 64>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, k);
                    }
                };
                if (n == 32) launch_warp_merge(std::integral_constant<int, 32>());
                else if (n == 64) launch_warp_merge(std::integral_constant<int, 64>());
                else if (n == 128) launch_warp_merge(std::integral_constant<int, 128>());
                else if (n == 256) launch_warp_merge(std::integral_constant<int, 256>());

                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                time_warp_merge = (ms * 1000.0f) / num_iterations;
                CUDA_CHECK(cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost));
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "CubWarpMergeSort");
            }
            std::cout << std::setw(20) << time_warp_merge;

            // --- 3. Shared Memory Bitonic Sort ---
            float time_smem_bitonic = -1.0f;
            if (n >= 128) {
                const int sort_size_po2 = Generators::cuda::topk_common::NextPowerOfTwo(n);

                auto launch_smem_bitonic = [&](auto sort_size_const){
                    constexpr int SORT_SIZE = sort_size_const.value;
                    sharedMemBitonicSortKernel<block_size, SORT_SIZE><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaEventRecord(start));
                    for(int i = 0; i < num_iterations; ++i) {
                        sharedMemBitonicSortKernel<block_size, SORT_SIZE><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    }
                };

                if (sort_size_po2 == 128) launch_smem_bitonic(std::integral_constant<int, 128>());
                else if (sort_size_po2 == 256) launch_smem_bitonic(std::integral_constant<int, 256>());
                else if (sort_size_po2 == 512) launch_smem_bitonic(std::integral_constant<int, 512>());
                else if (sort_size_po2 == 1024) launch_smem_bitonic(std::integral_constant<int, 1024>());
                else if (sort_size_po2 == 2048) launch_smem_bitonic(std::integral_constant<int, 2048>());
                else if (sort_size_po2 == 4096) launch_smem_bitonic(std::integral_constant<int, 4096>());
                // else if (sort_size_po2 == 8192) launch_smem_bitonic(std::integral_constant<int, 8192>());

                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                time_smem_bitonic = (ms * 1000.0f) / num_iterations;
                CUDA_CHECK(cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost));
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "SharedMemBitonicSort");
            }
            std::cout << std::setw(20) << time_smem_bitonic;


            // --- 4. CUB Block Merge Sort ---
            float time_block_merge = -1.0f;
            if (n >= 64) { // Not efficient for very small N
                const int items_per_thread = (n + block_size - 1) / block_size;
                auto launch_block_merge = [&](auto ipt_const) {
                    constexpr int IPT = ipt_const.value;
                    cubBlockMergeSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaEventRecord(start));
                    for(int i = 0; i < num_iterations; ++i) {
                         cubBlockMergeSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    }
                };
                // FIXED: Added missing cases for items_per_thread to prevent error
                if(items_per_thread == 1) launch_block_merge(std::integral_constant<int, 1>());
                else if (items_per_thread == 2) launch_block_merge(std::integral_constant<int, 2>());
                else if (items_per_thread == 3) launch_block_merge(std::integral_constant<int, 3>());
                else if (items_per_thread == 4) launch_block_merge(std::integral_constant<int, 4>());
                else if (items_per_thread == 8) launch_block_merge(std::integral_constant<int, 8>());
                else if (items_per_thread == 16) launch_block_merge(std::integral_constant<int, 16>());
                else if (items_per_thread == 32) launch_block_merge(std::integral_constant<int, 32>());

                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                time_block_merge = (ms * 1000.0f) / num_iterations;
                CUDA_CHECK(cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost));
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
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaEventRecord(start));
                    for(int i = 0; i < num_iterations; ++i) {
                         cubBlockRadixSortKernel<block_size, IPT><<<1, block_size>>>(d_scores_in, d_indices_in, d_scores_out, d_indices_out, n, k);
                    }
                };
                // FIXED: Added missing cases for items_per_thread to prevent error
                if(items_per_thread == 1) launch_block_radix(std::integral_constant<int, 1>());
                else if (items_per_thread == 2) launch_block_radix(std::integral_constant<int, 2>());
                else if (items_per_thread == 3) launch_block_radix(std::integral_constant<int, 3>());
                else if (items_per_thread == 4) launch_block_radix(std::integral_constant<int, 4>());
                else if (items_per_thread == 8) launch_block_radix(std::integral_constant<int, 8>());
                else if (items_per_thread == 16) launch_block_radix(std::integral_constant<int, 16>());
                else if (items_per_thread == 32) launch_block_radix(std::integral_constant<int, 32>());

                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                time_block_radix = (ms * 1000.0f) / num_iterations;
                CUDA_CHECK(cudaMemcpy(h_scores_out.data(), d_scores_out, k * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_indices_out.data(), d_indices_out, k * sizeof(int), cudaMemcpyDeviceToHost));
                verifyTopK(h_scores_out, h_indices_out, h_scores_in, h_indices_in, k, "CubBlockRadixSort");
            }
            std::cout << std::setw(20) << time_block_radix;

            // --- Cleanup ---
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_scores_in));
            CUDA_CHECK(cudaFree(d_indices_in));
            CUDA_CHECK(cudaFree(d_scores_out));
            CUDA_CHECK(cudaFree(d_indices_out));

            std::cout << "\n";
        }
    }
    std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "Benchmark finished successfully.\n";
}


int main() {
    int deviceId;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

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
       N       K   Warp Bitonic Sort      CUB Warp Merge   SMEM Bitonic Sort     CUB Block Merge     CUB Block Radix
-------------------------------------------------------------------------------------------------------------------
      32       4               9.145               8.535              -1.000              -1.000              -1.000
      32       8               5.976               8.860              -1.000              -1.000              -1.000
      32      16               8.093               8.880              -1.000              -1.000              -1.000
      32      32               4.675               5.748              -1.000              -1.000              -1.000
      64       4              -1.000               7.912              -1.000               9.100               8.230
      64       8              -1.000               5.430              -1.000               8.716               9.174
      64      16              -1.000               8.527              -1.000               8.386              10.444
      64      32              -1.000               7.980              -1.000               8.347              10.297
      64      64              -1.000               5.093              -1.000               8.309               8.993
     128       4              -1.000               8.129               7.935               7.955               8.961
     128       8              -1.000               7.874               7.983               7.780               8.961
     128      16              -1.000               6.733               8.243               8.452              10.095
     128      32              -1.000               5.202               8.090               7.851               8.953
     128      64              -1.000               7.927               8.038               7.936               8.644
     256       4              -1.000              10.421               9.656               7.784               8.759
     256       8              -1.000               7.612               9.829               8.365              10.261
     256      16              -1.000               8.513               9.697               8.282               8.126
     256      32              -1.000               7.784               9.845               8.075               7.081
     256      64              -1.000               7.434               6.727               5.406               6.944
     512       4              -1.000              -1.000              11.204               6.962               7.563
     512       8              -1.000              -1.000              13.589               7.139               7.805
     512      16              -1.000              -1.000              11.242               7.008               7.632
     512      32              -1.000              -1.000              11.500               8.042               9.795
     512      64              -1.000              -1.000              11.795               6.890               7.581
    1024       4              -1.000              -1.000              16.815               6.793               6.764
    1024       8              -1.000              -1.000              16.799               6.687               7.287
    1024      16              -1.000              -1.000              16.707               7.057               6.980
    1024      32              -1.000              -1.000              16.713               6.995               6.979
    1024      64              -1.000              -1.000              16.849               6.867               7.010
    2048       4              -1.000              -1.000              32.511               7.916              10.211
    2048       8              -1.000              -1.000              32.526               7.882               9.620
    2048      16              -1.000              -1.000              32.493               7.907              10.346
    2048      32              -1.000              -1.000              32.647               7.905              10.518
    2048      64              -1.000              -1.000              32.629               7.861              10.388
    4096       4              -1.000              -1.000              67.139              17.468              16.480
    4096       8              -1.000              -1.000              67.175              15.430              16.470
    4096      16              -1.000              -1.000              67.786              15.383              15.363
    4096      32              -1.000              -1.000              67.761              17.224              16.579
    4096      64              -1.000              -1.000              67.248              17.627              16.804
------------------------------------------------------------------------------------------------------------------

From the result, The best algo is:
32:       warp bitonic sort?
128:      cub warp merge
256~2048: cub block merge
4096:     cub Block Radix
*/
#endif