#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Template for BlockRadixSort benchmark
template<int BLOCK_SIZE>
__global__ void blockRadixSortKernel(int* d_data, int* d_output, int num_items) {
    typedef cub::BlockRadixSort<int, BLOCK_SIZE> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    
    int thread_data[1];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_items) {
        thread_data[0] = d_data[tid];
    } else {
        thread_data[0] = INT_MAX; // Padding for out-of-bounds
    }
    
    BlockRadixSort(temp_storage).Sort(thread_data);
    
    if (tid < num_items) {
        d_output[tid] = thread_data[0];
    }
}

// Template for BlockMergeSort benchmark
template<int BLOCK_SIZE>
__global__ void blockMergeSortKernel(int* d_data, int* d_output, int num_items) {
    typedef cub::BlockMergeSort<int, BLOCK_SIZE> BlockMergeSort;
    __shared__ typename BlockMergeSort::TempStorage temp_storage;
    
    int thread_data[1];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_items) {
        thread_data[0] = d_data[tid];
    } else {
        thread_data[0] = INT_MAX;
    }
    
    BlockMergeSort(temp_storage).Sort(thread_data);
    
    if (tid < num_items) {
        d_output[tid] = thread_data[0];
    }
}

// WarpMergeSort benchmark (fixed warp size of 32)
__global__ void warpMergeSortKernel(int* d_data, int* d_output, int num_items) {
    typedef cub::WarpMergeSort<int, 32> WarpMergeSort;
    __shared__ typename WarpMergeSort::TempStorage temp_storage[32]; // One per warp
    
    int thread_data[1];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    
    if (tid < num_items) {
        thread_data[0] = d_data[tid];
    } else {
        thread_data[0] = INT_MAX;
    }
    
    WarpMergeSort(temp_storage[warp_id]).Sort(thread_data);
    
    if (tid < num_items) {
        d_output[tid] = thread_data[0];
    }
}

// Verification function
bool verifySorted(const std::vector<int>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i-1] > data[i]) {
            return false;
        }
    }
    return true;
}

// Benchmark function template
template<int BLOCK_SIZE>
double benchmarkSort(const std::vector<int>& h_data, const std::string& algorithm_name, int num_iterations = 1000) {
    const int num_items = h_data.size();
    
    // Allocate GPU memory
    int *d_data, *d_output;
    cudaMalloc(&d_data, num_items * sizeof(int));
    cudaMalloc(&d_output, num_items * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_data, h_data.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        if (algorithm_name == "BlockRadixSort") {
            blockRadixSortKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        } else if (algorithm_name == "BlockMergeSort") {
            blockMergeSortKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        } else if (algorithm_name == "WarpMergeSort") {
            warpMergeSortKernel<<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        }
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        if (algorithm_name == "BlockRadixSort") {
            blockRadixSortKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        } else if (algorithm_name == "BlockMergeSort") {
            blockMergeSortKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        } else if (algorithm_name == "WarpMergeSort") {
            warpMergeSortKernel<<<1, BLOCK_SIZE>>>(d_data, d_output, num_items);
        }
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Verify result
    std::vector<int> h_output(num_items);
    cudaMemcpy(h_output.data(), d_output, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    
    if (!verifySorted(h_output)) {
        std::cout << "ERROR: " << algorithm_name << " did not sort correctly!" << std::endl;
    }
    
    // Clean up
    cudaFree(d_data);
    cudaFree(d_output);
    
    double total_time = std::chrono::duration<double, std::micro>(end - start).count();
    return total_time / num_iterations; // Average time per sort in microseconds
}

void runBenchmarks() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000000);
    
    std::vector<int> sizes = {128, 256, 512};
    const int num_iterations = 10000;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nCUB Sorting Performance Comparison (Average time per sort in microseconds)\n";
    std::cout << "=============================================================================\n";
    std::cout << std::setw(12) << "Size" 
              << std::setw(18) << "BlockRadixSort" 
              << std::setw(18) << "BlockMergeSort"
              << std::setw(18) << "WarpMergeSort" 
              << std::setw(15) << "Best Algorithm\n";
    std::cout << "-----------------------------------------------------------------------------\n";
    
    for (int size : sizes) {
        // Generate random data
        std::vector<int> data(size);
        for (int& val : data) {
            val = dis(gen);
        }
        
        double radix_time = 0, merge_time = 0, warp_time = 0;
        
        // Benchmark each algorithm
        if (size == 128) {
            radix_time = benchmarkSort<128>(data, "BlockRadixSort", num_iterations);
            merge_time = benchmarkSort<128>(data, "BlockMergeSort", num_iterations);
            warp_time = benchmarkSort<128>(data, "WarpMergeSort", num_iterations);
        } else if (size == 256) {
            radix_time = benchmarkSort<256>(data, "BlockRadixSort", num_iterations);
            merge_time = benchmarkSort<256>(data, "BlockMergeSort", num_iterations);
            warp_time = benchmarkSort<256>(data, "WarpMergeSort", num_iterations);
        } else if (size == 512) {
            radix_time = benchmarkSort<512>(data, "BlockRadixSort", num_iterations);
            merge_time = benchmarkSort<512>(data, "BlockMergeSort", num_iterations);
            warp_time = benchmarkSort<512>(data, "WarpMergeSort", num_iterations);
        }
        
        // Find the best algorithm
        std::string best_algo = "BlockRadixSort";
        double best_time = radix_time;
        if (merge_time < best_time) {
            best_algo = "BlockMergeSort";
            best_time = merge_time;
        }
        if (warp_time < best_time) {
            best_algo = "WarpMergeSort";
            best_time = warp_time;
        }
        
        std::cout << std::setw(12) << size
                  << std::setw(18) << radix_time
                  << std::setw(18) << merge_time
                  << std::setw(18) << warp_time
                  << std::setw(15) << best_algo << "\n";
    }
    
    std::cout << "\nNotes:\n";
    std::cout << "- BlockRadixSort: Best for integers, O(k*n) complexity\n";
    std::cout << "- BlockMergeSort: Best for comparison-based sorting, O(n log n)\n";
    std::cout << "- WarpMergeSort: Operates at warp level, good for smaller sizes\n";
    std::cout << "- All tests use random integer data in range [0, 1000000]\n";
    std::cout << "- Results may vary based on GPU architecture and data patterns\n";
}

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    runBenchmarks();
    
    return 0;
}
