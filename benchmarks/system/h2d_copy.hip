#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "system_benchmarks.hpp"
#include "common.hpp"
#include "benchmark.hpp"

// Pageable memory is GPU (device) memory
// Pinned memory is CPU memory that the GPU can access
// and the OS can't swap it out (i.e. its pinned to physical mem)
// So we are measuring host to device copy for both pageable memory and pinned memory
void run_h2d_copy_benchmark(){
    std::vector<int> n_sweep = {(int)1e6, (int)16e6, (int)128e6};

    for (int n : n_sweep){
        size_t bytes = n * sizeof(float);
        float* d_ptr;
        HIP_CHECK(hipMalloc(&d_ptr, bytes));

        float* h_pageable = new float[n];
        for (int i = 0; i < n; i++) h_pageable[i] = 1.0f * i;
        BenchmarkResult pageable = run_benchmark([&](){
            HIP_CHECK(hipMemcpy(d_ptr, h_pageable, bytes, hipMemcpyHostToDevice));
        });
        double pageable_gbps = (bytes / 1.0e9) / (pageable.avg_time_ms / 1000.0);
        delete[] h_pageable;

        float* h_pinned;
        HIP_CHECK(hipHostMalloc(&h_pinned, bytes));
        for (int i = 0; i < n; i++) h_pinned[i] = 1.0f * i;
        BenchmarkResult pinned = run_benchmark([&](){
            HIP_CHECK(hipMemcpy(d_ptr, h_pinned, bytes, hipMemcpyHostToDevice));
        });
        double pinned_gbps = (bytes / 1.0e9) / (pinned.avg_time_ms / 1000.0);
        HIP_CHECK(hipHostFree(h_pinned));

        std::cout << "Buffer Size (elements): " << n << std::endl;
        std::cout << "  Pageable H2D: " << pageable.avg_time_ms << " ms, " << pageable_gbps << " GB/s" << std::endl;
        std::cout << "  Pinned H2D:   " << pinned.avg_time_ms << " ms, " << pinned_gbps << " GB/s" << std::endl;

        HIP_CHECK(hipFree(d_ptr));
    }
}