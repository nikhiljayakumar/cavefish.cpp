#include <hip/hip_runtime.h>
#include <iostream>
#include "system_benchmarks.hpp"
#include "common.hpp"
#include "benchmark.hpp"


__global__ void acc(const float4* __restrict__ in, float* __restrict__ out, int n4){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int i = idx;

    for (; i + 3*stride < n4; i += 4*stride){
        float4 v0 = in[i];
        float4 v1 = in[i + stride];
        float4 v2 = in[i + 2*stride];
        float4 v3 = in[i + 3*stride];
        acc0 = acc0 * 1.000000119f + (v0.x+v0.y+v0.z+v0.w);
        acc1 = acc1 * 1.000000119f + (v1.x+v1.y+v1.z+v1.w);
        acc2 = acc2 * 1.000000119f + (v2.x+v2.y+v2.z+v2.w);
        acc3 = acc3 * 1.000000119f + (v3.x+v3.y+v3.z+v3.w);
    }
    for (; i < n4; i += stride){ // tail for leftovers that don't fit the 4x pattern
        float4 v = in[i];
        acc0 = acc0 * 1.000000119f + (v.x+v.y+v.z+v.w);
    }

    float accum = acc0 + acc1 + acc2 + acc3;
    if (idx < n4) out[idx] = accum;
}

void run_memory_read_benchmark(){
    std::vector<int> n_sweep = {(int) 125e3, (int) 5e5, (int) 4e6, (int) 128e6, (int) 1e9};

    for (int n : n_sweep){
        size_t bytes = n * sizeof(float);

        float* h_a = new float[n];

        for (int i = 0; i < n; i+=4) {
            h_a[i] = 1.0f * i;
        }

        float *d_a, *d_b;
        HIP_CHECK(hipMalloc(&d_a, bytes));
        HIP_CHECK(hipMalloc(&d_b, bytes>>2));

        HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));

        // try 86016 threads (28 CUs * 96 warps per CU * 32 threads per warp)
        dim3 blockSize(256, 1, 1); // threads per block
        dim3 gridSize(336, 1, 1);

        BenchmarkResult result = run_benchmark([&](){
            acc<<<gridSize, blockSize, 0, 0>>>(
                reinterpret_cast<float4*>(d_a), 
                (d_b), 
                n>>2
            );
            HIP_CHECK(hipGetLastError());
        });

        HIP_CHECK(hipDeviceSynchronize());

        double gbps = (1.0f * bytes / 1.0e9) / (result.avg_time_ms / 1000.0);

        std::cout << "Buffer Size (elements): " << n << '\n';
        // std::cout << "  Final reduced sum: " << final_sum << std::endl;
        std::cout << "  Avg Time taken (ms): " << result.avg_time_ms << std::endl;
        std::cout << "  Median Time taken (ms): " << result.median_time_ms << std::endl;
        std::cout << "  GB/s: " << gbps << std::endl;

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        
        delete[] h_a;
    }
}