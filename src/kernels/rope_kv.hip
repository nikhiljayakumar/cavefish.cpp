#include "common.hpp"

#include <hip/hip_runtime.h>

__global__ void rope_kv_kernel(const float* __restrict__ q,
                               const float* __restrict__ k,
                               const float* __restrict__ v
                               const int N){
    
}

void rope_kv(const float* q, const float* k, const float* v, const int N = 4096){
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

    dim3 block(kThreadsPerBlock);
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    rope_kv_kernel<<<grid, block, stream>>>(q,k,v,N);
    HIP_CHECK(hipGetLastError());
}