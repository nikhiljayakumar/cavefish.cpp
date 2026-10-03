// src/kernels/residual_add.hip
#include "residual_add.h"
#include "common.hpp"

namespace {

__global__ void residual_add_kernel(const float* __restrict__ x,
                                     const float* __restrict__ residual,
                                     float* __restrict__ y,
                                     int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = x[i] + residual[i];
    }
}

} // namespace

void residual_add(const float* x, const float* residual, float* y, int N, hipStream_t stream) {
    constexpr int kThreadsPerBlock = 1024;
    dim3 block(kThreadsPerBlock);
    dim3 grid(1);

    residual_add_kernel<<<grid, block, 0, stream>>>(x, residual, y, N);
    HIP_CHECK(hipGetLastError());
}