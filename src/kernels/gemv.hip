#include <hip/hip_runtime.h>
#include <iostream>
#include "gemv.h"
#include "common.hpp"
#include "verify.hpp"

namespace {

__global__ void gemv_kernel(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {

    int warp_id = threadIdx.x / kWarpSize;
    int lane    = threadIdx.x % kWarpSize;
    int row     = blockIdx.x * (blockDim.x / kWarpSize) + warp_id;
    if (row >= M) return;

    float partial = 0.0f;
    for (int col = lane; col < N; col += kWarpSize) {
        partial += A[static_cast<size_t>(row) * N + col] * x[col];
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        partial += __shfl_down(partial, offset);

    if (lane == 0) y[row] = partial;
}

} // namespace

void gemv(const float* A, const float* x, float* y, int M, int N, hipStream_t stream) {
    constexpr int kThreadsPerBlock = 1024;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

    dim3 block(kThreadsPerBlock);
    dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);
    size_t shmem_bytes = static_cast<size_t>(N) * sizeof(float);

    gemv_kernel<<<grid, block, shmem_bytes, stream>>>(A, x, y, M, N);
    HIP_CHECK(hipGetLastError());
}