#include "silu_mul.h"
#include "common.hpp"

namespace {

__global__ void silu_mul_kernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 float* __restrict__ y,
                                 int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

} // namespace

void silu_mul(const float* gate, const float* up, float* y, int N, hipStream_t stream) {
    constexpr int kThreadsPerBlock = 1024;
    dim3 block(kThreadsPerBlock);
    dim3 grid(1);
    silu_mul_kernel<<<grid, block, 0, stream>>>(gate, up, y, N);
    HIP_CHECK(hipGetLastError());
}