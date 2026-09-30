#include "gemv_q8_0.h"
#include "common.hpp"

namespace {

constexpr int kQuantBlockSize = 32;

__global__ void gemv_q8_0_kernel(const int8_t* __restrict__ A_q,
                                  const float* __restrict__ A_scales,
                                  const int8_t* __restrict__ x_q,
                                  const float* __restrict__ x_scales,
                                  float* __restrict__ y,
                                  int M, int N) {
    int warp_id = threadIdx.x / WarpSize;
    int lane    = threadIdx.x % WarpSize;
    int row     = blockIdx.x * (blockDim.x / WarpSize) + warp_id;
    if (row >= M) return;

    int num_blocks = N / kQuantBlockSize;
    float partial = 0.0f;

    // Each lane owns one full 32-element quantization block per iteration.
    for (int b = lane; b < num_blocks; b += WarpSize) {
        const int8_t* A_block = A_q + static_cast<size_t>(row) * N + b * kQuantBlockSize;
        const int8_t* x_block = x_q + b * kQuantBlockSize;

        // Reinterpret 4 contiguous int8s as one packed int32 — that's how
        // sdot4 gets "4 int8s": you don't build an array, you just read
        // 4 bytes as one 32-bit word. kQuantBlockSize=32 bytes / 4 = 8 words.
        const int32_t* A_packed = reinterpret_cast<const int32_t*>(A_block);
        const int32_t* x_packed = reinterpret_cast<const int32_t*>(x_block);

        int32_t dot = 0;  // MUST be int32 — int8 overflows after ~2 terms
        #pragma unroll
        for (int k = 0; k < kQuantBlockSize / 4; ++k) {
            dot = __builtin_amdgcn_sdot4(A_packed[k], x_packed[k], dot, false);
        }

        // Dequantize once per block, then accumulate into the running fp32 sum.
        partial += static_cast<float>(dot) * A_scales[row * num_blocks + b] * x_scales[b];
    }

    for (int offset = WarpSize / 2; offset > 0; offset /= 2)
        partial += __shfl_down(partial, offset);

    if (lane == 0) y[row] = partial;
}

} // namespace

void gemv_q8_0(const int8_t* A_q, const int8_t* x_q, 
                float* y, const float* A_scales,
                const float* x_scales,
                int M, int N, hipStream_t stream) {
    constexpr int kThreadsPerBlock = 1024;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / WarpSize;

    dim3 block(kThreadsPerBlock);
    dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);

    gemv_q8_0_kernel<<<grid, block, 0, stream>>>(A_q, A_scales, x_q, x_scales, y, M, N);
    HIP_CHECK(hipGetLastError());
}