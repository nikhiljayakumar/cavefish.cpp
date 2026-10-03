#include "rms_norm.h"
#include "common.hpp"

namespace {
    __global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict weight, float* __restrict y, const float eps, const int N){
        // since we know 32 warps per block, and we only dispatching one block
        // we define array size to be 32
        __shared__ float s_partial[32];
        // rms(x)
        __shared__ float s_inv_rms;

        int tid = threadIdx.x;
        int warp_id = threadIdx.x / kWarpSize;
        int lane = threadIdx.x % kWarpSize;
        int num_warps = blockDim.x / 32;

        float sum_sq = 0.0f;
        float x_b;
        for (int b = tid; b < N; b += blockDim.x){
            x_b = x[b];
            sum_sq += x_b * x_b;
        }

        // reduce to lane 0 of each warp
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2){
            sum_sq += __shfl_down(sum_sq, offset);
            
        }

        if (lane == 0) 
            s_partial[warp_id] = sum_sq;

        __syncthreads();
        
    
        if (warp_id == 0){
            float v = (lane < num_warps) ? s_partial[lane] : 0.0f;
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2){
                v += __shfl_down(v, offset);
            }
            if (lane == 0) 
                s_inv_rms = rsqrtf((v / N) + eps);
        }
        __syncthreads();

        float inv_rms = 0.0f;
        for (int b = tid; b < N; b += blockDim.x){
            y[b] = x[b] * s_inv_rms * weight[b];
        }
    }
}
void rms_norm(const float* x, const float* weight, float* y, float eps, int N, hipStream_t stream){
    constexpr int kThreadsPerBlock = 1024;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

    dim3 block(kThreadsPerBlock);
    dim3 grid(1);

    rms_norm_kernel<<<grid, block, 0, stream>>>(x, weight, y, eps, N);
}