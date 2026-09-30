#include <hip/hip_runtime.h>
void gemv_q8_0(const int8_t* __restrict__ A_q,
                const int8_t* __restrict__ x_q,
                float* __restrict__ y,
                const float* __restrict__ A_scales,
                const float* __restrict__ x_scales,
                int M,
                int N,
                hipStream_t stream = 0);