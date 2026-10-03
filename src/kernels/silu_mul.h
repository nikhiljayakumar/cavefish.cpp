#include <hip/hip_runtime.h>

void silu_mul(const float* gate, const float* up, float* y, int N, hipStream_t stream = 0);