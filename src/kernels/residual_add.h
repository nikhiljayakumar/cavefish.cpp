#include <hip/hip_runtime.h>

void residual_add(const float* x, const float* residual, float* y, int N, hipStream_t stream = 0);