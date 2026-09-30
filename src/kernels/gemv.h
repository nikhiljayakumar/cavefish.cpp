#include <hip/hip_runtime.h>
void gemv(const float* A, const float* x, float* y, int M, int N, hipStream_t stream = 0);