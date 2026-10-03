#include <hip/hip_runtime.h>
void rms_norm(const float* x, const float* weight, float* y, float eps, int N, hipStream_t stream);