// tests/test_residual_add.cpp
#include "residual_add.h"
#include "common.hpp"
#include "verify.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

static bool run_case(int N) {
    std::vector<float> x(N), residual(N), y_ref(N), y_gpu(N);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : residual) v = dist(rng);
    for (int i = 0; i < N; ++i) y_ref[i] = x[i] + residual[i];

    float *d_x, *d_residual, *d_y;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_residual, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, x.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_residual, residual.data(), N * sizeof(float), hipMemcpyHostToDevice));

    residual_add(d_x, d_residual, d_y, N, 0);
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_residual)); HIP_CHECK(hipFree(d_y));

    // Exact-add, no reduction involved, so tolerance can be tight.
    bool ok = verify_gemv(y_gpu.data(), y_ref.data(), N, 1e-5f, 1e-6f);
    std::printf("[%s] N=%d\n", ok ? "PASS" : "FAIL", N);
    return ok;
}

int main() {
    bool all_ok = true;
    all_ok &= run_case(1);       // single element edge case
    all_ok &= run_case(4096);    // hidden_size
    all_ok &= run_case(14336);   // intermediate_size (FFN residual)
    return all_ok ? 0 : 1;
}