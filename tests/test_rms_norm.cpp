// tests/test_rms_norm.cpp
#include "rms_norm.h"
#include "common.hpp"
#include "verify.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

static void cpu_rmsnorm_reference(const float* x, const float* weight, float* y_ref, int N, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) sum_sq += static_cast<double>(x[i]) * x[i];
    double inv_rms = 1.0 / std::sqrt(sum_sq / N + eps);
    for (int i = 0; i < N; ++i) y_ref[i] = static_cast<float>(x[i] * inv_rms * weight[i]);
}

static bool run_case(int N) {
    std::vector<float> x(N), weight(N), y_ref(N), y_gpu(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : weight) v = dist(rng);

    const float eps = 1e-5f;
    cpu_rmsnorm_reference(x.data(), weight.data(), y_ref.data(), N, eps);

    float *d_x, *d_weight, *d_y;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, x.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, weight.data(), N * sizeof(float), hipMemcpyHostToDevice));

    rms_norm(d_x, d_weight, d_y, eps, N, 0);
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_weight)); HIP_CHECK(hipFree(d_y));

    bool ok = verify_gemv(y_gpu.data(), y_ref.data(), N, /*rel_tol=*/1e-4f, /*abs_tol=*/1e-5f);
    std::printf("[%s] N=%d\n", ok ? "PASS" : "FAIL", N);
    return ok;
}

int main() {
    bool all_ok = true;
    all_ok &= run_case(32);    // exactly one warp's width — tests the single-warp path
    all_ok &= run_case(4096);  // real Llama hidden_size
    return all_ok ? 0 : 1;
}