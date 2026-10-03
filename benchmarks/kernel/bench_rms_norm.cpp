// benchmarks/kernel/bench_rms_norm.cpp
#include "rms_norm.h"
#include "common.hpp"
#include "benchmark.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr int N = 4096;  // Llama 3.1 8B hidden_size
    std::vector<float> x(N), weight(N);
    for (auto& v : x) v = dist(rng);
    for (auto& v : weight) v = dist(rng);

    float *d_x, *d_weight, *d_y;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, x.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, weight.data(), N * sizeof(float), hipMemcpyHostToDevice));

    const float eps = 1e-5f;
    auto result = run_benchmark([&]() {
        rms_norm(d_x, d_weight, d_y, eps, N, 0);
        HIP_CHECK(hipDeviceSynchronize());
    });

    std::printf("rms_norm    N=%-7d  %.2f us/call\n", N, result.avg_time_ms * 1e3);

    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_weight)); HIP_CHECK(hipFree(d_y));
    return 0;
}