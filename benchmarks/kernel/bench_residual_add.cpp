// benchmarks/kernel/bench_residual_add.cpp
#include "residual_add.h"
#include "common.hpp"
#include "benchmark.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

int main() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr int N = 14336;
    std::vector<float> x(N), residual(N);
    for (auto& v : x) v = dist(rng);
    for (auto& v : residual) v = dist(rng);

    float *d_x, *d_residual, *d_y;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_residual, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, x.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_residual, residual.data(), N * sizeof(float), hipMemcpyHostToDevice));

    auto result = run_benchmark([&]() {
        residual_add(d_x, d_residual, d_y, N, 0);
        HIP_CHECK(hipDeviceSynchronize());
    });

    std::printf("residual_add N=%-7d  %.2f us/call\n", N, result.avg_time_ms * 1e3);

    HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_residual)); HIP_CHECK(hipFree(d_y));
    return 0;
}