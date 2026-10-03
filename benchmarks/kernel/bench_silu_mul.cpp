#include "silu_mul.h"
#include "common.hpp"
#include "benchmark.hpp"
#include <cstdio>
#include <vector>
#include <random>

int main() {
    const int N = 14336;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> h_gate(N), h_up(N);
    for (int i = 0; i < N; ++i) {
        h_gate[i] = dist(rng);
        h_up[i] = dist(rng);
    }

    float *d_gate, *d_up, *d_y;
    HIP_CHECK(hipMalloc(&d_gate, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_up, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up, h_up.data(), N * sizeof(float), hipMemcpyHostToDevice));

    auto result = run_benchmark([&]() {
        silu_mul(d_gate, d_up, d_y, N);
        HIP_CHECK(hipDeviceSynchronize());
    });
    std::printf("silu_mul    N=%-7d  %.2f us/call\n", N, result.avg_time_ms * 1e3);

    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_y));
    return 0;
}