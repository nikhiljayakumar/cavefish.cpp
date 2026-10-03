#include "silu_mul.h"
#include "common.hpp"
#include "verify.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

float cpu_silu_mul_ref(float gate, float up) {
    double g = static_cast<double>(gate);
    double silu = g / (1.0 + std::exp(-g));
    return static_cast<float>(silu * static_cast<double>(up));
}

bool run_case(int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> h_gate(N), h_up(N), h_ref(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        h_gate[i] = dist(rng);
        h_up[i] = dist(rng);
        h_ref[i] = cpu_silu_mul_ref(h_gate[i], h_up[i]);
    }

    float *d_gate, *d_up, *d_y;
    HIP_CHECK(hipMalloc(&d_gate, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_up, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up, h_up.data(), N * sizeof(float), hipMemcpyHostToDevice));

    silu_mul(d_gate, d_up, d_y, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_y.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    bool ok = verify_gemv(h_y.data(), h_ref.data(), 1e-4f, 1e-5f);

    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_y));

    std::printf("silu_mul N=%-7d %s\n", N, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool all_pass = true;
    all_pass &= run_case(1);
    all_pass &= run_case(14336);
    return all_pass ? 0 : 1;
}