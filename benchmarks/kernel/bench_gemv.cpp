#include "gemv.h"
#include "benchmark.hpp"
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

struct Shape { const char* name; int M; int N; };

static constexpr Shape kShapes[] = {
    {"q_proj",     4096,   4096},
    {"k_proj",     1024,   4096},
    {"v_proj",     1024,   4096},
    {"o_proj",     4096,   4096},
    {"gate_proj",  14336,  4096},
    {"up_proj",    14336,  4096},
    {"down_proj",  4096,   14336},
    {"lm_head",    128256, 4096},
};

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& shape : kShapes) {
        size_t A_bytes = static_cast<size_t>(shape.M) * shape.N * sizeof(float);

        std::vector<float> A(static_cast<size_t>(shape.M) * shape.N);
        std::vector<float> x(shape.N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : x) v = dist(rng);

        // benchmarks/kernel/bench_gemv.cpp — inside the loop
        float *d_A, *d_x, *d_y;
        HIP_CHECK(hipMalloc(&d_A, A_bytes));
        HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_y, shape.M * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_A, A.data(), A_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));

        auto result = run_benchmark([&]() {
            gemv(d_A, d_x, d_y, shape.M, shape.N);
            HIP_CHECK(hipGetLastError());
        });
        HIP_CHECK(hipDeviceSynchronize());


        double gb_per_sec = (A_bytes / 1e9) / (result.avg_time_ms / 1e3);
        std::printf("%-12s M=%-7d N=%-7d  %.1f GB/s\n",
                    shape.name, shape.M, shape.N, gb_per_sec);

        HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_y));
    }
    return 0;
}