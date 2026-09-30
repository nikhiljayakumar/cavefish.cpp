#include "gemv_q8_0.h"
#include "quantize.hpp"
#include "common.hpp"
#include "benchmark.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

struct Shape { const char* name; int M; int N; };
static constexpr Shape kShapes[] = {
    {"q_proj",     4096,   4096}, {"k_proj",     1024,   4096},
    {"v_proj",     1024,   4096}, {"o_proj",     4096,   4096},
    {"gate_proj",  14336,  4096}, {"up_proj",    14336,  4096},
    {"down_proj",  4096,   14336}, {"lm_head",    128256, 4096},
};

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& shape : kShapes) {
        int num_blocks = shape.N / kQuantBlockSize;
        size_t A_q_bytes = static_cast<size_t>(shape.M) * shape.N;
        size_t A_scale_bytes = static_cast<size_t>(shape.M) * num_blocks * sizeof(float);
        size_t total_bytes = A_q_bytes + A_scale_bytes;   // what the kernel actually reads

        std::vector<float> A_f(static_cast<size_t>(shape.M) * shape.N), x_f(shape.N);
        for (auto& v : A_f) v = dist(rng);
        for (auto& v : x_f) v = dist(rng);

        std::vector<int8_t> A_q(A_q_bytes);
        std::vector<float> A_scales(static_cast<size_t>(shape.M) * num_blocks);
        for (int row = 0; row < shape.M; ++row) {
            quantize_q8_0(A_f.data() + static_cast<size_t>(row) * shape.N, shape.N,
                          A_q.data() + static_cast<size_t>(row) * shape.N,
                          A_scales.data() + static_cast<size_t>(row) * num_blocks);
        }
        std::vector<int8_t> x_q(shape.N);
        std::vector<float> x_scales(num_blocks);
        quantize_q8_0(x_f.data(), shape.N, x_q.data(), x_scales.data());

        int8_t *d_Aq, *d_xq;
        float *d_Ascales, *d_xscales, *d_y;
        HIP_CHECK(hipMalloc(&d_Aq, A_q.size()));
        HIP_CHECK(hipMalloc(&d_xq, x_q.size()));
        HIP_CHECK(hipMalloc(&d_Ascales, A_scales.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_xscales, x_scales.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_y, shape.M * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_Aq, A_q.data(), A_q.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_xq, x_q.data(), x_q.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_Ascales, A_scales.data(), A_scales.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_xscales, x_scales.data(), x_scales.size() * sizeof(float), hipMemcpyHostToDevice));

        auto result = run_benchmark([&]() {
            gemv_q8_0(d_Aq, d_xq, d_y, d_Ascales, d_xscales, shape.M, shape.N);
            HIP_CHECK(hipDeviceSynchronize());
        });

        double gb_per_sec = (total_bytes / 1e9) / (result.avg_time_ms / 1e3);
        std::printf("%-12s M=%-7d N=%-7d  %.1f GB/s  %.1f us/call\n",
                    shape.name, shape.M, shape.N, gb_per_sec, result.avg_time_ms * 1e3);

        HIP_CHECK(hipFree(d_Aq)); HIP_CHECK(hipFree(d_xq));
        HIP_CHECK(hipFree(d_Ascales)); HIP_CHECK(hipFree(d_xscales)); HIP_CHECK(hipFree(d_y));
    }
    return 0;
}