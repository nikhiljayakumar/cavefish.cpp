#include "gemv_q8_0.h"
#include "quantize.hpp"
#include "verify.hpp"
#include "common.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

static bool run_case(int M, int N) {
    int num_blocks = N / kQuantBlockSize;

    std::vector<float> A_f(static_cast<size_t>(M) * N), x_f(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A_f) v = dist(rng);
    for (auto& v : x_f) v = dist(rng);

    std::vector<int8_t> A_q(static_cast<size_t>(M) * N);
    std::vector<float> A_scales(static_cast<size_t>(M) * num_blocks);
    for (int row = 0; row < M; ++row) {
        quantize_q8_0(A_f.data() + static_cast<size_t>(row) * N, N,
                      A_q.data() + static_cast<size_t>(row) * N,
                      A_scales.data() + static_cast<size_t>(row) * num_blocks);
    }
    std::vector<int8_t> x_q(N);
    std::vector<float> x_scales(num_blocks);
    quantize_q8_0(x_f.data(), N, x_q.data(), x_scales.data());

    std::vector<float> y_ref(M), y_gpu(M);
    cpu_gemv_q8_0_reference(A_q.data(), A_scales.data(), x_q.data(), x_scales.data(), y_ref.data(), M, N);

    int8_t *d_Aq, *d_xq;
    float *d_Ascales, *d_xscales, *d_y;
    HIP_CHECK(hipMalloc(&d_Aq, A_q.size()));
    HIP_CHECK(hipMalloc(&d_xq, x_q.size()));
    HIP_CHECK(hipMalloc(&d_Ascales, A_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_xscales, x_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, y_gpu.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_Aq, A_q.data(), A_q.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xq, x_q.data(), x_q.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Ascales, A_scales.data(), A_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xscales, x_scales.data(), x_scales.size() * sizeof(float), hipMemcpyHostToDevice));

    gemv_q8_0(d_Aq, d_xq, d_y, d_Ascales, d_xscales, M, N);
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_Aq)); HIP_CHECK(hipFree(d_xq));
    HIP_CHECK(hipFree(d_Ascales)); HIP_CHECK(hipFree(d_xscales)); HIP_CHECK(hipFree(d_y));

    // Reference used the same quantized data — tolerance can be tight, this
    // isn't testing quantization noise, it's testing the kernel's math.
    bool ok = verify_gemv(y_gpu.data(), y_ref.data(), M, /*rel_tol=*/1e-4f, /*abs_tol=*/1e-4f);
    std::printf("[%s] M=%d N=%d\n", ok ? "PASS" : "FAIL", M, N);
    return ok;
}

int main() {
    bool all_ok = true;
    all_ok &= run_case(32, 32);      // exactly one block, one lane does real work
    all_ok &= run_case(64, 64);      // two blocks per row
    all_ok &= run_case(33, 4096);    // M not a multiple of warps/block
    all_ok &= run_case(1024, 4096);  // k_proj shape
    return all_ok ? 0 : 1;
}