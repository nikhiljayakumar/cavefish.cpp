#include "gemv.h"
#include "common.hpp"
#include "verify.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <cstdio>

static bool run_case(int M, int N) {
    std::vector<float> A(M * N), x(N), y_ref(M), y_gpu(M);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : x) v = dist(rng);

    cpu_gemv_reference(A.data(), x.data(), y_ref.data(), M, N);

    float *d_A, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_A, A.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, y_gpu.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, A.data(), A.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));

    gemv(d_A, d_x, d_y, M, N);
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_x)); HIP_CHECK(hipFree(d_y));

    bool ok = verify_gemv(y_gpu.data(), y_ref.data(), M);
    std::printf("[%s] M=%d N=%d\n", ok ? "PASS" : "FAIL", M, N);
    return ok;
}

int main() {
    bool all_ok = true;
    // small: trivial case, edge alignment
    all_ok &= run_case(1, 1);
    all_ok &= run_case(8, 8);
    // non-multiple-of-warp-size M, to catch bounds-check bugs
    all_ok &= run_case(33, 4096);
    // one real Llama shape, kept small enough to run fast every time
    all_ok &= run_case(1024, 4096);   // k_proj shape
    return all_ok ? 0 : 1;
}