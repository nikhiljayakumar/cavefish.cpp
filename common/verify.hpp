#ifndef VERIFY_HPP
#define VERIFY_HPP
#include <vector>
#include <cmath>
#include <cstdio>

// Reference GEMV on CPU. Accumulate in double even though A/x/y are fp32,
// so rounding-order differences between CPU and GPU don't masquerade as bugs.
inline void cpu_gemv_reference(const float* A, const float* x, float* y_ref, int M, int N) {
    #pragma omp parallel for
    for (int row = 0; row < M; ++row) {
        double acc = 0.0;
        const float* row_ptr = A + static_cast<size_t>(row) * N;
        for (int col = 0; col < N; ++col) {
            acc += static_cast<double>(row_ptr[col]) * static_cast<double>(x[col]);
        }
        y_ref[row] = static_cast<float>(acc);
    }
}
inline void cpu_gemv_q8_0_reference(const int8_t* A_q, const float* A_scales,
                                     const int8_t* x_q, const float* x_scales,
                                     float* y_ref, int M, int N) {
    int num_blocks = N / kQuantBlockSize;
    for (int row = 0; row < M; ++row) {
        double acc = 0.0;
        for (int b = 0; b < num_blocks; ++b) {
            const int8_t* A_block = A_q + static_cast<size_t>(row) * N + b * kQuantBlockSize;
            const int8_t* x_block = x_q + b * kQuantBlockSize;
            double block_sum = 0.0;
            for (int i = 0; i < kQuantBlockSize; ++i)
                block_sum += static_cast<double>(A_block[i]) * static_cast<double>(x_block[i]);
            acc += block_sum * static_cast<double>(A_scales[row * num_blocks + b])
                              * static_cast<double>(x_scales[b]);
        }
        y_ref[row] = static_cast<float>(acc);
    }
}

// Returns true if all elements match within tolerance; prints the worst
// mismatch on failure so you don't have to hunt for it.
inline bool verify_gemv(const float* y_gpu, const float* y_ref, int M,
                         float rel_tol = 1e-3f, float abs_tol = 1e-4f) {
    int worst_idx = -1;
    float worst_err = 0.0f;
    for (int i = 0; i < M; ++i) {
        float diff = std::fabs(y_gpu[i] - y_ref[i]);
        float tol = abs_tol + rel_tol * std::fabs(y_ref[i]);
        if (diff > tol && diff > worst_err) {
            worst_err = diff;
            worst_idx = i;
        }
    }
    if (worst_idx >= 0) {
        std::fprintf(stderr, "verify_gemv: mismatch at row %d: gpu=%f ref=%f diff=%f\n",
                     worst_idx, y_gpu[worst_idx], y_ref[worst_idx], worst_err);
        return false;
    }
    return true;
}
#endif // VERIFY_HPP