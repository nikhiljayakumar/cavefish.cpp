#pragma once
#include "common.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>

// Symmetric int8 quantization, GGUF Q8_0-style: one scale per 32-element
// block, no zero point. `n` must be a multiple of kQuantBlockSize.
inline void quantize_q8_0(const float* src, int n, int8_t* out, float* scales) {
    int num_blocks = n / kQuantBlockSize;
    for (int b = 0; b < num_blocks; ++b) {
        const float* block = src + b * kQuantBlockSize;
        float max_abs = 0.0f;
        for (int i = 0; i < kQuantBlockSize; ++i) max_abs = std::max(max_abs, std::fabs(block[i]));
        float scale = max_abs / 127.0f;
        scales[b] = scale;

        int8_t* out_block = out + b * kQuantBlockSize;
        for (int i = 0; i < kQuantBlockSize; ++i) {
            float q = (scale > 0.0f) ? std::round(block[i] / scale) : 0.0f;
            q = std::clamp(q, -127.0f, 127.0f);
            out_block[i] = static_cast<int8_t>(q);
        }
    }
}