#pragma once

#include <hip/hip_runtime.h>

namespace cavefish{

class RopeTable {
public:
    explicit RopeTable(int max_ctx);
    RopeTable(const RopeTable& other) = delete;
    RopeTable& operator=(const RopeTable& other) = delete;
    RopeTable(RopeTable&& other) noexcept ;
    RopeTable& operator=(RopeTable&& other) noexcept;
    ~RopeTable();
    [[nodiscard]] std::size_t bytes() const noexcept { return static_cast<size_t>(max_ctx_ * 64) * sizeof(float2); }

private:
    int max_ctx_;
    void release() noexcept;
    float2* base_ = nullptr;
    // Llama 3.1 RoPE frequency for pair i (0..63), with its long-context scaling.
};

double llama31_freq(int i);
} // namespace cavefish