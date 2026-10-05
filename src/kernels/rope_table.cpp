#include "rope_table.hpp"
#include "common.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace cavefish { 

RopeTable::RopeTable(int max_ctx) : max_ctx_(max_ctx)
{
    void* p = nullptr;
    hipError_t err = hipMalloc(&p, bytes());
    if (err != hipSuccess){
        throw std::runtime_error("RopeTable: hipMalloc of " + std::to_string(bytes() >> 20) +
                                 " MiB failed: " + hipGetErrorString(err));
    }
    base_ = static_cast<float2*>(p);

    std::vector<float2> host(static_cast<std::size_t>(max_ctx) * 64);
    for (int i = 0; i < 64; i++){
        double f = cavefish::llama31_freq(i);
        for (int pos = 0; pos < max_ctx_; pos++){
            double a = pos * f;
            host[pos*64 + i] = {float(std::cos(a)), float(std::sin(a))};
        }
    }
    HIP_CHECK(hipMemcpy(base_, host.data(), bytes(), hipMemcpyHostToDevice));
}
RopeTable::RopeTable(RopeTable&& other) noexcept : max_ctx_(std::exchange(other.max_ctx_, 0)), base_(std::exchange(other.base_, nullptr))
{}
RopeTable& RopeTable::operator=(RopeTable&& other) noexcept
{
    if (this != &other){
        release();
        max_ctx_ = std::exchange(other.max_ctx_, 0);
        base_ = std::exchange(other.base_, nullptr);
    }
    return *this;

}
RopeTable::~RopeTable()
{
    release();
}
void RopeTable::release() noexcept
{
    const hipError_t err = hipFree(base_);
    base_ = nullptr;
}

double llama31_freq(int i) {
    constexpr double kBase = 500000.0, kFactor = 8.0, kLow = 1.0, kHigh = 4.0, kOrigCtx = 8192.0;
    constexpr double kPi = 3.14159265358979323846;
    const double f  = std::pow(kBase, -2.0 * i / 128.0);
    const double wl = 2.0 * kPi / f;
    if (wl < kOrigCtx / kHigh) return f;              // < 2048: unchanged
    if (wl > kOrigCtx / kLow)  return f / kFactor;    // > 8192: scaled down 8x
    const double s = (kOrigCtx / wl - kLow) / (kHigh - kLow);
    return (1.0 - s) * f / kFactor + s * f;           // blend in between
}
}