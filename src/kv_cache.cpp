#include "kernels/kv_view.hpp"
#include "kv_cache.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <utility>
#include <stdexcept>

namespace cavefish {

KVCache::KVCache(int n_layers, int max_ctx) : n_layers_(n_layers), max_ctx_(max_ctx)
{ 
    slab_elems_ = static_cast<std::size_t>(n_layers_ * kNumKVHeads * max_ctx_ * kHeadDim);

    if (n_layers_ <= 0 || max_ctx_ <= 0){
        throw std::invalid_argument("KVCache: n_layers and max_ctx must be positive");
    }

    void* p = nullptr;
    const hipError_t err = hipMalloc(&p, bytes());
    if (err != hipSuccess){
        throw std::runtime_error("KVCache: hipMalloc of " + std::to_string(bytes() >> 20) +
                                 " MiB failed: " + hipGetErrorString(err));
    }
    base_ = static_cast<__half*>(p);
}

KVCache::~KVCache()
{
    release();
}

// y = std::exchange(x, 10) makes y = x and x = 10;
KVCache::KVCache(KVCache&& other) noexcept : n_layers_(std::exchange(other.n_layers_, 0)),
                           max_ctx_(std::exchange(other.max_ctx_, 0)),
                           slab_elems_(std::exchange(other.slab_elems_, 0)),
                           base_(std::exchange(other.base_, nullptr))
{}

KVCache& KVCache::operator=(KVCache&& other) noexcept
{
    if (this != &other){
        release();
        n_layers_ = std::exchange(other.n_layers_, 0);
        max_ctx_ = std::exchange(other.max_ctx_, 0);
        slab_elems_ = std::exchange(other.slab_elems_, 0);
        base_ = std::exchange(other.base_, nullptr);
    }
    return *this;
}

KVView KVCache::view() const noexcept
{
    return KVView(base_, base_ + slab_elems_, n_layers_, max_ctx_);
}

void KVCache::release() noexcept
{
    const hipError_t err = hipFree(base_);
    // can't throw from a destructor because then could cause two exceptions 
    // if (err != hipSuccess){
    //     throw std::runtime_error("KVCache: hipFree of " + std::to_string(bytes() >> 20) +
    //                              " MiB failed: " + hipGetErrorString(err));
    // }
}

} // namespace cavefish
