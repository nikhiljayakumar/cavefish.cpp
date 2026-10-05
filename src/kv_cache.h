#pragma once

#include "kernels/kv_view.hpp"
#include <cstddef>

namespace cavefish {

// Owns the KV cache: one device allocation, K slab followed by V slab.
// Non-copyable, movable. Kernels never see this type, only view().
class KVCache {
public:
    KVCache(int n_layers, int max_ctx);
    ~KVCache();

    KVCache(const KVCache&)            = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;

    [[nodiscard]] KVView view() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept { return 2 * slab_elems_ * sizeof(__half); }
    [[nodiscard]] int max_ctx() const noexcept { return max_ctx_; }

private:
    void release() noexcept;

    __half* base_ = nullptr;
    std::size_t slab_elems_ = 0;  // elements in ONE slab (K or V)
    int n_layers_ = 0;
    int max_ctx_ = 0;
};

} // namespace cavefish
