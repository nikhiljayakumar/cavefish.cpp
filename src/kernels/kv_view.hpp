#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstddef>

namespace cavefish {

// These are set for Llama 3.1B 
inline constexpr int kNumKVHeads = 8;
// Size of each K/V head/vector
inline constexpr int kHeadDim    = 128;

// Non-owning view of the KV cache. Plain data so it can be passed to kernels by value.
// Layout of each slab: [layer][kv_head][pos][dim], fp16.
struct KVView {
    __half* k;
    __half* v;
    int n_layers;
    int max_ctx;
};

// Element offset of row (layer, kv_head, pos) inside the K or V slab.
// The only place the layout math lives; kernels and tests both use it.
__host__ __device__ inline std::size_t kv_offset(int layer, int kv_head, int pos, int max_ctx) {
    return ((static_cast<std::size_t>(layer) * kNumKVHeads + kv_head) * max_ctx + pos) * kHeadDim;
}

} // namespace cavefish