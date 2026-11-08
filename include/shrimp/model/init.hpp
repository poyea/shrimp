#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/random.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>
#include <cstdint>

namespace poyea::shrimp {

template <typename T, ModelConfig C>
[[nodiscard]] constexpr auto random_transformer(std::uint64_t seed) -> Transformer<T, C> {
    Transformer<T, C> model{};
    SplitMix64 rng{seed};

    T const embed_sigma = T{1} / cmath::sqrt(static_cast<T>(C.dim));
    T const attn_sigma = T{1} / cmath::sqrt(static_cast<T>(C.dim));
    T const hidden_sigma = T{1} / cmath::sqrt(static_cast<T>(C.hidden));

    fill_normal(model.embedding, rng, embed_sigma);
    if constexpr (C.pos == Pos::Learned) {
        fill_normal(model.positions.value, rng, embed_sigma);
    }

    for (std::size_t layer = 0; layer < C.layers; ++layer) {
        auto& block = model.blocks[layer];
        fill_normal(block.attn.wq, rng, attn_sigma);
        fill_normal(block.attn.wk, rng, attn_sigma);
        fill_normal(block.attn.wv, rng, attn_sigma);
        fill_normal(block.attn.wo, rng, attn_sigma);
        if constexpr (C.act == Act::SwiGLU) {
            fill_normal(block.ffn.gate, rng, attn_sigma);
        }
        fill_normal(block.ffn.up, rng, attn_sigma);
        fill_normal(block.ffn.down, rng, hidden_sigma);
    }

    // Biases stay at zero, as GPT-2 initialises them.
    if constexpr (!C.tie_embeddings) {
        fill_normal(model.lm_head.value, rng, embed_sigma);
    }
    return model;
}

} // namespace poyea::shrimp