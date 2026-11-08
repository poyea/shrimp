#pragma once

// GPT-2's architecture: learned absolute positions, LayerNorm, GELU, a bias on
// every projection, output weights tied to the embedding, hidden = 4 * dim.
//
// The four published shapes are here as configs, not as models: `parameters()`
// is arithmetic on the config, while instantiating one needs the weights in
// memory during constant evaluation. `config` is the shape that fits.

#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace poyea::shrimp::gpt2 {

[[nodiscard]] constexpr auto shape(std::size_t vocab, std::size_t dim, std::size_t heads,
                                   std::size_t layers,
                                   std::size_t max_seq) noexcept -> ModelConfig {
    return ModelConfig{
        .vocab = vocab,
        .dim = dim,
        .heads = heads,
        .kv_heads = heads,
        .layers = layers,
        .max_seq = max_seq,
        .hidden = 4 * dim,
        .norm = Norm::Layer,
        .act = Act::GELU,
        .pos = Pos::Learned,
        .bias = true,
        .tie_embeddings = true,
        .norm_eps = 1e-5,
    };
}

inline constexpr std::size_t bpe_vocab = 50257;

inline constexpr ModelConfig small = shape(bpe_vocab, 768, 12, 12, 1024);
inline constexpr ModelConfig medium = shape(bpe_vocab, 1024, 16, 24, 1024);
inline constexpr ModelConfig large = shape(bpe_vocab, 1280, 20, 36, 1024);
inline constexpr ModelConfig xl = shape(bpe_vocab, 1600, 25, 48, 1024);

// Same architecture inside the constant-evaluation budget.
inline constexpr ModelConfig config = shape(16, 16, 2, 2, 8);

} // namespace poyea::shrimp::gpt2
