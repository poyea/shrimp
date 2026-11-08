#pragma once

// Two-layer induction head, weights derived in closed form.
// Given ...A B ... A it predicts B.
//
// Layout, dim == 2 * vocab:  tok [0, vocab)   prev [vocab, 2*vocab)
//
// omega_i = theta^(-2i/dim), so a large theta leaves low pairs position
// sensitive and high pairs position invariant across the window.
// L0 uses low pairs  -> matches position -> previous-token head.
// L1 uses high pairs -> matches content  -> induction.
//
// Token 0 is BOS and must not appear in content: prev(0) is a self-copy that
// otherwise matches a query for tok(0) and splits L1's attention.

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace poyea::shrimp::induction {

inline constexpr ModelConfig config{
    .vocab = 8,
    .dim = 16,
    .heads = 1,
    .layers = 2,
    .max_seq = 12,
    .hidden = 1, // FFN is zeroed; cheapest legal width
    .rope_theta = 1e6,
    .norm_eps = 1e-5,
};

inline constexpr std::size_t bos = 0;

namespace detail {

inline constexpr double embed_gain = 1.0;
inline constexpr double prev_gain = 1.0;
inline constexpr double copy_gain = 10.0; // must dominate embed_gain
inline constexpr double score_gain = 14.5;
inline constexpr std::size_t low_pairs = 3;

template <typename T>
[[nodiscard]] constexpr auto omega(std::size_t i, ModelConfig c) noexcept -> T {
    T const exponent = static_cast<T>(2 * i) / static_cast<T>(c.head_dim());
    return T{1} / cmath::pow(static_cast<T>(c.rope_theta), exponent);
}

[[nodiscard]] constexpr auto high_begin(ModelConfig c) noexcept -> std::size_t {
    return (c.head_dim() / 2) - ((c.vocab + 1) / 2);
}

[[nodiscard]] constexpr auto match_slot(std::size_t v, ModelConfig c) noexcept
    -> std::size_t {
    std::size_t const half = c.head_dim() / 2;
    std::size_t const pairs = (c.vocab + 1) / 2;
    return v < pairs ? high_begin(c) + v : half + high_begin(c) + (v - pairs);
}

template <typename T>
[[nodiscard]] constexpr auto position_kernel(long delta, ModelConfig c) noexcept -> T {
    T sum{};
    for (std::size_t i = 0; i < low_pairs; ++i) {
        sum += cmath::cos(static_cast<T>(delta) * omega<T>(i, c));
    }
    return sum;
}

} // namespace detail

template <typename T>
[[nodiscard]] constexpr auto construction_is_sound(ModelConfig c) noexcept -> bool {
    if (!c.valid() || c.dim != 2 * c.vocab || c.heads != 1 || c.layers != 2) {
        return false;
    }
    if (c.kv_head_count() != c.heads || c.norm != Norm::RMS || c.act != Act::SwiGLU) {
        return false;
    }
    if (detail::high_begin(c) <= detail::low_pairs) {
        return false;
    }
    T const slowest = detail::omega<T>(detail::high_begin(c), c);
    if (cmath::cos(static_cast<T>(c.max_seq) * slowest) < T{0.999}) {
        return false;
    }
    T const gain = static_cast<T>(detail::score_gain * detail::score_gain) /
                   cmath::sqrt(static_cast<T>(c.head_dim()));
    T const gap = detail::position_kernel<T>(0, c) - detail::position_kernel<T>(1, c);
    return gain * gap > T{20};
}

template <typename T, ModelConfig C = config>
    requires(construction_is_sound<T>(C))
[[nodiscard]] constexpr auto build() -> Transformer<T, C> {
    constexpr std::size_t vocab = C.vocab;
    constexpr std::size_t dim = C.dim;
    constexpr std::size_t half = C.head_dim() / 2;

    Transformer<T, C> model{};

    for (std::size_t v = 0; v < vocab; ++v) {
        model.embedding[v, v] = static_cast<T>(detail::embed_gain);
    }

    // L0. RMSNorm maps the embedding to sqrt(dim) in one slot; cancel it.
    T const norm0 = cmath::sqrt(static_cast<T>(dim));

    auto& l0 = model.blocks[0].attn;
    for (std::size_t i = 0; i < detail::low_pairs; ++i) {
        T const w = detail::omega<T>(i, C);
        // Query pre-rotated by -1 so rope at t yields R(t-1), peaking at j=t-1.
        T const q_lo = cmath::cos(w);
        T const q_hi = -cmath::sin(w);
        for (std::size_t v = 0; v < vocab; ++v) {
            l0.wq[i, v] = static_cast<T>(detail::score_gain) * q_lo / norm0;
            l0.wq[i + half, v] = static_cast<T>(detail::score_gain) * q_hi / norm0;
            l0.wk[i, v] = static_cast<T>(detail::score_gain) / norm0;
        }
    }
    for (std::size_t v = 0; v < vocab; ++v) {
        l0.wv[v, v] = T{1} / norm0;
        l0.wo[vocab + v, v] = static_cast<T>(detail::prev_gain);
    }

    // L1.
    T const scale1 =
        T{1} / cmath::sqrt(((static_cast<T>(detail::embed_gain * detail::embed_gain) +
                             static_cast<T>(detail::prev_gain * detail::prev_gain)) /
                            static_cast<T>(dim)) +
                           static_cast<T>(C.norm_eps));
    T const tok_mag = static_cast<T>(detail::embed_gain) * scale1;
    T const prev_mag = static_cast<T>(detail::prev_gain) * scale1;

    auto& l1 = model.blocks[1].attn;
    for (std::size_t v = 0; v < vocab; ++v) {
        std::size_t const m = detail::match_slot(v, C);
        l1.wq[m, v] = static_cast<T>(detail::score_gain) / tok_mag;
        l1.wk[m, vocab + v] = static_cast<T>(detail::score_gain) / prev_mag;
        l1.wv[v, v] = T{1} / tok_mag;
        l1.wo[v, v] = static_cast<T>(detail::copy_gain);
    }

    for (std::size_t v = 0; v < vocab; ++v) {
        model.lm_head[v, v] = T{1};
    }

    return model;
}

} // namespace poyea::shrimp::induction
