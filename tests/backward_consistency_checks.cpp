#include "shrimp/core/math.hpp"
#include "shrimp/model/attn_backward.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Matrix;
using poyea::shrimp::ModelConfig;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr ModelConfig tiny{
    .vocab = 4, .dim = 4, .heads = 1, .layers = 1, .max_seq = 3, .hidden = 4};

constexpr ModelConfig gqa{
    .vocab = 4, .dim = 8, .heads = 4, .kv_heads = 2, .layers = 1, .max_seq = 3, .hidden = 4};

template <ModelConfig C> constexpr auto seeded_stream() -> Matrix<double, C.max_seq, C.dim> {
    Matrix<double, C.max_seq, C.dim> s{};
    for (std::size_t i = 0; i < decltype(s)::size; ++i) {
        s.storage[i] =
            0.31 - (0.17 * static_cast<double>(i % 5)) + (0.09 * static_cast<double>(i % 3));
    }
    return s;
}

// attend and attention_tape are two transcriptions of one forward pass. If they
// ever disagree the backward is differentiating something the model does not do,
// and every finite-difference test would still pass.
template <ModelConfig C> constexpr auto tape_reproduces_attend(std::size_t count) -> bool {
    auto const model = poyea::shrimp::random_transformer<double, C>(5);
    auto const& w = model.blocks[0].attn;
    auto const& norm = model.blocks[0].attn_norm;
    auto const before = seeded_stream<C>();

    auto after = before;
    poyea::shrimp::attend<double, C>(w, after, norm, count);

    auto const tape = poyea::shrimp::detail::attention_tape<double, C>(w, before, norm, count);

    for (std::size_t t = 0; t < count; ++t) {
        auto const projected =
            poyea::shrimp::ops::linear(w.wo, poyea::shrimp::row(tape.context, t));
        for (std::size_t i = 0; i < C.dim; ++i) {
            double const expected = before.storage[(t * C.dim) + i] + projected.storage[i];
            if (!near(after.storage[(t * C.dim) + i], expected, 1e-15)) {
                return false;
            }
        }
    }
    return true;
}

static_assert(tape_reproduces_attend<tiny>(3));
static_assert(tape_reproduces_attend<tiny>(2));
static_assert(tape_reproduces_attend<tiny>(1));
static_assert(tape_reproduces_attend<gqa>(3));
static_assert(tape_reproduces_attend<gqa>(2));

// The saved attention weights must be a distribution over the causal prefix.
template <ModelConfig C> constexpr auto tape_weights_are_normalised(std::size_t count) -> bool {
    auto const model = poyea::shrimp::random_transformer<double, C>(5);
    auto const before = seeded_stream<C>();
    auto const tape = poyea::shrimp::detail::attention_tape<double, C>(
        model.blocks[0].attn, before, model.blocks[0].attn_norm, count);

    for (std::size_t t = 0; t < count; ++t) {
        for (std::size_t h = 0; h < C.heads; ++h) {
            std::size_t const base = (t * C.heads * C.max_seq) + (h * C.max_seq);
            double total = 0.0;
            for (std::size_t j = 0; j <= t; ++j) {
                if (tape.weights.storage[base + j] < 0.0) {
                    return false;
                }
                total += tape.weights.storage[base + j];
            }
            if (!near(total, 1.0, 1e-13)) {
                return false;
            }
            for (std::size_t j = t + 1; j < C.max_seq; ++j) {
                if (tape.weights.storage[base + j] != 0.0) {
                    return false;
                }
            }
        }
    }
    return true;
}

static_assert(tape_weights_are_normalised<tiny>(3));
static_assert(tape_weights_are_normalised<gqa>(3));

} // namespace

auto main() -> int {
    return 0;
}
