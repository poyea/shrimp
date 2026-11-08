#include "shrimp/core/math.hpp"
#include "shrimp/core/random.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/model/ffn_backward.hpp"
#include "shrimp/model/gradients.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Act;
using poyea::shrimp::feed_forward;
using poyea::shrimp::feed_forward_backward;
using poyea::shrimp::FeedForward;
using poyea::shrimp::FeedForwardGradients;
using poyea::shrimp::fill_uniform;
using poyea::shrimp::Matrix;
using poyea::shrimp::ModelConfig;
using poyea::shrimp::Norm;
using poyea::shrimp::NormGradients;
using poyea::shrimp::NormWeights;
using poyea::shrimp::random_transformer;
using poyea::shrimp::SplitMix64;
using poyea::shrimp::Tensor;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr double h = 1e-5;
constexpr double tol = 1e-6;

constexpr ModelConfig swiglu_cfg{
    .vocab = 4, .dim = 4, .heads = 1, .layers = 1, .max_seq = 3, .hidden = 6};

constexpr ModelConfig gelu_cfg{.vocab = 4,
                               .dim = 4,
                               .heads = 1,
                               .layers = 1,
                               .max_seq = 3,
                               .hidden = 6,
                               .norm = Norm::Layer,
                               .act = Act::GELU};

template <ModelConfig C> using StreamOf = Matrix<double, C.max_seq, C.dim>;
template <ModelConfig C> using NormOf = NormWeights<double, C.dim, C.norm>;

template <ModelConfig C> struct Case {
    FeedForward<double, C> w{};
    NormOf<C> norm{};
    StreamOf<C> stream{};
    StreamOf<C> seed{};
};

template <ModelConfig C> struct Analytic {
    StreamOf<C> dstream{};
    FeedForwardGradients<double, C> dw{};
    NormGradients<double, C.dim, C.norm> dnorm{};
};

template <ModelConfig C> constexpr auto make_case() -> Case<C> {
    auto const model = random_transformer<double, C>(7);
    Case<C> fx{};
    fx.w = model.blocks[0].ffn;
    fx.norm = model.blocks[0].ffn_norm;
    for (std::size_t i = 0; i < C.dim; ++i) {
        fx.norm.gain.storage[i] = 0.8 + (0.35 * static_cast<double>(i));
    }
    if constexpr (C.norm == Norm::Layer) {
        for (std::size_t i = 0; i < C.dim; ++i) {
            fx.norm.bias.storage[i] = 0.15 - (0.2 * static_cast<double>(i));
        }
    }
    SplitMix64 stream_rng{11};
    fill_uniform(fx.stream, stream_rng, -1.0, 1.0);
    SplitMix64 seed_rng{23};
    fill_uniform(fx.seed, seed_rng, -1.0, 1.0);
    return fx;
}

template <ModelConfig C>
constexpr auto objective(FeedForward<double, C> const& w, NormOf<C> const& norm,
                         StreamOf<C> const& stream, StreamOf<C> const& seed) -> double {
    auto out = stream;
    feed_forward<double, C>(w, out, norm, C.max_seq);
    double total = 0.0;
    for (std::size_t i = 0; i < C.max_seq * C.dim; ++i) {
        total += out.storage[i] * seed.storage[i];
    }
    return total;
}

template <ModelConfig C> constexpr auto run_backward(Case<C> const& fx) -> Analytic<C> {
    Analytic<C> out{};
    out.dstream = fx.seed;
    feed_forward_backward<double, C>(fx.w, fx.stream, fx.norm, out.dstream, out.dw, out.dnorm,
                                     C.max_seq);
    return out;
}

template <std::size_t... Dims, typename F>
constexpr auto matches_finite_difference(Tensor<double, Dims...> const& base,
                                         Tensor<double, Dims...> const& analytic,
                                         F const& objective_of) -> bool {
    for (std::size_t i = 0; i < Tensor<double, Dims...>::size; ++i) {
        auto plus = base;
        auto minus = base;
        plus.storage[i] += h;
        minus.storage[i] -= h;
        double const fd = (objective_of(plus) - objective_of(minus)) / (2.0 * h);
        if (!near(analytic.storage[i], fd, tol)) {
            return false;
        }
    }
    return true;
}

template <ModelConfig C> constexpr auto down_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.w.down, r.dw.down, [&fx](auto const& p) {
        auto w = fx.w;
        w.down = p;
        return objective<C>(w, fx.norm, fx.stream, fx.seed);
    });
}

template <ModelConfig C> constexpr auto up_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.w.up, r.dw.up, [&fx](auto const& p) {
        auto w = fx.w;
        w.up = p;
        return objective<C>(w, fx.norm, fx.stream, fx.seed);
    });
}

template <ModelConfig C> constexpr auto gate_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.w.gate, r.dw.gate, [&fx](auto const& p) {
        auto w = fx.w;
        w.gate = p;
        return objective<C>(w, fx.norm, fx.stream, fx.seed);
    });
}

template <ModelConfig C> constexpr auto gain_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.norm.gain, r.dnorm.gain, [&fx](auto const& p) {
        auto norm = fx.norm;
        norm.gain = p;
        return objective<C>(fx.w, norm, fx.stream, fx.seed);
    });
}

template <ModelConfig C> constexpr auto bias_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.norm.bias, r.dnorm.bias, [&fx](auto const& p) {
        auto norm = fx.norm;
        norm.bias = p;
        return objective<C>(fx.w, norm, fx.stream, fx.seed);
    });
}

template <ModelConfig C> constexpr auto dstream_matches() -> bool {
    auto const fx = make_case<C>();
    auto const r = run_backward<C>(fx);
    return matches_finite_difference(fx.stream, r.dstream, [&fx](auto const& p) {
        return objective<C>(fx.w, fx.norm, p, fx.seed);
    });
}

// With zero weights the block is the identity, so the input gradient must equal dout.
// Dropping the residual term leaves dstream at zero and fails here.
template <ModelConfig C> constexpr auto zero_block_passes_dout() -> bool {
    auto fx = make_case<C>();
    fx.w = FeedForward<double, C>{};
    auto const r = run_backward<C>(fx);
    for (std::size_t i = 0; i < C.max_seq * C.dim; ++i) {
        if (!near(r.dstream.storage[i], fx.seed.storage[i], 1e-12)) {
            return false;
        }
    }
    return true;
}

static_assert(down_matches<swiglu_cfg>());
static_assert(up_matches<swiglu_cfg>());
static_assert(gate_matches<swiglu_cfg>());
static_assert(gain_matches<swiglu_cfg>());
static_assert(dstream_matches<swiglu_cfg>());
static_assert(zero_block_passes_dout<swiglu_cfg>());

static_assert(down_matches<gelu_cfg>());
static_assert(up_matches<gelu_cfg>());
static_assert(gain_matches<gelu_cfg>());
static_assert(bias_matches<gelu_cfg>());
static_assert(dstream_matches<gelu_cfg>());
static_assert(zero_block_passes_dout<gelu_cfg>());

} // namespace

auto main() -> int {
    return 0;
}
