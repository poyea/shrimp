#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/model/attn_backward.hpp"
#include "shrimp/model/gradients.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::attend;
using poyea::shrimp::attend_backward;
using poyea::shrimp::Attention;
using poyea::shrimp::AttentionGradients;
using poyea::shrimp::Matrix;
using poyea::shrimp::ModelConfig;
using poyea::shrimp::NormGradients;
using poyea::shrimp::NormWeights;
using poyea::shrimp::random_transformer;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr double h = 1e-5;
constexpr double tol = 1e-6;

constexpr ModelConfig cfg{.vocab = 4, .dim = 4, .heads = 1, .layers = 1, .max_seq = 3, .hidden = 4};
constexpr ModelConfig gqa{
    .vocab = 4, .dim = 8, .heads = 4, .kv_heads = 2, .layers = 1, .max_seq = 3, .hidden = 4};

template <ModelConfig C>
constexpr auto patterned(std::size_t salt) -> Matrix<double, C.max_seq, C.dim> {
    Matrix<double, C.max_seq, C.dim> out{};
    for (std::size_t i = 0; i < out.size; ++i) {
        out.storage[i] = (0.25 * static_cast<double>(((i + salt) * 7) % 11)) - 1.1;
    }
    return out;
}

template <ModelConfig C> struct Backward {
    Matrix<double, C.max_seq, C.dim> dstream{};
    AttentionGradients<double, C> dw{};
    NormGradients<double, C.dim, C.norm> dnorm{};
};

template <ModelConfig C>
constexpr auto backward_of(Attention<double, C> const& w,
                           NormWeights<double, C.dim, C.norm> const& norm,
                           Matrix<double, C.max_seq, C.dim> const& stream,
                           Matrix<double, C.max_seq, C.dim> const& seed) -> Backward<C> {
    Backward<C> out{};
    out.dstream = seed;
    attend_backward<double, C>(w, stream, norm, out.dstream, out.dw, out.dnorm, C.max_seq);
    return out;
}

template <ModelConfig C>
constexpr auto forward_loss(Attention<double, C> const& w,
                            NormWeights<double, C.dim, C.norm> const& norm,
                            Matrix<double, C.max_seq, C.dim> const& stream,
                            Matrix<double, C.max_seq, C.dim> const& seed) -> double {
    auto out = stream;
    attend<double, C>(w, out, norm, C.max_seq);
    double total = 0.0;
    for (std::size_t i = 0; i < out.size; ++i) {
        total += out.storage[i] * seed.storage[i];
    }
    return total;
}

template <typename P, typename F>
[[nodiscard]] constexpr auto matches_finite_difference(P const& base, P const& analytic,
                                                       F const& loss) noexcept -> bool {
    for (std::size_t i = 0; i < P::size; ++i) {
        P plus = base;
        P minus = base;
        plus.storage[i] += h;
        minus.storage[i] -= h;
        double const fd = (loss(plus) - loss(minus)) / (2.0 * h);
        if (!near(analytic.storage[i], fd, tol)) {
            return false;
        }
    }
    return true;
}

constexpr auto tiny_model = random_transformer<double, cfg>(7);
constexpr auto tiny_stream = patterned<cfg>(0);
constexpr auto tiny_seed = patterned<cfg>(5);
constexpr auto tiny_back = backward_of<cfg>(tiny_model.blocks[0].attn,
                                            tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);

constexpr auto wq_matches() -> bool {
    return matches_finite_difference(
        tiny_model.blocks[0].attn.wq, tiny_back.dw.wq, [](auto const& p) {
            auto attn = tiny_model.blocks[0].attn;
            attn.wq = p;
            return forward_loss<cfg>(attn, tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);
        });
}
static_assert(wq_matches());

constexpr auto wk_matches() -> bool {
    return matches_finite_difference(
        tiny_model.blocks[0].attn.wk, tiny_back.dw.wk, [](auto const& p) {
            auto attn = tiny_model.blocks[0].attn;
            attn.wk = p;
            return forward_loss<cfg>(attn, tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);
        });
}
static_assert(wk_matches());

constexpr auto wv_matches() -> bool {
    return matches_finite_difference(
        tiny_model.blocks[0].attn.wv, tiny_back.dw.wv, [](auto const& p) {
            auto attn = tiny_model.blocks[0].attn;
            attn.wv = p;
            return forward_loss<cfg>(attn, tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);
        });
}
static_assert(wv_matches());

constexpr auto wo_matches() -> bool {
    return matches_finite_difference(
        tiny_model.blocks[0].attn.wo, tiny_back.dw.wo, [](auto const& p) {
            auto attn = tiny_model.blocks[0].attn;
            attn.wo = p;
            return forward_loss<cfg>(attn, tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);
        });
}
static_assert(wo_matches());

constexpr auto gain_matches() -> bool {
    return matches_finite_difference(
        tiny_model.blocks[0].attn_norm.gain, tiny_back.dnorm.gain, [](auto const& p) {
            auto norm = tiny_model.blocks[0].attn_norm;
            norm.gain = p;
            return forward_loss<cfg>(tiny_model.blocks[0].attn, norm, tiny_stream, tiny_seed);
        });
}
static_assert(gain_matches());

constexpr auto dstream_matches() -> bool {
    return matches_finite_difference(tiny_stream, tiny_back.dstream, [](auto const& p) {
        return forward_loss<cfg>(tiny_model.blocks[0].attn, tiny_model.blocks[0].attn_norm, p,
                                 tiny_seed);
    });
}
static_assert(dstream_matches());

// wo = 0 leaves only the residual, so dstream must equal dout.
constexpr auto residual_retains_dout() -> bool {
    auto attn = tiny_model.blocks[0].attn;
    attn.wo = Matrix<double, cfg.dim, cfg.dim>{};
    auto const back =
        backward_of<cfg>(attn, tiny_model.blocks[0].attn_norm, tiny_stream, tiny_seed);
    for (std::size_t i = 0; i < tiny_seed.size; ++i) {
        if (!near(back.dstream.storage[i], tiny_seed.storage[i], 1e-12)) {
            return false;
        }
    }
    return true;
}
static_assert(residual_retains_dout());

constexpr auto gqa_model = random_transformer<double, gqa>(11);
constexpr auto gqa_stream = patterned<gqa>(0);
constexpr auto gqa_seed = patterned<gqa>(5);
constexpr auto gqa_back =
    backward_of<gqa>(gqa_model.blocks[0].attn, gqa_model.blocks[0].attn_norm, gqa_stream, gqa_seed);

// wk and wv feed every query head of a group, so overwrites fail here.
constexpr auto gqa_wk_matches() -> bool {
    return matches_finite_difference(
        gqa_model.blocks[0].attn.wk, gqa_back.dw.wk, [](auto const& p) {
            auto attn = gqa_model.blocks[0].attn;
            attn.wk = p;
            return forward_loss<gqa>(attn, gqa_model.blocks[0].attn_norm, gqa_stream, gqa_seed);
        });
}
static_assert(gqa_wk_matches());

constexpr auto gqa_wv_matches() -> bool {
    return matches_finite_difference(
        gqa_model.blocks[0].attn.wv, gqa_back.dw.wv, [](auto const& p) {
            auto attn = gqa_model.blocks[0].attn;
            attn.wv = p;
            return forward_loss<gqa>(attn, gqa_model.blocks[0].attn_norm, gqa_stream, gqa_seed);
        });
}
static_assert(gqa_wv_matches());

} // namespace

auto main() -> int {
    return 0;
}
