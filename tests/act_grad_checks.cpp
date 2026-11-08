#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/act_grad.hpp"
#include "shrimp/ops/loss.hpp"
#include "shrimp/ops/nn.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Vector;
using poyea::shrimp::ops::gelu;
using poyea::shrimp::ops::gelu_backward;
using poyea::shrimp::ops::gelu_grad_scalar;
using poyea::shrimp::ops::gelu_scalar;
using poyea::shrimp::ops::log_softmax;
using poyea::shrimp::ops::log_softmax_backward;
using poyea::shrimp::ops::rope;
using poyea::shrimp::ops::rope_backward;
using poyea::shrimp::ops::silu;
using poyea::shrimp::ops::silu_backward;
using poyea::shrimp::ops::silu_grad_scalar;
using poyea::shrimp::ops::silu_scalar;
using poyea::shrimp::ops::softmax;
using poyea::shrimp::ops::softmax_backward;
using poyea::shrimp::ops::softmax_prefix;
using poyea::shrimp::ops::softmax_prefix_backward;

template <typename T>
[[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr double h = 1e-5;
constexpr double tolerance = 1e-6;

template <typename F>
[[nodiscard]] constexpr auto central_difference(F f, double x) noexcept -> double {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

constexpr auto silu_value(double x) noexcept -> double { return silu_scalar(x); }
constexpr auto gelu_value(double x) noexcept -> double { return gelu_scalar(x); }

constexpr auto silu_grad_matches(double x) noexcept -> bool {
    return near(silu_grad_scalar(x), central_difference(silu_value, x), tolerance);
}

constexpr auto gelu_grad_matches(double x) noexcept -> bool {
    return near(gelu_grad_scalar(x), central_difference(gelu_value, x), tolerance);
}

static_assert(silu_grad_matches(-3.0));
static_assert(silu_grad_matches(-0.5));
static_assert(silu_grad_matches(0.0));
static_assert(silu_grad_matches(0.7));
static_assert(silu_grad_matches(4.0));

static_assert(gelu_grad_matches(-3.0));
static_assert(gelu_grad_matches(-0.5));
static_assert(gelu_grad_matches(0.0));
static_assert(gelu_grad_matches(0.7));
static_assert(gelu_grad_matches(4.0));

static_assert(gelu_grad_scalar(0.0) == 0.5);

constexpr Vector<double, 4> sample{{0.7, -2.3, 1.1, 0.05}};
constexpr Vector<double, 4> seed{{0.2, -1.1, 0.6, 1.7}};

template <std::size_t N, typename F>
[[nodiscard]] constexpr auto weighted_sum(F f, Vector<double, N> const& v,
                                          Vector<double, N> const& weights) noexcept -> double {
    auto const y = f(v);
    double total = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        total += y.storage[i] * weights.storage[i];
    }
    return total;
}

template <std::size_t N, typename F>
[[nodiscard]] constexpr auto gradient_matches(F f, Vector<double, N> const& v,
                                              Vector<double, N> const& weights,
                                              Vector<double, N> const& grad) noexcept -> bool {
    for (std::size_t k = 0; k < N; ++k) {
        auto plus = v;
        auto minus = v;
        plus.storage[k] += h;
        minus.storage[k] -= h;
        double const up = weighted_sum(f, plus, weights);
        double const down = weighted_sum(f, minus, weights);
        if (!near(grad.storage[k], (up - down) / (2.0 * h), tolerance)) {
            return false;
        }
    }
    return true;
}

constexpr auto silu_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return silu(v);
}

constexpr auto gelu_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return gelu(v);
}

constexpr auto softmax_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return softmax(v);
}

constexpr auto softmax_prefix_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return softmax_prefix(v, 3);
}

constexpr auto log_softmax_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return log_softmax(v);
}

constexpr auto rope_vec(Vector<double, 4> const& v) noexcept -> Vector<double, 4> {
    return rope(v, 5, 10000.0);
}

static_assert(gradient_matches(silu_vec, sample, seed, silu_backward(sample, seed)));
static_assert(gradient_matches(gelu_vec, sample, seed, gelu_backward(sample, seed)));
static_assert(
    gradient_matches(softmax_vec, sample, seed, softmax_backward(softmax(sample), seed)));
static_assert(gradient_matches(softmax_prefix_vec, sample, seed,
                               softmax_prefix_backward(softmax_prefix(sample, 3), seed, 3)));
static_assert(gradient_matches(log_softmax_vec, sample, seed,
                               log_softmax_backward(log_softmax(sample), seed)));
static_assert(gradient_matches(rope_vec, sample, seed, rope_backward(seed, 5, 10000.0)));

constexpr auto constant_seed_gives_zero_softmax_grad() noexcept -> bool {
    Vector<double, 4> const constant{{0.9, 0.9, 0.9, 0.9}};
    auto const dx = softmax_backward(softmax(sample), constant);
    for (std::size_t i = 0; i < 4; ++i) {
        if (!near(dx.storage[i], 0.0, 1e-12)) {
            return false;
        }
    }
    return true;
}
static_assert(constant_seed_gives_zero_softmax_grad());

constexpr auto rope_round_trip_is_identity() noexcept -> bool {
    auto const rotated = rope(sample, 5, 10000.0);
    auto const restored = rope_backward(rotated, 5, 10000.0);
    for (std::size_t i = 0; i < 4; ++i) {
        if (!near(restored.storage[i], sample.storage[i], 1e-12)) {
            return false;
        }
    }
    return true;
}
static_assert(rope_round_trip_is_identity());

} // namespace

auto main() -> int {
    return 0;
}
