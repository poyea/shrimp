#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/nn.hpp"
#include "shrimp/ops/norm_grad.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Vector;
using poyea::shrimp::cmath::rsqrt;
using poyea::shrimp::ops::layernorm;
using poyea::shrimp::ops::layernorm_backward;
using poyea::shrimp::ops::rmsnorm;
using poyea::shrimp::ops::rmsnorm_backward;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr double h = 1e-5;
constexpr double tol = 1e-6;

constexpr Vector<double, 4> x{{0.7, -2.3, 1.1, 0.05}};
constexpr Vector<double, 4> gain{{1.3, 0.4, -0.9, 2.0}};
constexpr Vector<double, 4> bias{{0.25, -0.5, 0.75, -1.0}};
constexpr Vector<double, 4> seed{{0.2, -1.1, 0.6, 1.7}};
constexpr Vector<double, 4> ones{{1.0, 1.0, 1.0, 1.0}};

template <std::size_t N>
[[nodiscard]] constexpr auto dot(Vector<double, N> const& a,
                                 Vector<double, N> const& b) noexcept -> double {
    double total = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        total += a.storage[i] * b.storage[i];
    }
    return total;
}

template <std::size_t N, typename F>
[[nodiscard]] constexpr auto matches_finite_difference(Vector<double, N> const& base,
                                                       Vector<double, N> const& analytic,
                                                       F const& objective) noexcept -> bool {
    for (std::size_t i = 0; i < N; ++i) {
        auto plus = base;
        auto minus = base;
        plus.storage[i] += h;
        minus.storage[i] -= h;
        double const fd = (objective(plus) - objective(minus)) / (2.0 * h);
        if (!near(analytic.storage[i], fd, tol)) {
            return false;
        }
    }
    return true;
}

constexpr auto rms_dx_matches() -> bool {
    auto const grad = rmsnorm_backward(x, gain, 0.0, seed);
    return matches_finite_difference(
        x, grad.dx, [](auto const& p) { return dot(rmsnorm(p, gain, 0.0), seed); });
}
static_assert(rms_dx_matches());

constexpr auto rms_dgain_matches() -> bool {
    auto const grad = rmsnorm_backward(x, gain, 0.0, seed);
    return matches_finite_difference(gain, grad.dgain,
                                     [](auto const& p) { return dot(rmsnorm(x, p, 0.0), seed); });
}
static_assert(rms_dgain_matches());

constexpr auto ln_dx_matches() -> bool {
    auto const grad = layernorm_backward(x, gain, 0.0, seed);
    return matches_finite_difference(
        x, grad.dx, [](auto const& p) { return dot(layernorm(p, gain, bias, 0.0), seed); });
}
static_assert(ln_dx_matches());

constexpr auto ln_dgain_matches() -> bool {
    auto const grad = layernorm_backward(x, gain, 0.0, seed);
    return matches_finite_difference(
        gain, grad.dgain, [](auto const& p) { return dot(layernorm(x, p, bias, 0.0), seed); });
}
static_assert(ln_dgain_matches());

constexpr auto ln_dbias_matches() -> bool {
    auto const grad = layernorm_backward(x, gain, 0.0, seed);
    return matches_finite_difference(
        bias, grad.dbias, [](auto const& p) { return dot(layernorm(x, gain, p, 0.0), seed); });
}
static_assert(ln_dbias_matches());

constexpr auto radial_sum(Vector<double, 4> const& dx) noexcept -> double {
    return dot(dx, x);
}

constexpr auto rms_radially_orthogonal_for_seed() -> bool {
    auto const grad = rmsnorm_backward(x, ones, 0.0, seed);
    return near(radial_sum(grad.dx), 0.0, 1e-9);
}
static_assert(rms_radially_orthogonal_for_seed());

constexpr auto naive_rms_dx_is_rejected() -> bool {
    auto const grad = rmsnorm_backward(x, gain, 0.0, seed);
    double const inv_rms = rsqrt(dot(x, x) / 4.0);
    for (std::size_t i = 0; i < 4; ++i) {
        double const naive = inv_rms * gain.storage[i] * seed.storage[i];
        if (!near(naive, grad.dx.storage[i], 1e-3)) {
            return true;
        }
    }
    return false;
}
static_assert(naive_rms_dx_is_rejected());

constexpr auto naive_ln_dx_is_rejected() -> bool {
    auto const grad = layernorm_backward(x, gain, 0.0, seed);
    double mean = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        mean += x.storage[i] / 4.0;
    }
    double variance = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        variance += (x.storage[i] - mean) * (x.storage[i] - mean) / 4.0;
    }
    double const inv_std = rsqrt(variance);
    for (std::size_t i = 0; i < 4; ++i) {
        double const naive = inv_std * gain.storage[i] * seed.storage[i];
        if (!near(naive, grad.dx.storage[i], 1e-3)) {
            return true;
        }
    }
    return false;
}
static_assert(naive_ln_dx_is_rejected());

} // namespace

auto main() -> int {
    return 0;
}
