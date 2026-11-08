#include "shrimp/core/math.hpp"
#include "shrimp/ops/loss.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Vector;
using poyea::shrimp::ops::cross_entropy;
using poyea::shrimp::ops::cross_entropy_grad;
using poyea::shrimp::ops::log_softmax;

template <typename T>
[[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr Vector<double, 4> logits{{1.0, 2.0, 3.0, 4.0}};

constexpr auto log_softmax_normalizes() -> bool {
    auto const ls = log_softmax(logits);
    double total = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        total += poyea::shrimp::cmath::exp(ls.storage[i]);
    }
    return near(total, 1.0, 1e-14);
}
static_assert(log_softmax_normalizes());

static_assert(near(cross_entropy(logits, 0), 3.4401896985611957, 1e-13));
static_assert(near(cross_entropy(logits, 1), 2.4401896985611957, 1e-13));
static_assert(near(cross_entropy(logits, 2), 1.4401896985611957, 1e-13));
static_assert(near(cross_entropy(logits, 3), 0.4401896985611957, 1e-13));

constexpr Vector<double, 4> shifted{{101.0, 102.0, 103.0, 104.0}};
static_assert(near(cross_entropy(shifted, 3), cross_entropy(logits, 3), 1e-13));

constexpr Vector<double, 4> sharp{{0.0, 0.0, 0.0, 50.0}};
static_assert(near(cross_entropy(sharp, 3), 0.0, 1e-15));
static_assert(cross_entropy(sharp, 0) > 49.0);

constexpr auto grad_sums_to_zero() -> bool {
    auto const g = cross_entropy_grad(logits, 3);
    double total = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        total += g.storage[i];
    }
    return near(total, 0.0, 1e-14);
}
static_assert(grad_sums_to_zero());

static_assert(cross_entropy_grad(logits, 3).storage[3] < 0.0);
static_assert(cross_entropy_grad(logits, 3).storage[0] > 0.0);

// The analytic gradient must match central finite differences of the loss.
constexpr auto grad_matches_finite_difference(std::size_t target) -> bool {
    constexpr double h = 1e-5;
    auto const analytic = cross_entropy_grad(logits, target);

    for (std::size_t i = 0; i < 4; ++i) {
        auto plus = logits;
        auto minus = logits;
        plus.storage[i] += h;
        minus.storage[i] -= h;
        double const fd =
            (cross_entropy(plus, target) - cross_entropy(minus, target)) / (2.0 * h);
        if (!near(analytic.storage[i], fd, 1e-7)) {
            return false;
        }
    }
    return true;
}
static_assert(grad_matches_finite_difference(0));
static_assert(grad_matches_finite_difference(1));
static_assert(grad_matches_finite_difference(2));
static_assert(grad_matches_finite_difference(3));

} // namespace

auto main() -> int {
    return 0;
}
