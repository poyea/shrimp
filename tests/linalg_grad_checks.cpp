#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/linalg.hpp"
#include "shrimp/ops/linalg_grad.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Matrix;
using poyea::shrimp::Tensor;
using poyea::shrimp::Vector;
using poyea::shrimp::ops::add;
using poyea::shrimp::ops::add_backward;
using poyea::shrimp::ops::dot;
using poyea::shrimp::ops::dot_backward;
using poyea::shrimp::ops::hadamard;
using poyea::shrimp::ops::hadamard_backward;
using poyea::shrimp::ops::linear;
using poyea::shrimp::ops::linear_backward;
using poyea::shrimp::ops::matmul;
using poyea::shrimp::ops::matmul_backward;
using poyea::shrimp::ops::scale;
using poyea::shrimp::ops::scale_backward;

constexpr double h = 1e-5;

template <typename T>
[[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

template <typename TensorT>
[[nodiscard]] constexpr auto inner(TensorT const& a, TensorT const& b) noexcept -> double {
    double acc = 0.0;
    for (std::size_t i = 0; i < TensorT::size; ++i) {
        acc += a.storage[i] * b.storage[i];
    }
    return acc;
}

template <typename TensorT, typename F>
[[nodiscard]] constexpr auto fd_matches(TensorT const& x, TensorT const& grad,
                                        F const& f) noexcept -> bool {
    for (std::size_t i = 0; i < TensorT::size; ++i) {
        TensorT plus = x;
        TensorT minus = x;
        plus.storage[i] += h;
        minus.storage[i] -= h;
        double const fd = (f(plus) - f(minus)) / (2.0 * h);
        if (!near(grad.storage[i], fd, 1e-7)) {
            return false;
        }
    }
    return true;
}

constexpr double seed = 0.7;
constexpr double sf = -0.65;

constexpr Vector<double, 4> va{{0.3, -1.7, 2.1, 0.45}};
constexpr Vector<double, 4> vb{{-0.8, 1.3, 0.6, -2.2}};

constexpr Matrix<double, 2, 3> lw{{0.3, -1.7, 2.1, 0.45, -0.9, 1.25}};
constexpr Vector<double, 3> lx{{-0.6, 0.8, 1.9}};
constexpr Vector<double, 2> ldy{{0.7, -1.3}};

constexpr Matrix<double, 2, 3> ma{{0.6, -0.4, 1.3, -2.05, 0.25, 0.9}};
constexpr Matrix<double, 3, 4> mb{{0.3, -1.7, 2.1, 0.45, -0.9, 1.25, -0.15, 0.8, 1.1, -0.6, 0.35,
                                   -1.45}};
constexpr Matrix<double, 2, 4> mc{{0.7, -1.3, 0.2, 1.6, -0.5, 0.95, -2.4, 0.15}};

constexpr Tensor<double, 2, 2> ta{{0.3, -1.7, 2.1, 0.45}};
constexpr Tensor<double, 2, 2> tb{{-0.8, 1.3, 0.6, -2.2}};
constexpr Tensor<double, 2, 2> ts{{0.7, -1.3, 0.2, 1.6}};

constexpr auto dot_grads_match_fd() -> bool {
    auto const g = dot_backward(va, vb, seed);
    return fd_matches(va, g.da, [](auto const& a) { return seed * dot(a, vb); }) &&
           fd_matches(vb, g.db, [](auto const& b) { return seed * dot(va, b); });
}
static_assert(dot_grads_match_fd());

constexpr auto linear_grads_match_fd() -> bool {
    auto const g = linear_backward(lw, lx, ldy);
    return fd_matches(lx, g.dx, [](auto const& x) { return inner(ldy, linear(lw, x)); }) &&
           fd_matches(lw, g.dw, [](auto const& w) { return inner(ldy, linear(w, lx)); });
}
static_assert(linear_grads_match_fd());

constexpr auto matmul_grads_match_fd() -> bool {
    auto const g = matmul_backward(ma, mb, mc);
    return fd_matches(ma, g.da, [](auto const& a) { return inner(mc, matmul(a, mb)); }) &&
           fd_matches(mb, g.db, [](auto const& b) { return inner(mc, matmul(ma, b)); });
}
static_assert(matmul_grads_match_fd());

constexpr auto add_grads_match_fd() -> bool {
    auto const g = add_backward(ts);
    return fd_matches(ta, g.da, [](auto const& a) { return inner(ts, add(a, tb)); }) &&
           fd_matches(tb, g.db, [](auto const& b) { return inner(ts, add(ta, b)); });
}
static_assert(add_grads_match_fd());

constexpr auto hadamard_grads_match_fd() -> bool {
    auto const g = hadamard_backward(ta, tb, ts);
    return fd_matches(ta, g.da, [](auto const& a) { return inner(ts, hadamard(a, tb)); }) &&
           fd_matches(tb, g.db, [](auto const& b) { return inner(ts, hadamard(ta, b)); });
}
static_assert(hadamard_grads_match_fd());

constexpr auto scale_grads_match_fd() -> bool {
    auto const g = scale_backward(ta, sf, ts);
    double const fd_factor =
        (inner(ts, scale(ta, sf + h)) - inner(ts, scale(ta, sf - h))) / (2.0 * h);
    return fd_matches(ta, g.da, [](auto const& a) { return inner(ts, scale(a, sf)); }) &&
           near(g.dfactor, fd_factor, 1e-7);
}
static_assert(scale_grads_match_fd());

static_assert(decltype(linear_backward(lw, lx, ldy).dw)::extent<0> == 2);
static_assert(decltype(linear_backward(lw, lx, ldy).dw)::extent<1> == 3);
static_assert(decltype(linear_backward(lw, lx, ldy).dx)::size == 3);
static_assert(decltype(matmul_backward(ma, mb, mc).da)::extent<0> == 2);
static_assert(decltype(matmul_backward(ma, mb, mc).da)::extent<1> == 3);
static_assert(decltype(matmul_backward(ma, mb, mc).db)::extent<0> == 3);
static_assert(decltype(matmul_backward(ma, mb, mc).db)::extent<1> == 4);

} // namespace

auto main() -> int {
    return 0;
}
