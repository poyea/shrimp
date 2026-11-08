#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/nn.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T> [[nodiscard]] constexpr auto silu_grad_scalar(T x) noexcept -> T {
    T const s = cmath::sigmoid(x);
    return s * (T{1} + (x * (T{1} - s)));
}

template <typename T> [[nodiscard]] constexpr auto gelu_grad_scalar(T x) noexcept -> T {
    constexpr T coeff = T{0.7978845608028654};
    constexpr T cubic = T{0.044715};
    T const t = cmath::tanh(coeff * (x + (cubic * x * x * x)));
    T const du = coeff * (T{1} + (T{3} * cubic * x * x));
    return T{0.5} * (T{1} + t) + (T{0.5} * x * (T{1} - (t * t)) * du);
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto
silu_backward(Tensor<T, Dims...> const& x,
              Tensor<T, Dims...> const& dy) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> dx{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        dx.storage[i] = dy.storage[i] * silu_grad_scalar(x.storage[i]);
    }
    return dx;
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto
gelu_backward(Tensor<T, Dims...> const& x,
              Tensor<T, Dims...> const& dy) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> dx{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        dx.storage[i] = dy.storage[i] * gelu_grad_scalar(x.storage[i]);
    }
    return dx;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto softmax_backward(Vector<T, N> const& out,
                                              Vector<T, N> const& dy) noexcept -> Vector<T, N> {
    T dot{};
    for (std::size_t i = 0; i < N; ++i) {
        dot += out.storage[i] * dy.storage[i];
    }
    Vector<T, N> dx{};
    for (std::size_t i = 0; i < N; ++i) {
        dx.storage[i] = out.storage[i] * (dy.storage[i] - dot);
    }
    return dx;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto softmax_prefix_backward(Vector<T, N> const& out,
                                                     Vector<T, N> const& dy,
                                                     std::size_t count) noexcept -> Vector<T, N> {
    T dot{};
    for (std::size_t i = 0; i < count; ++i) {
        dot += out.storage[i] * dy.storage[i];
    }
    Vector<T, N> dx{};
    for (std::size_t i = 0; i < count; ++i) {
        dx.storage[i] = out.storage[i] * (dy.storage[i] - dot);
    }
    return dx;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto log_softmax_backward(Vector<T, N> const& out,
                                                  Vector<T, N> const& dy) noexcept -> Vector<T, N> {
    T total{};
    for (std::size_t i = 0; i < N; ++i) {
        total += dy.storage[i];
    }
    Vector<T, N> dx{};
    for (std::size_t i = 0; i < N; ++i) {
        dx.storage[i] = dy.storage[i] - (cmath::exp(out.storage[i]) * total);
    }
    return dx;
}

// The forward is a rotation R; its transpose R^T is rotation by the negative angle.
template <typename T, std::size_t HeadDim>
    requires(HeadDim % 2 == 0)
[[nodiscard]] constexpr auto rope_backward(Vector<T, HeadDim> const& dy, std::size_t position,
                                           T theta_base = T{10000}) noexcept -> Vector<T, HeadDim> {
    constexpr std::size_t half = HeadDim / 2;
    Vector<T, HeadDim> dx{};
    for (std::size_t i = 0; i < half; ++i) {
        T const exponent = static_cast<T>(2 * i) / static_cast<T>(HeadDim);
        T const inv_freq = T{1} / cmath::pow(theta_base, exponent);
        T const angle = static_cast<T>(position) * inv_freq;
        T const cos_a = cmath::cos(angle);
        T const sin_a = cmath::sin(angle);

        T const lo = dy.storage[i];
        T const hi = dy.storage[i + half];
        dx.storage[i] = (lo * cos_a) + (hi * sin_a);
        dx.storage[i + half] = (hi * cos_a) - (lo * sin_a);
    }
    return dx;
}

template <typename T, std::size_t HeadDim, std::size_t MaxSeq>
    requires(HeadDim % 2 == 0)
[[nodiscard]] constexpr auto rope_backward(Vector<T, HeadDim> const& dy,
                                           RopeTable<T, HeadDim, MaxSeq> const& table,
                                           std::size_t position) noexcept -> Vector<T, HeadDim> {
    constexpr std::size_t half = HeadDim / 2;
    Vector<T, HeadDim> dx{};
    std::size_t const base = position * half;
    for (std::size_t i = 0; i < half; ++i) {
        T const cos_a = table.cosines.storage[base + i];
        T const sin_a = table.sines.storage[base + i];
        T const lo = dy.storage[i];
        T const hi = dy.storage[i + half];
        dx.storage[i] = (lo * cos_a) + (hi * sin_a);
        dx.storage[i + half] = (hi * cos_a) - (lo * sin_a);
    }
    return dx;
}

} // namespace poyea::shrimp::ops
