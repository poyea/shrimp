#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/nn.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto log_softmax(Vector<T, N> const& x) noexcept -> Vector<T, N> {
    T peak = x.storage[0];
    for (std::size_t i = 1; i < N; ++i) {
        peak = cmath::max(peak, x.storage[i]);
    }

    T total{};
    for (std::size_t i = 0; i < N; ++i) {
        total += cmath::exp(x.storage[i] - peak);
    }
    T const log_total = cmath::log(total);

    Vector<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out.storage[i] = x.storage[i] - peak - log_total;
    }
    return out;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto cross_entropy(Vector<T, N> const& logits,
                                           std::size_t target) noexcept -> T {
    return -log_softmax(logits).storage[target];
}

// d/dlogits of cross_entropy: softmax minus the one-hot target.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto cross_entropy_grad(Vector<T, N> const& logits,
                                                std::size_t target) noexcept
    -> Vector<T, N> {
    Vector<T, N> grad = softmax(logits);
    grad.storage[target] -= T{1};
    return grad;
}

} // namespace poyea::shrimp::ops
