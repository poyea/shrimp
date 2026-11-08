#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T, std::size_t N> struct RmsNormGrad {
    Vector<T, N> dx{};
    Vector<T, N> dgain{};
};

template <typename T, std::size_t N> struct LayerNormGrad {
    Vector<T, N> dx{};
    Vector<T, N> dgain{};
    Vector<T, N> dbias{};
};

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto rmsnorm_backward(Vector<T, N> const& x, Vector<T, N> const& gain,
                                              T epsilon,
                                              Vector<T, N> const& dy) noexcept
    -> RmsNormGrad<T, N> {
    T sum_squares{};
    for (std::size_t i = 0; i < N; ++i) {
        sum_squares += x.storage[i] * x.storage[i];
    }
    T const inv_rms = cmath::rsqrt((sum_squares / static_cast<T>(N)) + epsilon);

    T projection{};
    for (std::size_t i = 0; i < N; ++i) {
        projection += dy.storage[i] * gain.storage[i] * x.storage[i];
    }
    T const coeff = inv_rms * inv_rms * inv_rms / static_cast<T>(N);

    RmsNormGrad<T, N> grad{};
    for (std::size_t i = 0; i < N; ++i) {
        grad.dx.storage[i] = (inv_rms * gain.storage[i] * dy.storage[i]) -
                             (coeff * x.storage[i] * projection);
        grad.dgain.storage[i] = dy.storage[i] * x.storage[i] * inv_rms;
    }
    return grad;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto layernorm_backward(Vector<T, N> const& x, Vector<T, N> const& gain,
                                                T epsilon, Vector<T, N> const& dy) noexcept
    -> LayerNormGrad<T, N> {
    T mean{};
    for (std::size_t i = 0; i < N; ++i) {
        mean += x.storage[i];
    }
    mean /= static_cast<T>(N);

    T variance{};
    for (std::size_t i = 0; i < N; ++i) {
        T const d = x.storage[i] - mean;
        variance += d * d;
    }
    T const inv_std = cmath::rsqrt((variance / static_cast<T>(N)) + epsilon);

    Vector<T, N> xhat{};
    T mean_g{};
    T mean_g_xhat{};
    for (std::size_t i = 0; i < N; ++i) {
        xhat.storage[i] = (x.storage[i] - mean) * inv_std;
        T const g = dy.storage[i] * gain.storage[i];
        mean_g += g;
        mean_g_xhat += g * xhat.storage[i];
    }
    mean_g /= static_cast<T>(N);
    mean_g_xhat /= static_cast<T>(N);

    LayerNormGrad<T, N> grad{};
    for (std::size_t i = 0; i < N; ++i) {
        T const g = dy.storage[i] * gain.storage[i];
        grad.dx.storage[i] = inv_std * (g - mean_g - (xhat.storage[i] * mean_g_xhat));
        grad.dgain.storage[i] = dy.storage[i] * xhat.storage[i];
        grad.dbias.storage[i] = dy.storage[i];
    }
    return grad;
}

} // namespace poyea::shrimp::ops
