#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto softmax(Vector<T, N> const& x) noexcept -> Vector<T, N> {
    T peak = x.storage[0];
    for (std::size_t i = 1; i < N; ++i) {
        peak = cmath::max(peak, x.storage[i]);
    }

    Vector<T, N> out{};
    T total{};
    for (std::size_t i = 0; i < N; ++i) {
        out.storage[i] = cmath::exp(x.storage[i] - peak);
        total += out.storage[i];
    }
    T const inv = T{1} / total;
    for (std::size_t i = 0; i < N; ++i) {
        out.storage[i] *= inv;
    }
    return out;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto softmax_prefix(Vector<T, N> const& x,
                                            std::size_t count) noexcept -> Vector<T, N> {
    Vector<T, N> out{};
    if (count == 0) {
        return out;
    }

    T peak = x.storage[0];
    for (std::size_t i = 1; i < count; ++i) {
        peak = cmath::max(peak, x.storage[i]);
    }

    T total{};
    for (std::size_t i = 0; i < count; ++i) {
        out.storage[i] = cmath::exp(x.storage[i] - peak);
        total += out.storage[i];
    }
    T const inv = T{1} / total;
    for (std::size_t i = 0; i < count; ++i) {
        out.storage[i] *= inv;
    }
    return out;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto rmsnorm(Vector<T, N> const& x, Vector<T, N> const& gain,
                                     T epsilon = T{1e-5}) noexcept -> Vector<T, N> {
    T sum_squares{};
    for (std::size_t i = 0; i < N; ++i) {
        sum_squares += x.storage[i] * x.storage[i];
    }
    T const inv_rms = cmath::rsqrt((sum_squares / static_cast<T>(N)) + epsilon);

    Vector<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out.storage[i] = x.storage[i] * inv_rms * gain.storage[i];
    }
    return out;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto layernorm(Vector<T, N> const& x, Vector<T, N> const& gain,
                                       Vector<T, N> const& bias,
                                       T epsilon = T{1e-5}) noexcept -> Vector<T, N> {
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

    Vector<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out.storage[i] = ((x.storage[i] - mean) * inv_std * gain.storage[i]) + bias.storage[i];
    }
    return out;
}

template <typename T> [[nodiscard]] constexpr auto gelu_scalar(T x) noexcept -> T {
    constexpr T coeff = T{0.7978845608028654}; // sqrt(2/pi)
    constexpr T cubic = T{0.044715};
    return T{0.5} * x * (T{1} + cmath::tanh(coeff * (x + (cubic * x * x * x))));
}

template <typename T> [[nodiscard]] constexpr auto silu_scalar(T x) noexcept -> T {
    return x * cmath::sigmoid(x);
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto gelu(Tensor<T, Dims...> const& x) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.storage[i] = gelu_scalar(x.storage[i]);
    }
    return out;
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto silu(Tensor<T, Dims...> const& x) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.storage[i] = silu_scalar(x.storage[i]);
    }
    return out;
}

// Split-half: i pairs with i + HeadDim/2, as in GPT-NeoX and HF Llama.
// RoFormer pairs adjacent elements; the two are not interchangeable.
template <typename T, std::size_t HeadDim>
    requires(HeadDim % 2 == 0)
[[nodiscard]] constexpr auto rope(Vector<T, HeadDim> const& x, std::size_t position,
                                  T theta_base = T{10000}) noexcept -> Vector<T, HeadDim> {
    constexpr std::size_t half = HeadDim / 2;
    Vector<T, HeadDim> out{};
    for (std::size_t i = 0; i < half; ++i) {
        T const exponent = static_cast<T>(2 * i) / static_cast<T>(HeadDim);
        T const inv_freq = T{1} / cmath::pow(theta_base, exponent);
        T const angle = static_cast<T>(position) * inv_freq;
        T const cos_a = cmath::cos(angle);
        T const sin_a = cmath::sin(angle);

        T const lo = x.storage[i];
        T const hi = x.storage[i + half];
        out.storage[i] = (lo * cos_a) - (hi * sin_a);
        out.storage[i + half] = (lo * sin_a) + (hi * cos_a);
    }
    return out;
}

template <typename T, std::size_t HeadDim, std::size_t MaxSeq>
    requires(HeadDim % 2 == 0)
struct RopeTable {
    Matrix<T, MaxSeq, HeadDim / 2> cosines{};
    Matrix<T, MaxSeq, HeadDim / 2> sines{};
};

template <typename T, std::size_t HeadDim, std::size_t MaxSeq>
    requires(HeadDim % 2 == 0)
[[nodiscard]] constexpr auto
make_rope_table(T theta_base) noexcept -> RopeTable<T, HeadDim, MaxSeq> {
    constexpr std::size_t half = HeadDim / 2;
    RopeTable<T, HeadDim, MaxSeq> table{};
    for (std::size_t i = 0; i < half; ++i) {
        T const exponent = static_cast<T>(2 * i) / static_cast<T>(HeadDim);
        T const inv_freq = T{1} / cmath::pow(theta_base, exponent);
        for (std::size_t p = 0; p < MaxSeq; ++p) {
            T const angle = static_cast<T>(p) * inv_freq;
            table.cosines.storage[(p * half) + i] = cmath::cos(angle);
            table.sines.storage[(p * half) + i] = cmath::sin(angle);
        }
    }
    return table;
}

// One object per instantiation, so the transcendentals leave the hot loop.
template <typename T, std::size_t HeadDim, std::size_t MaxSeq, T Theta>
    requires(HeadDim % 2 == 0)
inline constexpr RopeTable<T, HeadDim, MaxSeq> rope_table =
    make_rope_table<T, HeadDim, MaxSeq>(Theta);

template <typename T, std::size_t HeadDim, std::size_t MaxSeq>
    requires(HeadDim % 2 == 0)
[[nodiscard]] constexpr auto rope(Vector<T, HeadDim> const& x,
                                  RopeTable<T, HeadDim, MaxSeq> const& table,
                                  std::size_t position) noexcept -> Vector<T, HeadDim> {
    constexpr std::size_t half = HeadDim / 2;
    Vector<T, HeadDim> out{};
    std::size_t const base = position * half;
    for (std::size_t i = 0; i < half; ++i) {
        T const cos_a = table.cosines.storage[base + i];
        T const sin_a = table.sines.storage[base + i];
        T const lo = x.storage[i];
        T const hi = x.storage[i + half];
        out.storage[i] = (lo * cos_a) - (hi * sin_a);
        out.storage[i + half] = (lo * sin_a) + (hi * cos_a);
    }
    return out;
}

} // namespace poyea::shrimp::ops