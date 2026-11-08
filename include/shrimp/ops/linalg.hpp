#pragma once

#include "shrimp/core/tensor.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto dot(Vector<T, N> const& a, Vector<T, N> const& b) noexcept -> T {
    T acc{};
    for (std::size_t i = 0; i < N; ++i) {
        acc += a.storage[i] * b.storage[i];
    }
    return acc;
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
[[nodiscard]] constexpr auto matmul(Matrix<T, M, K> const& a,
                                    Matrix<T, K, N> const& b) noexcept -> Matrix<T, M, N> {
    Matrix<T, M, N> out{};
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            T const a_ik = a.storage[(i * K) + k];
            if (a_ik == T{}) { // skips a row of multiplies against the step budget
                continue;
            }
            for (std::size_t j = 0; j < N; ++j) {
                out.storage[(i * N) + j] += a_ik * b.storage[(k * N) + j];
            }
        }
    }
    return out;
}

template <typename T, std::size_t Out, std::size_t In>
[[nodiscard]] constexpr auto linear(Matrix<T, Out, In> const& weight,
                                    Vector<T, In> const& x) noexcept -> Vector<T, Out> {
    Vector<T, Out> out{};
    for (std::size_t o = 0; o < Out; ++o) {
        T acc{};
        std::size_t const base = o * In;
        for (std::size_t i = 0; i < In; ++i) {
            acc += weight.storage[base + i] * x.storage[i];
        }
        out.storage[o] = acc;
    }
    return out;
}

template <typename T, std::size_t Out, std::size_t In>
[[nodiscard]] constexpr auto linear(Matrix<T, Out, In> const& weight, Vector<T, Out> const& bias,
                                    Vector<T, In> const& x) noexcept -> Vector<T, Out> {
    Vector<T, Out> out = linear(weight, x);
    for (std::size_t o = 0; o < Out; ++o) {
        out.storage[o] += bias.storage[o];
    }
    return out;
}

template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr auto
transpose(Matrix<T, Rows, Cols> const& m) noexcept -> Matrix<T, Cols, Rows> {
    Matrix<T, Cols, Rows> out{};
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            out.storage[(j * Rows) + i] = m.storage[(i * Cols) + j];
        }
    }
    return out;
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto add(Tensor<T, Dims...> const& a,
                                 Tensor<T, Dims...> const& b) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.storage[i] = a.storage[i] + b.storage[i];
    }
    return out;
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto hadamard(Tensor<T, Dims...> const& a,
                                      Tensor<T, Dims...> const& b) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.storage[i] = a.storage[i] * b.storage[i];
    }
    return out;
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto scale(Tensor<T, Dims...> const& a,
                                   T factor) noexcept -> Tensor<T, Dims...> {
    Tensor<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.storage[i] = a.storage[i] * factor;
    }
    return out;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto argmax(Vector<T, N> const& v) noexcept -> std::size_t {
    std::size_t best = 0;
    for (std::size_t i = 1; i < N; ++i) {
        if (v.storage[i] > v.storage[best]) {
            best = i;
        }
    }
    return best;
}

} // namespace poyea::shrimp::ops