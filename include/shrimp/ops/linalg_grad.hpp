#pragma once

#include "shrimp/core/tensor.hpp"

#include <cstddef>

namespace poyea::shrimp::ops {

template <typename T, std::size_t N>
struct DotGrad {
    Vector<T, N> da{};
    Vector<T, N> db{};
};

template <typename T, std::size_t N>
[[nodiscard]] constexpr auto dot_backward(Vector<T, N> const& a, Vector<T, N> const& b,
                                          T dy) noexcept -> DotGrad<T, N> {
    DotGrad<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out.da.storage[i] = dy * b.storage[i];
        out.db.storage[i] = dy * a.storage[i];
    }
    return out;
}

template <typename T, std::size_t Out, std::size_t In>
struct LinearGrad {
    Vector<T, In> dx{};
    Matrix<T, Out, In> dw{};
};

template <typename T, std::size_t Out, std::size_t In>
[[nodiscard]] constexpr auto linear_backward(Matrix<T, Out, In> const& weight,
                                             Vector<T, In> const& x,
                                             Vector<T, Out> const& dy) noexcept
    -> LinearGrad<T, Out, In> {
    LinearGrad<T, Out, In> out{};
    for (std::size_t o = 0; o < Out; ++o) {
        for (std::size_t i = 0; i < In; ++i) {
            out.dx.storage[i] += weight.storage[(o * In) + i] * dy.storage[o];
            out.dw.storage[(o * In) + i] = dy.storage[o] * x.storage[i];
        }
    }
    return out;
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct MatmulGrad {
    Matrix<T, M, K> da{};
    Matrix<T, K, N> db{};
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
[[nodiscard]] constexpr auto matmul_backward(Matrix<T, M, K> const& a, Matrix<T, K, N> const& b,
                                             Matrix<T, M, N> const& dc) noexcept
    -> MatmulGrad<T, M, K, N> {
    MatmulGrad<T, M, K, N> out{};
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t j = 0; j < N; ++j) {
                out.da.storage[(i * K) + k] += dc.storage[(i * N) + j] * b.storage[(k * N) + j];
                out.db.storage[(k * N) + j] += a.storage[(i * K) + k] * dc.storage[(i * N) + j];
            }
        }
    }
    return out;
}

template <typename T, std::size_t... Dims>
struct BinaryGrad {
    Tensor<T, Dims...> da{};
    Tensor<T, Dims...> db{};
};

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto add_backward(Tensor<T, Dims...> const& dy) noexcept
    -> BinaryGrad<T, Dims...> {
    return BinaryGrad<T, Dims...>{dy, dy};
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto hadamard_backward(Tensor<T, Dims...> const& a,
                                               Tensor<T, Dims...> const& b,
                                               Tensor<T, Dims...> const& dy) noexcept
    -> BinaryGrad<T, Dims...> {
    BinaryGrad<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.da.storage[i] = dy.storage[i] * b.storage[i];
        out.db.storage[i] = dy.storage[i] * a.storage[i];
    }
    return out;
}

template <typename T, std::size_t... Dims>
struct ScaleGrad {
    Tensor<T, Dims...> da{};
    T dfactor{};
};

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto scale_backward(Tensor<T, Dims...> const& a, T factor,
                                            Tensor<T, Dims...> const& dy) noexcept
    -> ScaleGrad<T, Dims...> {
    ScaleGrad<T, Dims...> out{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        out.da.storage[i] = dy.storage[i] * factor;
        out.dfactor += dy.storage[i] * a.storage[i];
    }
    return out;
}

} // namespace poyea::shrimp::ops
