#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace poyea::shrimp {

template <typename T, std::size_t... Dims>
    requires(sizeof...(Dims) >= 1) && ((Dims > 0) && ...)
struct Tensor {
    using value_type = T;

    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::size_t size = (Dims * ... * 1u);
    static constexpr std::array<std::size_t, rank> extents{Dims...};

    static constexpr std::array<std::size_t, rank> strides = [] {
        std::array<std::size_t, rank> result{};
        std::size_t acc = 1;
        for (std::size_t i = rank; i-- > 0;) {
            result[i] = acc;
            acc *= extents[i];
        }
        return result;
    }();

    template <std::size_t Axis>
        requires(Axis < rank)
    static constexpr std::size_t extent = extents[Axis];

    std::array<T, size> storage{};

    [[nodiscard]] constexpr auto operator[](decltype(Dims)... indices) noexcept -> T& {
        return storage[offset_of(indices...)];
    }

    [[nodiscard]] constexpr auto operator[](decltype(Dims)... indices) const noexcept -> T const& {
        return storage[offset_of(indices...)];
    }

    [[nodiscard]] constexpr auto flat(std::size_t i) noexcept -> T& { return storage[i]; }

    [[nodiscard]] constexpr auto flat(std::size_t i) const noexcept -> T const& {
        return storage[i];
    }

    [[nodiscard]] constexpr auto begin() noexcept { return storage.begin(); }
    [[nodiscard]] constexpr auto begin() const noexcept { return storage.begin(); }
    [[nodiscard]] constexpr auto end() noexcept { return storage.end(); }
    [[nodiscard]] constexpr auto end() const noexcept { return storage.end(); }

    [[nodiscard]] static constexpr auto filled(T value) noexcept -> Tensor {
        Tensor result{};
        for (std::size_t i = 0; i < size; ++i) {
            result.storage[i] = value;
        }
        return result;
    }

    template <std::size_t... NewDims>
        requires((NewDims * ... * 1u) == size)
    [[nodiscard]] constexpr auto reshape() const noexcept -> Tensor<T, NewDims...> {
        Tensor<T, NewDims...> result{};
        result.storage = storage;
        return result;
    }

  private:
    [[nodiscard]] static constexpr auto
    offset_of(decltype(Dims)... indices) noexcept -> std::size_t {
        std::array<std::size_t, rank> const subscript{indices...};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < rank; ++i) {
            offset += subscript[i] * strides[i];
        }
        return offset;
    }
};

template <typename T, std::size_t N> using Vector = Tensor<T, N>;

template <typename T, std::size_t Rows, std::size_t Cols> using Matrix = Tensor<T, Rows, Cols>;

template <typename T>
concept TensorLike = requires {
    typename T::value_type;
    { T::rank } -> std::convertible_to<std::size_t>;
    { T::size } -> std::convertible_to<std::size_t>;
};

template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr auto row(Matrix<T, Rows, Cols> const& m,
                                 std::size_t index) noexcept -> Vector<T, Cols> {
    Vector<T, Cols> result{};
    for (std::size_t j = 0; j < Cols; ++j) {
        result.storage[j] = m.storage[(index * Cols) + j];
    }
    return result;
}

template <typename T, std::size_t Rows, std::size_t Cols>
constexpr void set_row(Matrix<T, Rows, Cols>& m, std::size_t index,
                       Vector<T, Cols> const& value) noexcept {
    for (std::size_t j = 0; j < Cols; ++j) {
        m.storage[(index * Cols) + j] = value.storage[j];
    }
}

} // namespace poyea::shrimp