#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace poyea::shrimp {

struct SplitMix64 {
    std::uint64_t state = 0x853c49e6748fea9bULL;

    constexpr auto next() noexcept -> std::uint64_t {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

template <std::floating_point T>
[[nodiscard]] constexpr auto uniform(SplitMix64& rng) noexcept -> T {
    constexpr auto denominator = static_cast<T>(std::uint64_t{1} << 53);
    return static_cast<T>(rng.next() >> 11) / denominator;
}

template <std::floating_point T>
[[nodiscard]] constexpr auto uniform(SplitMix64& rng, T low, T high) noexcept -> T {
    return low + (uniform<T>(rng) * (high - low));
}

template <std::floating_point T>
[[nodiscard]] constexpr auto normal(SplitMix64& rng) noexcept -> T {
    T u1 = uniform<T>(rng);
    while (u1 <= T{0}) { // log(0) is undefined; redraw
        u1 = uniform<T>(rng);
    }
    T const u2 = uniform<T>(rng);
    return cmath::sqrt(T{-2} * cmath::log(u1)) *
           cmath::cos(T{2} * std::numbers::pi_v<T> * u2);
}

template <typename T, std::size_t... Dims>
constexpr void fill_normal(Tensor<T, Dims...>& t, SplitMix64& rng, T stddev) noexcept {
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        t.storage[i] = normal<T>(rng) * stddev;
    }
}

template <typename T, std::size_t... Dims>
constexpr void fill_uniform(Tensor<T, Dims...>& t, SplitMix64& rng, T low,
                            T high) noexcept {
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        t.storage[i] = uniform<T>(rng, low, high);
    }
}

} // namespace poyea::shrimp