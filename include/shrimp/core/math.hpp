#pragma once

#include <concepts>
#include <limits>
#include <numbers>

namespace poyea::shrimp::cmath {

template <std::floating_point T> [[nodiscard]] constexpr auto abs(T x) noexcept -> T {
    return x < T{0} ? -x : x;
}

template <std::floating_point T> [[nodiscard]] constexpr auto is_nan(T x) noexcept -> bool {
    return !(x == x);
}

template <std::floating_point T> [[nodiscard]] constexpr auto is_inf(T x) noexcept -> bool {
    return x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity();
}

template <std::floating_point T> [[nodiscard]] constexpr auto min(T a, T b) noexcept -> T {
    return b < a ? b : a;
}

template <std::floating_point T> [[nodiscard]] constexpr auto max(T a, T b) noexcept -> T {
    return a < b ? b : a;
}

template <std::floating_point T> [[nodiscard]] constexpr auto ldexp(T x, int k) noexcept -> T {
    T base = k >= 0 ? T{2} : T{0.5};
    int n = k >= 0 ? k : -k;
    T result = x;
    while (n > 0) {
        if ((n & 1) != 0) {
            result *= base;
        }
        n >>= 1;
        if (n > 0) { // guarded: the final squaring would overflow unused
            base *= base;
        }
    }
    return result;
}

template <std::floating_point T> [[nodiscard]] constexpr auto exp(T x) noexcept -> T {
    if (is_nan(x)) {
        return x;
    }
    constexpr T ln2 = std::numbers::ln2_v<T>;
    if (x > T{1024} * ln2) {
        return std::numeric_limits<T>::infinity();
    }
    if (x < T{-1075} * ln2) {
        return T{0};
    }

    T const quotient = x / ln2;
    int const k = static_cast<int>(quotient >= T{0} ? quotient + T{0.5} : quotient - T{0.5});
    T const r = x - static_cast<T>(k) * ln2;

    T term = T{1};
    T sum = T{1};
    for (int i = 1; i <= 16; ++i) {
        term *= r / static_cast<T>(i);
        sum += term;
    }
    return ldexp(sum, k);
}

template <std::floating_point T> [[nodiscard]] constexpr auto log(T x) noexcept -> T {
    if (is_nan(x)) {
        return x;
    }
    if (x < T{0}) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == T{0}) {
        return -std::numeric_limits<T>::infinity();
    }
    if (is_inf(x)) {
        return x;
    }

    int k = 0;
    T m = x;
    while (m >= T{2}) {
        m *= T{0.5};
        ++k;
    }
    while (m < T{1}) {
        m *= T{2};
        --k;
    }

    T const s = (m - T{1}) / (m + T{1});
    T const s2 = s * s;
    T power = s;
    T sum = s;
    for (int i = 3; i <= 31; i += 2) {
        power *= s2;
        sum += power / static_cast<T>(i);
    }
    return T{2} * sum + static_cast<T>(k) * std::numbers::ln2_v<T>;
}

template <std::floating_point T> [[nodiscard]] constexpr auto sqrt(T x) noexcept -> T {
    if (is_nan(x)) {
        return x;
    }
    if (x < T{0}) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == T{0} || is_inf(x)) {
        return x;
    }

    int k = 0;
    T m = x;
    while (m >= T{2}) {
        m *= T{0.25};
        ++k;
    }
    while (m < T{0.5}) {
        m *= T{4};
        --k;
    }

    T y = T{1};
    for (int i = 0; i < 8; ++i) {
        y = T{0.5} * (y + m / y);
    }
    return ldexp(y, k);
}

template <std::floating_point T> [[nodiscard]] constexpr auto rsqrt(T x) noexcept -> T {
    return T{1} / sqrt(x);
}

template <std::floating_point T> [[nodiscard]] constexpr auto tanh(T x) noexcept -> T {
    if (is_nan(x)) {
        return x;
    }
    T const a = abs(x);
    if (a > T{20}) {
        return x < T{0} ? T{-1} : T{1};
    }
    T const e = exp(T{2} * a);
    T const t = (e - T{1}) / (e + T{1});
    return x < T{0} ? -t : t;
}

template <std::floating_point T> [[nodiscard]] constexpr auto sigmoid(T x) noexcept -> T {
    if (x >= T{0}) {
        return T{1} / (T{1} + exp(-x));
    }
    T const e = exp(x);
    return e / (T{1} + e);
}

template <std::floating_point T>
[[nodiscard]] constexpr auto pow(T base, T exponent) noexcept -> T {
    if (exponent == T{0}) {
        return T{1};
    }
    if (base == T{0}) {
        return T{0};
    }
    return exp(exponent * log(base));
}

template <std::floating_point T> [[nodiscard]] constexpr auto sin(T x) noexcept -> T {
    if (is_nan(x) || is_inf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T two_pi = T{2} * pi;

    T r = x;
    while (r > pi) {
        r -= two_pi;
    }
    while (r < -pi) {
        r += two_pi;
    }

    T term = r;
    T sum = r;
    T const r2 = r * r;
    for (int i = 1; i <= 12; ++i) {
        auto const d = static_cast<T>((2 * i) * (2 * i + 1));
        term *= -r2 / d;
        sum += term;
    }
    return sum;
}

template <std::floating_point T> [[nodiscard]] constexpr auto cos(T x) noexcept -> T {
    if (is_nan(x) || is_inf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T two_pi = T{2} * pi;

    T r = x;
    while (r > pi) {
        r -= two_pi;
    }
    while (r < -pi) {
        r += two_pi;
    }

    T term = T{1};
    T sum = T{1};
    T const r2 = r * r;
    for (int i = 1; i <= 12; ++i) {
        term *= -r2 / static_cast<T>(((2 * i) - 1) * (2 * i));
        sum += term;
    }
    return sum;
}

} // namespace poyea::shrimp::cmath