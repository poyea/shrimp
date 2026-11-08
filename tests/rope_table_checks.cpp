#include "shrimp/core/math.hpp"
#include "shrimp/ops/act_grad.hpp"
#include "shrimp/ops/nn.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::Vector;
using poyea::shrimp::ops::make_rope_table;
using poyea::shrimp::ops::rope;
using poyea::shrimp::ops::rope_backward;
using poyea::shrimp::ops::rope_table;

constexpr std::size_t head_dim = 8;
constexpr std::size_t max_seq = 12;
constexpr double theta = 1e6;

// Forces the table through constant evaluation: a constexpr variable cannot be
// initialised at runtime, so this failing to compile means the table escaped.
constexpr auto table = make_rope_table<double, head_dim, max_seq>(theta);
static_assert(table.cosines.storage[0] == 1.0);
static_assert(table.sines.storage[0] == 0.0);
static_assert(decltype(table.cosines)::extent<0> == max_seq);
static_assert(decltype(table.cosines)::extent<1> == head_dim / 2);

// The namespace-scope variable template must be usable in a constant expression.
static_assert(rope_table<double, head_dim, max_seq, theta>.cosines.storage[0] == 1.0);

constexpr Vector<double, head_dim> x{{0.7, -2.3, 1.1, 0.05, 1.9, -0.4, 0.33, -1.25}};
constexpr Vector<double, head_dim> seed{{0.2, -1.1, 0.6, 1.7, -0.8, 0.45, 2.2, -0.15}};

// Table and formula must agree exactly: identical expressions, identical values.
constexpr auto table_matches_formula() -> bool {
    for (std::size_t p = 0; p < max_seq; ++p) {
        auto const viaTable = rope(x, table, p);
        auto const viaFormula = rope(x, p, theta);
        for (std::size_t i = 0; i < head_dim; ++i) {
            if (viaTable.storage[i] != viaFormula.storage[i]) {
                return false;
            }
        }
    }
    return true;
}
static_assert(table_matches_formula());

constexpr auto backward_table_matches_formula() -> bool {
    for (std::size_t p = 0; p < max_seq; ++p) {
        auto const viaTable = rope_backward(seed, table, p);
        auto const viaFormula = rope_backward(seed, p, theta);
        for (std::size_t i = 0; i < head_dim; ++i) {
            if (viaTable.storage[i] != viaFormula.storage[i]) {
                return false;
            }
        }
    }
    return true;
}
static_assert(backward_table_matches_formula());

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

// Orthogonality through the table path.
constexpr auto round_trip_is_identity() -> bool {
    for (std::size_t p = 0; p < max_seq; ++p) {
        auto const back = rope_backward(rope(x, table, p), table, p);
        for (std::size_t i = 0; i < head_dim; ++i) {
            if (!near(back.storage[i], x.storage[i], 1e-12)) {
                return false;
            }
        }
    }
    return true;
}
static_assert(round_trip_is_identity());

constexpr auto position_zero_is_identity() -> bool {
    auto const out = rope(x, table, 0);
    for (std::size_t i = 0; i < head_dim; ++i) {
        if (out.storage[i] != x.storage[i]) {
            return false;
        }
    }
    return true;
}
static_assert(position_zero_is_identity());

} // namespace

auto main() -> int {
    return 0;
}
