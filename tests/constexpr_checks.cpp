#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/linalg.hpp"
#include "shrimp/ops/nn.hpp"

#include <cstddef>
#include <numbers>

namespace {

using poyea::shrimp::Matrix;
using poyea::shrimp::Tensor;
using poyea::shrimp::Vector;

template <typename T>
[[nodiscard]] constexpr auto near(T a, T b, T tolerance = T{1e-9}) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

// ---------------------------------------------------------------------------
// Transcendentals
// ---------------------------------------------------------------------------
namespace math_checks {

using namespace poyea::shrimp::cmath;

static_assert(exp(0.0) == 1.0);
static_assert(near(exp(1.0), std::numbers::e_v<double>, 1e-14));
static_assert(near(exp(-1.0), 1.0 / std::numbers::e_v<double>, 1e-14));
static_assert(near(exp(10.0), 22026.465794806718, 1e-9));

static_assert(near(log(std::numbers::e_v<double>), 1.0, 1e-14));
static_assert(near(log(1.0), 0.0, 1e-15));
static_assert(near(log(1000.0), 6.907755278982137, 1e-13));

static_assert(near(log(exp(7.5)), 7.5, 1e-12));
static_assert(near(exp(log(1234.5)), 1234.5, 1e-9));

static_assert(near(sqrt(4.0), 2.0, 1e-15));
static_assert(near(sqrt(2.0), 1.4142135623730951, 1e-15));
static_assert(near(sqrt(1e-8), 1e-4, 1e-16));
static_assert(near(sqrt(1e8), 1e4, 1e-9));
static_assert(near(rsqrt(16.0), 0.25, 1e-15));

static_assert(tanh(0.0) == 0.0);
static_assert(near(tanh(1.0), 0.7615941559557649, 1e-14));
static_assert(tanh(50.0) == 1.0);
static_assert(tanh(-50.0) == -1.0);

static_assert(near(sigmoid(0.0), 0.5, 1e-15));
static_assert(near(sigmoid(2.0), 0.8807970779778823, 1e-14));
static_assert(near(sigmoid(-2.0), 1.0 - 0.8807970779778823, 1e-14));

static_assert(near(pow(2.0, 10.0), 1024.0, 1e-9));
static_assert(near(pow(10000.0, 0.25), 10.0, 1e-11));

static_assert(near(sin(0.0), 0.0, 1e-15));
static_assert(near(sin(std::numbers::pi_v<double> / 2), 1.0, 1e-14));
static_assert(near(cos(0.0), 1.0, 1e-14));
static_assert(near((sin(0.7) * sin(0.7)) + (cos(0.7) * cos(0.7)), 1.0, 1e-13));

static_assert(ldexp(3.0, 4) == 48.0);
static_assert(ldexp(3.0, -2) == 0.75);
} // namespace math_checks

// ---------------------------------------------------------------------------
// Tensor shape algebra
// ---------------------------------------------------------------------------
namespace tensor_checks {

using T3x4 = Tensor<double, 3, 4>;
static_assert(T3x4::rank == 2);
static_assert(T3x4::size == 12);
static_assert(T3x4::extent<0> == 3);
static_assert(T3x4::extent<1> == 4);
static_assert(T3x4::strides[0] == 4);
static_assert(T3x4::strides[1] == 1);

using T2x3x4 = Tensor<double, 2, 3, 4>;
static_assert(T2x3x4::size == 24);
static_assert(T2x3x4::strides[0] == 12);
static_assert(T2x3x4::strides[1] == 4);
static_assert(T2x3x4::strides[2] == 1);

constexpr auto subscript_roundtrip() -> bool {
    T3x4 t{};
    t[1, 2] = 7.0;
    return t.storage[(1 * 4) + 2] == 7.0 && t[1, 2] == 7.0;
}
static_assert(subscript_roundtrip());

constexpr auto filled_is_uniform() -> bool {
    auto const t = Tensor<double, 2, 5>::filled(1.5);
    for (std::size_t i = 0; i < Tensor<double, 2, 5>::size; ++i) {
        if (t.storage[i] != 1.5) {
            return false;
        }
    }
    return true;
}
static_assert(filled_is_uniform());

constexpr auto reshape_preserves_order() -> bool {
    Tensor<double, 2, 6> t{};
    for (std::size_t i = 0; i < 12; ++i) {
        t.storage[i] = static_cast<double>(i);
    }
    auto const r = t.reshape<3, 4>();
    return r[0, 0] == 0.0 && r[1, 0] == 4.0 && r[2, 3] == 11.0;
}
static_assert(reshape_preserves_order());

constexpr auto row_roundtrip() -> bool {
    Matrix<double, 3, 2> m{};
    poyea::shrimp::set_row(m, 1, Vector<double, 2>{{8.0, 9.0}});
    auto const r = poyea::shrimp::row(m, 1);
    return r.storage[0] == 8.0 && r.storage[1] == 9.0 && m[1, 0] == 8.0;
}
static_assert(row_roundtrip());
} // namespace tensor_checks

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------
namespace linalg_checks {

using namespace poyea::shrimp::ops;

constexpr Vector<double, 3> v_a{{1.0, 2.0, 3.0}};
constexpr Vector<double, 3> v_b{{4.0, 5.0, 6.0}};
static_assert(dot(v_a, v_b) == 32.0);

// hand computed
constexpr Matrix<double, 2, 3> m_a{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
constexpr Matrix<double, 3, 2> m_b{{7.0, 8.0, 9.0, 10.0, 11.0, 12.0}};
constexpr auto m_prod = matmul(m_a, m_b);
static_assert(decltype(m_prod)::extent<0> == 2);
static_assert(decltype(m_prod)::extent<1> == 2);
static_assert(m_prod[0, 0] == 58.0);  // 1*7 + 2*9 + 3*11
static_assert(m_prod[0, 1] == 64.0);  // 1*8 + 2*10 + 3*12
static_assert(m_prod[1, 0] == 139.0); // 4*7 + 5*9 + 6*11
static_assert(m_prod[1, 1] == 154.0); // 4*8 + 5*10 + 6*12

constexpr Matrix<double, 2, 2> m_c{{1.0, 0.5, -2.0, 3.0}};
static_assert(matmul(matmul(m_a, m_b), m_c)[0, 0] == matmul(m_a, matmul(m_b, m_c))[0, 0]);

// weights are [Out, In]
constexpr Matrix<double, 2, 3> weight{{1.0, 0.0, 0.0, 0.0, 1.0, 1.0}};
constexpr auto projected = linear(weight, v_a);
static_assert(projected.storage[0] == 1.0); // picks x[0]
static_assert(projected.storage[1] == 5.0); // x[1] + x[2]

constexpr Vector<double, 2> bias{{10.0, 20.0}};
static_assert(linear(weight, bias, v_a).storage[0] == 11.0);
static_assert(linear(weight, bias, v_a).storage[1] == 25.0);

static_assert(transpose(m_a)[2, 1] == m_a[1, 2]);
static_assert(decltype(transpose(m_a))::extent<0> == 3);

static_assert(add(v_a, v_b).storage[2] == 9.0);
static_assert(hadamard(v_a, v_b).storage[2] == 18.0);
static_assert(scale(v_a, 2.0).storage[1] == 4.0);

static_assert(argmax(Vector<double, 4>{{0.1, 0.9, 0.3, 0.2}}) == 1);
static_assert(argmax(Vector<double, 3>{{0.5, 0.5, 0.1}}) == 0);
} // namespace linalg_checks

// ---------------------------------------------------------------------------
// Neural network kernels
// ---------------------------------------------------------------------------
namespace nn_checks {

using namespace poyea::shrimp::ops;

constexpr Vector<double, 4> logits{{1.0, 2.0, 3.0, 4.0}};
constexpr auto probs = softmax(logits);

constexpr auto softmax_normalizes() -> bool {
    double total = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        total += probs.storage[i];
    }
    return near(total, 1.0, 1e-14);
}
static_assert(softmax_normalizes());
static_assert(near(probs.storage[3], 0.6439142598879722, 1e-13));
static_assert(probs.storage[0] < probs.storage[3]);
static_assert(argmax(probs) == argmax(logits));

constexpr Vector<double, 4> shifted{{101.0, 102.0, 103.0, 104.0}};
static_assert(near(softmax(shifted).storage[3], probs.storage[3], 1e-14));
constexpr Vector<double, 3> extreme{{1000.0, 1001.0, 999.0}};
static_assert(!poyea::shrimp::cmath::is_nan(softmax(extreme).storage[0]));

constexpr Vector<double, 4> scores{{1.0, 2.0, -5.0, -5.0}};
constexpr auto prefix = softmax_prefix(scores, 2);
static_assert(near(prefix.storage[0] + prefix.storage[1], 1.0, 1e-14));
static_assert(prefix.storage[2] == 0.0); // masked positions stay exactly zero
static_assert(near(prefix.storage[1], softmax(Vector<double, 2>{{1.0, 2.0}}).storage[1], 1e-14));

constexpr Vector<double, 4> raw{{3.0, -4.0, 1.0, 2.0}};
constexpr auto unit_gain = Vector<double, 4>::filled(1.0);
constexpr auto normed = rmsnorm(raw, unit_gain, 0.0);

constexpr auto rms_is_unit() -> bool {
    double sum_squares = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum_squares += normed.storage[i] * normed.storage[i];
    }
    return near(poyea::shrimp::cmath::sqrt(sum_squares / 4.0), 1.0, 1e-12);
}
static_assert(rms_is_unit());
static_assert(near(rmsnorm(scale(raw, 7.0), unit_gain, 0.0).storage[0], normed.storage[0], 1e-12));

constexpr auto zero_bias = Vector<double, 4>::filled(0.0);
constexpr auto ln = layernorm(raw, unit_gain, zero_bias, 0.0);

constexpr auto layernorm_moments() -> bool {
    double mean = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        mean += ln.storage[i];
    }
    mean /= 4.0;
    double variance = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        variance += (ln.storage[i] - mean) * (ln.storage[i] - mean);
    }
    return near(mean, 0.0, 1e-13) && near(variance / 4.0, 1.0, 1e-12);
}
static_assert(layernorm_moments());

static_assert(gelu_scalar(0.0) == 0.0);
static_assert(near(gelu_scalar(1.0), 0.8411919906082768, 1e-9));
static_assert(gelu_scalar(-5.0) < 0.0 && gelu_scalar(-5.0) > -0.01);
static_assert(near(gelu_scalar(8.0), 8.0, 1e-9));

static_assert(silu_scalar(0.0) == 0.0);
static_assert(near(silu_scalar(1.0), 0.7310585786300049, 1e-14));
static_assert(near(silu_scalar(-20.0), 0.0, 1e-7));

constexpr Vector<double, 4> head{{1.0, 2.0, 3.0, 4.0}};
constexpr auto rotated = rope(head, 5, 10000.0);

constexpr auto rope_is_isometric() -> bool {
    double before = 0.0;
    double after = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        before += head.storage[i] * head.storage[i];
        after += rotated.storage[i] * rotated.storage[i];
    }
    return near(before, after, 1e-12);
}
static_assert(rope_is_isometric());
static_assert(near(rope(head, 0, 10000.0).storage[0], head.storage[0], 1e-15));
static_assert(near(rope(head, 0, 10000.0).storage[3], head.storage[3], 1e-15));
static_assert(near(rope(rope(head, 3, 10000.0), 4, 10000.0).storage[1],
                   rope(head, 7, 10000.0).storage[1], 1e-12));
} // namespace nn_checks

} // namespace

auto main() -> int {
    return 0;
}
