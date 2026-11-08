#include "shrimp/core/math.hpp"
#include "shrimp/model/backward.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/transformer.hpp"
#include "shrimp/ops/loss.hpp"

#include <cstddef>

namespace {

using poyea::shrimp::ForwardTape;
using poyea::shrimp::Gradients;
using poyea::shrimp::Matrix;
using poyea::shrimp::ModelConfig;
using poyea::shrimp::output_weight;
using poyea::shrimp::TokenSequence;
using poyea::shrimp::Transformer;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

constexpr ModelConfig cfg{.vocab = 4, .dim = 4, .heads = 1, .layers = 1, .max_seq = 4, .hidden = 4};

using Model = Transformer<double, cfg>;
using Grads = Gradients<double, cfg>;
using Input = TokenSequence<cfg.max_seq>;

constexpr Model model = poyea::shrimp::random_transformer<double, cfg>(3);

constexpr auto input() -> Input {
    return poyea::shrimp::tokens<cfg.max_seq>(1, 2, 3, 1);
}

constexpr auto objective(Model const& m, Input const& in) -> double {
    return poyea::shrimp::sequence_loss(poyea::shrimp::forward_train(m, in), in);
}

constexpr auto prefix(Input const& in, std::size_t length) -> Input {
    Input out{};
    for (std::size_t i = 0; i < length; ++i) {
        out.push(in[i]);
    }
    return out;
}

// Central differences on the first three entries of one tensor, starting at `first`.
template <typename Select>
constexpr auto matches_fd(Model const& base, Input const& in, Select select,
                          std::size_t first) -> bool {
    constexpr double h = 1e-5;
    constexpr double tolerance = 1e-6;

    Grads grads{};
    auto const tape = poyea::shrimp::forward_train(base, in);
    poyea::shrimp::backward(base, in, tape, grads);

    for (std::size_t i = first; i < first + 3; ++i) {
        Model plus = base;
        Model minus = base;
        select(plus).storage[i] += h;
        select(minus).storage[i] -= h;
        double const numeric = (objective(plus, in) - objective(minus, in)) / (2.0 * h);
        double const analytic = select(grads).storage[i];
        if (!near(numeric, analytic, tolerance)) {
            return false;
        }
    }
    return true;
}

// forward_train must agree with the existing last-position forward at every prefix.
constexpr auto tape_matches_forward() -> bool {
    auto const in = input();
    auto const tape = poyea::shrimp::forward_train(model, in);
    for (std::size_t t = 0; t < tape.count; ++t) {
        auto const expected = poyea::shrimp::forward(model, prefix(in, t + 1));
        for (std::size_t i = 0; i < cfg.vocab; ++i) {
            if (!near(tape.logits.storage[(t * cfg.vocab) + i], expected.storage[i], 1e-12)) {
                return false;
            }
        }
    }
    return true;
}

constexpr auto loss_is_mean_of_cross_entropy() -> bool {
    auto const in = input();
    auto const tape = poyea::shrimp::forward_train(model, in);
    double total = 0.0;
    for (std::size_t t = 0; t + 1 < tape.count; ++t) {
        total += poyea::shrimp::ops::cross_entropy(poyea::shrimp::row(tape.logits, t), in[t + 1]);
    }
    total /= static_cast<double>(tape.count - 1);
    return near(poyea::shrimp::sequence_loss(tape, in), total, 1e-12);
}

constexpr auto backward_returns_loss() -> bool {
    auto const in = input();
    auto const tape = poyea::shrimp::forward_train(model, in);
    Grads grads{};
    return poyea::shrimp::backward(model, in, tape, grads) ==
           poyea::shrimp::sequence_loss(tape, in);
}

constexpr auto short_sequence_has_zero_loss() -> bool {
    auto const one = poyea::shrimp::tokens<cfg.max_seq>(2);
    auto const empty = poyea::shrimp::tokens<cfg.max_seq>();
    Grads grads{};
    auto const tape_one = poyea::shrimp::forward_train(model, one);
    auto const tape_empty = poyea::shrimp::forward_train(model, empty);
    bool const loss_zero = poyea::shrimp::sequence_loss(tape_one, one) == 0.0 &&
                           poyea::shrimp::sequence_loss(tape_empty, empty) == 0.0;
    bool const back_zero = poyea::shrimp::backward(model, one, tape_one, grads) == 0.0 &&
                           poyea::shrimp::backward(model, empty, tape_empty, grads) == 0.0;
    bool grads_zero = true;
    for (auto const v : grads.embedding.storage) {
        grads_zero = grads_zero && v == 0.0;
    }
    for (auto const v : output_weight(grads).storage) {
        grads_zero = grads_zero && v == 0.0;
    }
    return loss_zero && back_zero && grads_zero;
}

constexpr auto backward_accumulates() -> bool {
    auto const in = input();
    auto const tape = poyea::shrimp::forward_train(model, in);
    Grads grads{};
    poyea::shrimp::backward(model, in, tape, grads);
    Grads const once = grads;
    poyea::shrimp::backward(model, in, tape, grads);

    for (std::size_t i = 0; i < 3; ++i) {
        if (!near(grads.embedding.storage[i], 2.0 * once.embedding.storage[i], 1e-12)) {
            return false;
        }
        if (!near(output_weight(grads).storage[i], 2.0 * output_weight(once).storage[i], 1e-12)) {
            return false;
        }
        if (!near(grads.blocks[0].attn.wq.storage[i], 2.0 * once.blocks[0].attn.wq.storage[i],
                  1e-12)) {
            return false;
        }
        if (!near(grads.blocks[0].ffn.down.storage[i], 2.0 * once.blocks[0].ffn.down.storage[i],
                  1e-12)) {
            return false;
        }
    }
    return true;
}

// Logits at the final position have no target, so perturbing them must not move the loss.
constexpr auto last_position_logits_ignored() -> bool {
    auto const in = input();
    auto tape = poyea::shrimp::forward_train(model, in);
    double const base = poyea::shrimp::sequence_loss(tape, in);
    for (std::size_t i = 0; i < cfg.vocab; ++i) {
        tape.logits.storage[((tape.count - 1) * cfg.vocab) + i] += 5.0;
    }
    return poyea::shrimp::sequence_loss(tape, in) == base;
}

static_assert(tape_matches_forward());
static_assert(loss_is_mean_of_cross_entropy());
static_assert(backward_returns_loss());
static_assert(short_sequence_has_zero_loss());
static_assert(backward_accumulates());
static_assert(last_position_logits_ignored());

static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.embedding; }, 0));
// Token 1 repeats at positions 0 and 3, so its row receives two scatter contributions.
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.embedding; }, cfg.dim));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return output_weight(m); }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.final_norm.gain; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].attn.wq; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].attn.wk; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].attn.wv; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].attn.wo; }, 0));
static_assert(
    matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].attn_norm.gain; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].ffn.gate; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].ffn.up; }, 0));
static_assert(matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].ffn.down; }, 0));
static_assert(
    matches_fd(model, input(), [](auto& m) -> auto& { return m.blocks[0].ffn_norm.gain; }, 0));

} // namespace

auto main() -> int {
    return 0;
}
