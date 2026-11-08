#include "shrimp/core/math.hpp"
#include "shrimp/model/backward.hpp"
#include "shrimp/model/gpt2.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/train.hpp"
#include "shrimp/model/transformer.hpp"
#include "shrimp/ops/linalg.hpp"

#include <cstddef>
#include <cstdint>

namespace {

namespace gpt2 = poyea::shrimp::gpt2;

using poyea::shrimp::Act;
using poyea::shrimp::Corpus;
using poyea::shrimp::Gradients;
using poyea::shrimp::ModelConfig;
using poyea::shrimp::Norm;
using poyea::shrimp::Pos;
using poyea::shrimp::TokenSequence;
using poyea::shrimp::Transformer;
using poyea::shrimp::output_weight;
using poyea::shrimp::random_transformer;
using poyea::shrimp::row;
using poyea::shrimp::tokens;
using poyea::shrimp::train;
using poyea::shrimp::ops::linear;

template <typename T> [[nodiscard]] constexpr auto near(T a, T b, T tolerance) noexcept -> bool {
    return poyea::shrimp::cmath::abs(a - b) <= tolerance;
}

// ------------------------------------------------------- published shapes
namespace shapes {

static_assert(gpt2::small.valid());
static_assert(gpt2::medium.valid());
static_assert(gpt2::large.valid());
static_assert(gpt2::xl.valid());

// The published counts. Positions, projection biases and the tied head all
// enter these totals, so matching them pins the accounting down.
static_assert(gpt2::small.parameters() == 124'439'808);
static_assert(gpt2::medium.parameters() == 354'823'168);
static_assert(gpt2::large.parameters() == 774'030'080);
static_assert(gpt2::xl.parameters() == 1'557'611'200);

constexpr ModelConfig base = gpt2::small;

constexpr auto with(ModelConfig c, Pos pos, bool bias, bool tie) -> ModelConfig {
    c.pos = pos;
    c.bias = bias;
    c.tie_embeddings = tie;
    return c;
}

constexpr std::size_t bias_weights = base.layers * ((4 * base.dim) + base.hidden + base.dim);

// Each flag accounts for exactly its own weights.
static_assert(with(base, Pos::RoPE, true, true).parameters() ==
              base.parameters() - (base.max_seq * base.dim));
static_assert(with(base, Pos::Learned, true, false).parameters() ==
              base.parameters() + (base.vocab * base.dim));
static_assert(with(base, Pos::Learned, false, true).parameters() ==
              base.parameters() - bias_weights);

static_assert(base.norm == Norm::Layer);
static_assert(base.act == Act::GELU);
static_assert(base.hidden == 4 * base.dim);
static_assert(base.kv_head_count() == base.heads);

} // namespace shapes

// ---------------------------------------------------------------- behaviour
namespace behaviour {

constexpr ModelConfig cfg = gpt2::config;

using Model = Transformer<double, cfg>;
using Input = TokenSequence<cfg.max_seq>;

constexpr Model model = random_transformer<double, cfg>(7);

constexpr auto input() -> Input {
    return tokens<cfg.max_seq>(1, 2, 3, 1, 2, 3);
}

static_assert(cfg.parameters() == 6976);

constexpr auto logits_are_finite() -> bool {
    auto const out = poyea::shrimp::forward(model, input());
    for (std::size_t v = 0; v < cfg.vocab; ++v) {
        double const logit = out.storage[v];
        if (poyea::shrimp::cmath::is_nan(logit) || poyea::shrimp::cmath::is_inf(logit)) {
            return false;
        }
    }
    return true;
}

// The tied head is the embedding matrix, so logits are E times the normed state.
constexpr auto tied_head_is_the_embedding() -> bool {
    auto const in = input();
    auto const tape = poyea::shrimp::forward_train(model, in);
    std::size_t const last = tape.count - 1;
    auto const expected = linear(model.embedding, row(tape.final_normed, last));
    for (std::size_t v = 0; v < cfg.vocab; ++v) {
        if (!near(tape.logits.storage[(last * cfg.vocab) + v], expected.storage[v], 1e-12)) {
            return false;
        }
    }
    return true;
}

// Position and bias weights are not decoration: zeroing or moving them moves the logits.
constexpr auto positions_change_the_output() -> bool {
    Model flat = model;
    for (auto& entry : flat.positions.value.storage) {
        entry = 0.0;
    }
    auto const before = poyea::shrimp::forward(model, input());
    auto const after = poyea::shrimp::forward(flat, input());
    for (std::size_t v = 0; v < cfg.vocab; ++v) {
        if (!near(before.storage[v], after.storage[v], 1e-9)) {
            return true;
        }
    }
    return false;
}

constexpr auto bias_changes_the_output() -> bool {
    Model shifted = model;
    shifted.blocks[0].attn.bo.value.storage[0] += 1.0;
    auto const plain = poyea::shrimp::forward(model, input());
    auto const other = poyea::shrimp::forward(shifted, input());
    for (std::size_t v = 0; v < cfg.vocab; ++v) {
        if (!near(plain.storage[v], other.storage[v], 1e-9)) {
            return true;
        }
    }
    return false;
}

static_assert(logits_are_finite());
static_assert(tied_head_is_the_embedding());
static_assert(positions_change_the_output());
static_assert(bias_changes_the_output());

} // namespace behaviour

// ----------------------------------------------------------- new gradients
namespace gradients {

// Smallest shape that still carries every GPT-2 feature, so the finite
// differences stay inside the step budget.
constexpr ModelConfig cfg = gpt2::shape(8, 8, 2, 1, 6);

using Model = Transformer<double, cfg>;
using Grads = Gradients<double, cfg>;
using Input = TokenSequence<cfg.max_seq>;

constexpr Model model = random_transformer<double, cfg>(5);

constexpr auto input() -> Input {
    return tokens<cfg.max_seq>(1, 2, 3, 1, 2, 3);
}

constexpr auto objective(Model const& m, Input const& in) -> double {
    return poyea::shrimp::sequence_loss(poyea::shrimp::forward_train(m, in), in);
}

template <typename Select>
constexpr auto matches_fd(Select select, std::size_t first) -> bool {
    constexpr double h = 1e-5;
    constexpr double tolerance = 1e-6;
    auto const in = input();

    Grads grads{};
    auto const tape = poyea::shrimp::forward_train(model, in);
    poyea::shrimp::backward(model, in, tape, grads);

    for (std::size_t i = first; i < first + 3; ++i) {
        Model plus = model;
        Model minus = model;
        select(plus).storage[i] += h;
        select(minus).storage[i] -= h;
        double const numeric = (objective(plus, in) - objective(minus, in)) / (2.0 * h);
        if (!near(numeric, select(grads).storage[i], tolerance)) {
            return false;
        }
    }
    return true;
}

static_assert(matches_fd([](auto& m) -> auto& { return m.positions.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.positions.value; }, 2 * cfg.dim));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].attn.bq.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].attn.bk.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].attn.bv.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].attn.bo.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].ffn.bup.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].ffn.bdown.value; }, 0));
static_assert(matches_fd([](auto& m) -> auto& { return m.blocks[0].attn_norm.bias; }, 0));

// Tied: the embedding gradient carries the scatter path and the output path at once.
static_assert(matches_fd([](auto& m) -> auto& { return output_weight(m); }, cfg.dim));

} // namespace gradients

// ------------------------------------------------------------------ training
namespace training {

constexpr ModelConfig cfg = gpt2::shape(8, 8, 2, 1, 6);
constexpr std::size_t steps = 12;
constexpr std::uint64_t seed = 5;
constexpr double rate = 0.1;

constexpr auto make_corpus() -> Corpus<cfg.max_seq, 1> {
    Corpus<cfg.max_seq, 1> corpus{};
    corpus.samples[0] = tokens<cfg.max_seq>(0, 1, 2, 1, 2, 1);
    return corpus;
}

constexpr auto corpus = make_corpus();
constexpr auto trained = train<steps>(random_transformer<double, cfg>(seed), corpus, rate);

constexpr auto losses_are_finite() -> bool {
    for (std::size_t s = 0; s < steps; ++s) {
        double const loss = trained.loss[s];
        if (poyea::shrimp::cmath::is_nan(loss) || poyea::shrimp::cmath::is_inf(loss)) {
            return false;
        }
    }
    return true;
}

static_assert(losses_are_finite());
static_assert(trained.loss[steps - 1] < trained.loss[0]);

} // namespace training

} // namespace

auto main() -> int {
    return 0;
}
