#include "shrimp/core/math.hpp"
#include "shrimp/model/induction.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>

namespace {

constexpr auto cfg = poyea::shrimp::induction::config;
constexpr std::size_t cap = cfg.max_seq;

using poyea::shrimp::tokens;

static_assert(poyea::shrimp::induction::construction_is_sound<double>(cfg));
static_assert(cfg.dim == 2 * cfg.vocab);
static_assert(cfg.parameters() == 2480);

constexpr auto model = poyea::shrimp::induction::build<double>();

// --------------------------------------------------------------- next token
namespace prediction {

using poyea::shrimp::ops::argmax;
using poyea::shrimp::forward;

// period 3: 1 2 3 1 -> 2
static_assert(argmax(forward(model, tokens<cap>(0, 1, 2, 3, 1))) == 2);
static_assert(argmax(forward(model, tokens<cap>(0, 1, 2, 3, 1, 2))) == 3);
static_assert(argmax(forward(model, tokens<cap>(0, 1, 2, 3, 1, 2, 3))) == 1);

// period 2
static_assert(argmax(forward(model, tokens<cap>(0, 1, 2, 1))) == 2);
static_assert(argmax(forward(model, tokens<cap>(0, 1, 2, 1, 2))) == 1);

static_assert(argmax(forward(model, tokens<cap>(0, 4, 7, 5, 4))) == 7);
static_assert(argmax(forward(model, tokens<cap>(0, 6, 3, 6))) == 3);

constexpr auto margin_is_decisive() -> bool {
    auto const logits = forward(model, tokens<cap>(0, 1, 2, 3, 1));
    std::size_t const best = argmax(logits);
    double runner_up = -1e300;
    for (std::size_t v = 0; v < cfg.vocab; ++v) {
        if (v != best && logits.storage[v] > runner_up) {
            runner_up = logits.storage[v];
        }
    }
    return (logits.storage[best] - runner_up) > 1.0;
}
static_assert(margin_is_decisive());
} // namespace prediction

// ---------------------------------------------------------------- generation
namespace generation {

using poyea::shrimp::generate;

static_assert(generate<3>(model, tokens<cap>(0, 1, 2, 3, 1)) ==
              tokens<cap>(0, 1, 2, 3, 1, 2, 3, 1));

static_assert(generate<4>(model, tokens<cap>(0, 1, 2, 1)) ==
              tokens<cap>(0, 1, 2, 1, 2, 1, 2, 1));

static_assert(generate<3>(model, tokens<cap>(0, 4, 7, 5, 4)) ==
              tokens<cap>(0, 4, 7, 5, 4, 7, 5, 4));

static_assert(generate<100>(model, tokens<cap>(0, 1, 2, 1)).size() == cfg.max_seq);
} // namespace generation

// ------------------------------------------------------------- random init
namespace random_init {

constexpr poyea::shrimp::ModelConfig tiny{
    .vocab = 8, .dim = 8, .heads = 1, .layers = 1, .max_seq = 4, .hidden = 8};

constexpr auto a = poyea::shrimp::random_transformer<double, tiny>(42);
constexpr auto b = poyea::shrimp::random_transformer<double, tiny>(42);
constexpr auto c = poyea::shrimp::random_transformer<double, tiny>(43);

constexpr auto same_seed_is_identical() -> bool {
    for (std::size_t i = 0; i < decltype(a.embedding)::size; ++i) {
        if (a.embedding.storage[i] != b.embedding.storage[i]) {
            return false;
        }
    }
    return true;
}
static_assert(same_seed_is_identical());
static_assert(a.embedding.storage[0] != c.embedding.storage[0]);

constexpr auto logits_are_well_formed() -> bool {
    auto const logits = poyea::shrimp::forward(a, tokens<4>(1, 2, 3));
    bool varies = false;
    for (std::size_t v = 0; v < tiny.vocab; ++v) {
        if (poyea::shrimp::cmath::is_nan(logits.storage[v]) ||
            poyea::shrimp::cmath::is_inf(logits.storage[v])) {
            return false;
        }
        if (logits.storage[v] != logits.storage[0]) {
            varies = true;
        }
    }
    return varies;
}
static_assert(logits_are_well_formed());
} // namespace random_init

} // namespace

auto main() -> int {
    return 0;
}
