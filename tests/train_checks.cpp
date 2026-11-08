#include "shrimp/core/math.hpp"
#include "shrimp/model/init.hpp"
#include "shrimp/model/train.hpp"
#include "shrimp/model/transformer.hpp"

#include <cstddef>
#include <cstdint>

namespace {

using poyea::shrimp::Corpus;
using poyea::shrimp::ModelConfig;
using poyea::shrimp::generate;
using poyea::shrimp::random_transformer;
using poyea::shrimp::tokens;
using poyea::shrimp::train;

constexpr ModelConfig cfg{.vocab = 4, .dim = 8, .heads = 1, .layers = 1, .max_seq = 6, .hidden = 8};
constexpr std::size_t steps = 20;
constexpr std::uint64_t seed = 1;
constexpr double rate = 0.3;

constexpr auto make_corpus() -> Corpus<cfg.max_seq, 1> {
    Corpus<cfg.max_seq, 1> corpus{};
    corpus.samples[0] = tokens<cfg.max_seq>(0, 1, 2, 1, 2, 1);
    return corpus;
}

constexpr auto corpus = make_corpus();
constexpr auto trained = train<steps>(random_transformer<double, cfg>(seed), corpus, rate);

static_assert(trained.loss[0] > 1.0);
static_assert(trained.loss[steps / 2] < trained.loss[0]);
static_assert(trained.loss[steps - 1] < 0.25 * trained.loss[0]);
static_assert(!poyea::shrimp::cmath::is_nan(trained.loss[steps - 1]));

static_assert(generate<3>(trained.model, tokens<cfg.max_seq>(0, 1)) ==
              tokens<cfg.max_seq>(0, 1, 2, 1, 2));

// Training must have done the work: the untrained weights do not produce the pattern.
static_assert(generate<3>(random_transformer<double, cfg>(seed), tokens<cfg.max_seq>(0, 1)) !=
              tokens<cfg.max_seq>(0, 1, 2, 1, 2));

} // namespace

auto main() -> int {
    return 0;
}
