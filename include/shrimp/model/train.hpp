#pragma once

#include "shrimp/model/backward.hpp"
#include "shrimp/model/gradients.hpp"
#include "shrimp/model/transformer.hpp"

#include <array>
#include <cstddef>

namespace poyea::shrimp {

template <std::size_t Capacity, std::size_t Count> struct Corpus {
    std::array<TokenSequence<Capacity>, Count> samples{};
};

template <typename T, ModelConfig C, std::size_t Steps> struct TrainResult {
    Transformer<T, C> model{};
    std::array<T, Steps> loss{};
};

template <std::size_t Steps, typename T, ModelConfig C, std::size_t Capacity, std::size_t Count>
[[nodiscard]] constexpr auto train(Transformer<T, C> model, Corpus<Capacity, Count> const& corpus,
                                   T rate) -> TrainResult<T, C, Steps> {
    static_assert(Count > 0, "training needs at least one sample");
    TrainResult<T, C, Steps> result{};
    result.model = model;

    Gradients<T, C> grads{};
    for (std::size_t step = 0; step < Steps; ++step) {
        auto const& sample = corpus.samples[step % Count];
        zero_gradients(grads);
        auto const tape = forward_train(result.model, sample);
        result.loss[step] = backward(result.model, sample, tape, grads);
        sgd_step(result.model, grads, rate);
    }
    return result;
}

} // namespace poyea::shrimp
