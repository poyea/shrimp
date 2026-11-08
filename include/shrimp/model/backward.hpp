#pragma once

#include "shrimp/core/tensor.hpp"
#include "shrimp/model/attn_backward.hpp"
#include "shrimp/model/ffn_backward.hpp"
#include "shrimp/model/gradients.hpp"
#include "shrimp/model/transformer.hpp"
#include "shrimp/ops/linalg.hpp"
#include "shrimp/ops/linalg_grad.hpp"
#include "shrimp/ops/loss.hpp"

#include <array>
#include <cstddef>

namespace poyea::shrimp {

template <typename T, ModelConfig C> struct ForwardTape {
    std::array<Matrix<T, C.max_seq, C.dim>, C.layers> before_attn{};
    std::array<Matrix<T, C.max_seq, C.dim>, C.layers> before_ffn{};
    Matrix<T, C.max_seq, C.dim> final_stream{};
    Matrix<T, C.max_seq, C.dim> final_normed{};
    Matrix<T, C.max_seq, C.vocab> logits{};
    std::size_t count = 0;
};

template <typename T, ModelConfig C, std::size_t Capacity>
[[nodiscard]] constexpr auto
forward_train(Transformer<T, C> const& model,
              TokenSequence<Capacity> const& input) -> ForwardTape<T, C> {
    ForwardTape<T, C> tape{};
    std::size_t const count = input.size() < C.max_seq ? input.size() : C.max_seq;
    tape.count = count;

    auto stream = embed<T, C>(model, input, count);

    for (std::size_t layer = 0; layer < C.layers; ++layer) {
        auto const& block = model.blocks[layer];
        tape.before_attn[layer] = stream;
        attend<T, C>(block.attn, stream, block.attn_norm, count);
        tape.before_ffn[layer] = stream;
        feed_forward<T, C>(block.ffn, stream, block.ffn_norm, count);
    }
    tape.final_stream = stream;

    constexpr T eps = static_cast<T>(C.norm_eps);
    for (std::size_t t = 0; t < count; ++t) {
        auto const normed = normalize(model.final_norm, row(stream, t), eps);
        set_row(tape.final_normed, t, normed);
        set_row(tape.logits, t, ops::linear(output_weight(model), normed));
    }
    return tape;
}

template <typename T, ModelConfig C, std::size_t Capacity>
[[nodiscard]] constexpr auto sequence_loss(ForwardTape<T, C> const& tape,
                                           TokenSequence<Capacity> const& input) -> T {
    if (tape.count < 2) {
        return T{0};
    }
    T total{};
    for (std::size_t t = 0; t + 1 < tape.count; ++t) {
        total += ops::cross_entropy(row(tape.logits, t), input[t + 1]);
    }
    return total / static_cast<T>(tape.count - 1);
}

template <typename T, ModelConfig C, std::size_t Capacity>
constexpr auto backward(Transformer<T, C> const& model, TokenSequence<Capacity> const& input,
                        ForwardTape<T, C> const& tape, Gradients<T, C>& grads) -> T {
    std::size_t const count = tape.count;
    if (count < 2) {
        return T{0};
    }
    constexpr T eps = static_cast<T>(C.norm_eps);
    T const inv = T{1} / static_cast<T>(count - 1);

    Matrix<T, C.max_seq, C.dim> dstream{};
    for (std::size_t t = 0; t + 1 < count; ++t) {
        auto const dlogits =
            ops::scale(ops::cross_entropy_grad(row(tape.logits, t), input[t + 1]), inv);
        auto const head =
            ops::linear_backward(output_weight(model), row(tape.final_normed, t), dlogits);
        accumulate(output_weight(grads), head.dw);
        auto const dx = norm_backward_into<T, C>(model.final_norm, row(tape.final_stream, t),
                                                 head.dx, grads.final_norm, eps);
        set_row(dstream, t, dx);
    }

    for (std::size_t layer = C.layers; layer-- > 0;) {
        auto const& block = model.blocks[layer];
        auto& block_grads = grads.blocks[layer];
        feed_forward_backward<T, C>(block.ffn, tape.before_ffn[layer], block.ffn_norm, dstream,
                                    block_grads.ffn, block_grads.ffn_norm, count);
        attend_backward<T, C>(block.attn, tape.before_attn[layer], block.attn_norm, dstream,
                              block_grads.attn, block_grads.attn_norm, count);
    }

    for (std::size_t t = 0; t < count; ++t) {
        auto const dx = row(dstream, t);
        add_to_row(grads.embedding, input[t], dx);
        if constexpr (C.pos == Pos::Learned) {
            add_to_row(grads.positions.value, t, dx);
        }
    }
    return sequence_loss(tape, input);
}

} // namespace poyea::shrimp
