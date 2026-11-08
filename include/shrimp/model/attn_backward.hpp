#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/model/gradients.hpp"
#include "shrimp/model/transformer.hpp"
#include "shrimp/ops/act_grad.hpp"
#include "shrimp/ops/linalg.hpp"
#include "shrimp/ops/linalg_grad.hpp"
#include "shrimp/ops/nn.hpp"
#include "shrimp/ops/norm_grad.hpp"

#include <cstddef>

namespace poyea::shrimp {

namespace detail {

template <typename T, ModelConfig C> struct AttentionTape {
    Matrix<T, C.max_seq, C.dim> normed{};
    Matrix<T, C.max_seq, C.dim> queries{};
    Matrix<T, C.max_seq, C.kv_dim()> keys{};
    Matrix<T, C.max_seq, C.kv_dim()> values{};
    Matrix<T, C.max_seq, C.dim> context{};
    Matrix<T, C.max_seq, C.heads * C.max_seq> weights{};
};

template <typename T, ModelConfig C>
[[nodiscard]] constexpr auto attention_tape(Attention<T, C> const& w,
                                            Matrix<T, C.max_seq, C.dim> const& stream,
                                            NormWeights<T, C.dim, C.norm> const& norm,
                                            std::size_t count) -> AttentionTape<T, C> {
    constexpr std::size_t head_dim = C.head_dim();
    constexpr std::size_t kv_dim = C.kv_dim();
    constexpr std::size_t group = C.group_size();
    constexpr T scale = T{1} / cmath::sqrt(static_cast<T>(head_dim));

    AttentionTape<T, C> tape{};

    for (std::size_t t = 0; t < count; ++t) {
        auto const normed = normalize(norm, row(stream, t), static_cast<T>(C.norm_eps));
        set_row(tape.normed, t, normed);

        auto q = project<C.bias>(w.wq, w.bq, normed);
        auto k = project<C.bias>(w.wk, w.bk, normed);
        auto const v = project<C.bias>(w.wv, w.bv, normed);
        if constexpr (C.pos == Pos::RoPE) {
            auto const& rotation =
                ops::rope_table<T, head_dim, C.max_seq, static_cast<T>(C.rope_theta)>;
            for (std::size_t h = 0; h < C.heads; ++h) {
                auto const q_head = head_slice<T, C.dim, head_dim>(q, h);
                head_splice(q, h, ops::rope(q_head, rotation, t));
            }
            for (std::size_t h = 0; h < C.kv_head_count(); ++h) {
                auto const k_head = head_slice<T, kv_dim, head_dim>(k, h);
                head_splice(k, h, ops::rope(k_head, rotation, t));
            }
        }

        set_row(tape.queries, t, q);
        set_row(tape.keys, t, k);
        set_row(tape.values, t, v);
    }

    for (std::size_t t = 0; t < count; ++t) {
        auto const query = row(tape.queries, t);
        Vector<T, C.dim> context{};

        for (std::size_t h = 0; h < C.heads; ++h) {
            std::size_t const kv = h / group;
            std::size_t const wbase = (t * C.heads * C.max_seq) + (h * C.max_seq);
            auto const q_head = head_slice<T, C.dim, head_dim>(query, h);

            Vector<T, C.max_seq> scores{};
            for (std::size_t j = 0; j <= t; ++j) {
                auto const k_head = head_slice<T, kv_dim, head_dim>(row(tape.keys, j), kv);
                scores.storage[j] = ops::dot(q_head, k_head) * scale;
            }
            auto const weights = ops::softmax_prefix(scores, t + 1);
            for (std::size_t j = 0; j < C.max_seq; ++j) {
                tape.weights.storage[wbase + j] = weights.storage[j];
            }

            Vector<T, head_dim> blended{};
            for (std::size_t j = 0; j <= t; ++j) {
                auto const v_head = head_slice<T, kv_dim, head_dim>(row(tape.values, j), kv);
                for (std::size_t i = 0; i < head_dim; ++i) {
                    blended.storage[i] += weights.storage[j] * v_head.storage[i];
                }
            }
            head_splice(context, h, blended);
        }
        set_row(tape.context, t, context);
    }
    return tape;
}

} // namespace detail

template <typename T, ModelConfig C>
constexpr void attend_backward(Attention<T, C> const& w,
                               Matrix<T, C.max_seq, C.dim> const& stream_in,
                               NormWeights<T, C.dim, C.norm> const& norm,
                               Matrix<T, C.max_seq, C.dim>& dstream, AttentionGradients<T, C>& dw,
                               NormGradients<T, C.dim, C.norm>& dnorm, std::size_t count) {
    constexpr std::size_t head_dim = C.head_dim();
    constexpr std::size_t kv_dim = C.kv_dim();
    constexpr std::size_t group = C.group_size();
    constexpr T scale = T{1} / cmath::sqrt(static_cast<T>(head_dim));
    constexpr T eps = static_cast<T>(C.norm_eps);

    auto const tape = detail::attention_tape<T, C>(w, stream_in, norm, count);

    Matrix<T, C.max_seq, C.dim> dq{};
    Matrix<T, C.max_seq, kv_dim> dk{};
    Matrix<T, C.max_seq, kv_dim> dv{};

    for (std::size_t t = 0; t < count; ++t) {
        auto const dout = row(dstream, t);
        auto const back = ops::linear_backward(w.wo, row(tape.context, t), dout);
        accumulate(dw.wo, back.dw);
        accumulate_present(dw.bo, dout);

        auto const query = row(tape.queries, t);
        for (std::size_t h = 0; h < C.heads; ++h) {
            std::size_t const kv = h / group;
            std::size_t const wbase = (t * C.heads * C.max_seq) + (h * C.max_seq);
            auto const dblended = detail::head_slice<T, C.dim, head_dim>(back.dx, h);
            auto const q_head = detail::head_slice<T, C.dim, head_dim>(query, h);

            Vector<T, C.max_seq> weights{};
            Vector<T, C.max_seq> dweights{};
            for (std::size_t j = 0; j <= t; ++j) {
                weights.storage[j] = tape.weights.storage[wbase + j];
                auto const v_head =
                    detail::head_slice<T, kv_dim, head_dim>(row(tape.values, j), kv);
                dweights.storage[j] = ops::dot(dblended, v_head);
                T const weight = weights.storage[j];
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dv.storage[(j * kv_dim) + (kv * head_dim) + i] += weight * dblended.storage[i];
                }
            }

            auto const dscores = ops::softmax_prefix_backward(weights, dweights, t + 1);
            for (std::size_t j = 0; j <= t; ++j) {
                auto const k_head = detail::head_slice<T, kv_dim, head_dim>(row(tape.keys, j), kv);
                T const ds = dscores.storage[j] * scale;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dq.storage[(t * C.dim) + (h * head_dim) + i] += ds * k_head.storage[i];
                    dk.storage[(j * kv_dim) + (kv * head_dim) + i] += ds * q_head.storage[i];
                }
            }
        }
    }

    for (std::size_t t = 0; t < count; ++t) {
        auto dq_t = row(dq, t);
        auto dk_t = row(dk, t);
        if constexpr (C.pos == Pos::RoPE) {
            auto const& rotation =
                ops::rope_table<T, head_dim, C.max_seq, static_cast<T>(C.rope_theta)>;
            for (std::size_t h = 0; h < C.heads; ++h) {
                auto const slice = detail::head_slice<T, C.dim, head_dim>(dq_t, h);
                detail::head_splice(dq_t, h, ops::rope_backward(slice, rotation, t));
            }
            for (std::size_t h = 0; h < C.kv_head_count(); ++h) {
                auto const slice = detail::head_slice<T, kv_dim, head_dim>(dk_t, h);
                detail::head_splice(dk_t, h, ops::rope_backward(slice, rotation, t));
            }
        }

        auto const normed = row(tape.normed, t);
        auto const dv_t = row(dv, t);
        auto const gq = ops::linear_backward(w.wq, normed, dq_t);
        auto const gk = ops::linear_backward(w.wk, normed, dk_t);
        auto const gv = ops::linear_backward(w.wv, normed, dv_t);
        accumulate(dw.wq, gq.dw);
        accumulate(dw.wk, gk.dw);
        accumulate(dw.wv, gv.dw);
        // Bias is added before the rotation, so its gradient is the pre-rope one.
        accumulate_present(dw.bq, dq_t);
        accumulate_present(dw.bk, dk_t);
        accumulate_present(dw.bv, dv_t);

        auto dn = gq.dx;
        accumulate(dn, gk.dx);
        accumulate(dn, gv.dx);

        auto const x = row(stream_in, t);
        add_to_row(dstream, t, norm_backward_into<T, C>(norm, x, dn, dnorm, eps));
    }
}

} // namespace poyea::shrimp
