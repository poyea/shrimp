#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/ops/linalg.hpp"
#include "shrimp/ops/nn.hpp"

#include <array>
#include <cstddef>

namespace poyea::shrimp {

enum class Norm { RMS, Layer };
enum class Act { SwiGLU, GELU };

struct ModelConfig {
    std::size_t vocab;
    std::size_t dim;
    std::size_t heads;
    std::size_t kv_heads = 0; // 0 means same as heads
    std::size_t layers;
    std::size_t max_seq;
    std::size_t hidden;
    Norm norm = Norm::RMS;
    Act act = Act::SwiGLU;
    double rope_theta = 10000.0;
    double norm_eps = 1e-5;

    [[nodiscard]] constexpr auto head_dim() const noexcept -> std::size_t { return dim / heads; }

    [[nodiscard]] constexpr auto kv_head_count() const noexcept -> std::size_t {
        return kv_heads == 0 ? heads : kv_heads;
    }

    [[nodiscard]] constexpr auto kv_dim() const noexcept -> std::size_t {
        return kv_head_count() * head_dim();
    }

    [[nodiscard]] constexpr auto group_size() const noexcept -> std::size_t {
        return heads / kv_head_count();
    }

    [[nodiscard]] constexpr auto norm_parameters() const noexcept -> std::size_t {
        return norm == Norm::Layer ? 2 * dim : dim;
    }

    [[nodiscard]] constexpr auto ffn_parameters() const noexcept -> std::size_t {
        return (act == Act::SwiGLU ? 3 : 2) * dim * hidden;
    }

    [[nodiscard]] constexpr auto parameters() const noexcept -> std::size_t {
        std::size_t const attn = (2 * dim * dim) + (2 * kv_dim() * dim);
        std::size_t const per_layer = (2 * norm_parameters()) + attn + ffn_parameters();
        return (vocab * dim) + (layers * per_layer) + norm_parameters() + (vocab * dim);
    }

    [[nodiscard]] constexpr auto valid() const noexcept -> bool {
        return vocab > 0 && dim > 0 && heads > 0 && layers > 0 && max_seq > 0 && hidden > 0 &&
               (dim % heads) == 0 && (head_dim() % 2) == 0 && kv_head_count() <= heads &&
               (heads % kv_head_count()) == 0;
    }
};

template <typename T, std::size_t N, Norm K> struct NormWeights {
    Vector<T, N> gain = Vector<T, N>::filled(T{1});
};

template <typename T, std::size_t N> struct NormWeights<T, N, Norm::Layer> {
    Vector<T, N> gain = Vector<T, N>::filled(T{1});
    Vector<T, N> bias{};
};

template <Norm K, typename T, std::size_t N>
[[nodiscard]] constexpr auto normalize(NormWeights<T, N, K> const& w, Vector<T, N> const& x,
                                       T epsilon) noexcept -> Vector<T, N> {
    if constexpr (K == Norm::Layer) {
        return ops::layernorm(x, w.gain, w.bias, epsilon);
    } else {
        return ops::rmsnorm(x, w.gain, epsilon);
    }
}

namespace detail {

template <typename T, std::size_t Dim, std::size_t HeadDim>
[[nodiscard]] constexpr auto head_slice(Vector<T, Dim> const& v,
                                        std::size_t head) noexcept -> Vector<T, HeadDim> {
    Vector<T, HeadDim> out{};
    std::size_t const base = head * HeadDim;
    for (std::size_t i = 0; i < HeadDim; ++i) {
        out.storage[i] = v.storage[base + i];
    }
    return out;
}

template <typename T, std::size_t Dim, std::size_t HeadDim>
constexpr void head_splice(Vector<T, Dim>& v, std::size_t head,
                           Vector<T, HeadDim> const& slice) noexcept {
    std::size_t const base = head * HeadDim;
    for (std::size_t i = 0; i < HeadDim; ++i) {
        v.storage[base + i] = slice.storage[i];
    }
}

} // namespace detail

template <typename T, ModelConfig C> struct Attention {
    Matrix<T, C.dim, C.dim> wq{};
    Matrix<T, C.kv_dim(), C.dim> wk{};
    Matrix<T, C.kv_dim(), C.dim> wv{};
    Matrix<T, C.dim, C.dim> wo{};
};

template <typename T, ModelConfig C, Act A> struct FeedForwardWeights {
    Matrix<T, C.hidden, C.dim> gate{};
    Matrix<T, C.hidden, C.dim> up{};
    Matrix<T, C.dim, C.hidden> down{};
};

template <typename T, ModelConfig C> struct FeedForwardWeights<T, C, Act::GELU> {
    Matrix<T, C.hidden, C.dim> up{};
    Matrix<T, C.dim, C.hidden> down{};
};

template <typename T, ModelConfig C> using FeedForward = FeedForwardWeights<T, C, C.act>;

template <typename T, ModelConfig C> struct Block {
    NormWeights<T, C.dim, C.norm> attn_norm{};
    Attention<T, C> attn{};
    NormWeights<T, C.dim, C.norm> ffn_norm{};
    FeedForward<T, C> ffn{};
};

template <typename T, ModelConfig C>
    requires(C.valid())
struct Transformer {
    static constexpr ModelConfig config = C;
    using value_type = T;

    Matrix<T, C.vocab, C.dim> embedding{};
    std::array<Block<T, C>, C.layers> blocks{};
    NormWeights<T, C.dim, C.norm> final_norm{};
    Matrix<T, C.vocab, C.dim> lm_head{};
};

// Not a std::vector: constant evaluation allows allocation only transiently.
template <std::size_t Capacity> struct TokenSequence {
    std::array<std::size_t, Capacity> tokens{};
    std::size_t length = 0;

    constexpr void push(std::size_t token) noexcept {
        if (length < Capacity) {
            tokens[length] = token;
            ++length;
        }
    }

    [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept -> std::size_t {
        return tokens[i];
    }

    [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return length; }

    [[nodiscard]] constexpr auto full() const noexcept -> bool { return length >= Capacity; }

    [[nodiscard]] constexpr auto operator==(TokenSequence const& other) const noexcept -> bool {
        if (length != other.length) {
            return false;
        }
        for (std::size_t i = 0; i < length; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

template <std::size_t Capacity, typename... Ids>
[[nodiscard]] constexpr auto tokens(Ids... ids) noexcept -> TokenSequence<Capacity> {
    TokenSequence<Capacity> sequence{};
    (sequence.push(static_cast<std::size_t>(ids)), ...);
    return sequence;
}

template <typename T, ModelConfig C>
constexpr void attend(Attention<T, C> const& w, Matrix<T, C.max_seq, C.dim>& stream,
                      NormWeights<T, C.dim, C.norm> const& norm, std::size_t count) {
    constexpr std::size_t head_dim = C.head_dim();
    constexpr std::size_t kv_dim = C.kv_dim();
    constexpr std::size_t group = C.group_size();
    constexpr T scale = T{1} / cmath::sqrt(static_cast<T>(head_dim));
    auto const& rotation = ops::rope_table<T, head_dim, C.max_seq,
                                          static_cast<T>(C.rope_theta)>;

    Matrix<T, C.max_seq, C.dim> queries{};
    Matrix<T, C.max_seq, kv_dim> keys{};
    Matrix<T, C.max_seq, kv_dim> values{};

    for (std::size_t t = 0; t < count; ++t) {
        auto const normed = normalize(norm, row(stream, t), static_cast<T>(C.norm_eps));
        auto q = ops::linear(w.wq, normed);
        auto k = ops::linear(w.wk, normed);
        auto const v = ops::linear(w.wv, normed);

        for (std::size_t h = 0; h < C.heads; ++h) {
            auto const q_head = detail::head_slice<T, C.dim, head_dim>(q, h);
            detail::head_splice(q, h, ops::rope(q_head, rotation, t));
        }
        for (std::size_t h = 0; h < C.kv_head_count(); ++h) {
            auto const k_head = detail::head_slice<T, kv_dim, head_dim>(k, h);
            detail::head_splice(k, h, ops::rope(k_head, rotation, t));
        }

        set_row(queries, t, q);
        set_row(keys, t, k);
        set_row(values, t, v);
    }

    for (std::size_t t = 0; t < count; ++t) {
        Vector<T, C.dim> context{};
        auto const query = row(queries, t);

        for (std::size_t h = 0; h < C.heads; ++h) {
            std::size_t const kv = h / group;
            auto const q_head = detail::head_slice<T, C.dim, head_dim>(query, h);

            Vector<T, C.max_seq> scores{};
            for (std::size_t j = 0; j <= t; ++j) {
                auto const k_head = detail::head_slice<T, kv_dim, head_dim>(row(keys, j), kv);
                scores.storage[j] = ops::dot(q_head, k_head) * scale;
            }
            auto const weights = ops::softmax_prefix(scores, t + 1);

            Vector<T, head_dim> blended{};
            for (std::size_t j = 0; j <= t; ++j) {
                auto const v_head = detail::head_slice<T, kv_dim, head_dim>(row(values, j), kv);
                for (std::size_t i = 0; i < head_dim; ++i) {
                    blended.storage[i] += weights.storage[j] * v_head.storage[i];
                }
            }
            detail::head_splice(context, h, blended);
        }

        auto const projected = ops::linear(w.wo, context);
        for (std::size_t i = 0; i < C.dim; ++i) {
            stream.storage[(t * C.dim) + i] += projected.storage[i];
        }
    }
}

template <typename T, ModelConfig C>
constexpr void feed_forward(FeedForward<T, C> const& w, Matrix<T, C.max_seq, C.dim>& stream,
                            NormWeights<T, C.dim, C.norm> const& norm, std::size_t count) {
    for (std::size_t t = 0; t < count; ++t) {
        auto const normed = normalize(norm, row(stream, t), static_cast<T>(C.norm_eps));

        Vector<T, C.dim> projected{};
        if constexpr (C.act == Act::SwiGLU) {
            auto const gated = ops::silu(ops::linear(w.gate, normed));
            auto const lifted = ops::linear(w.up, normed);
            projected = ops::linear(w.down, ops::hadamard(gated, lifted));
        } else {
            projected = ops::linear(w.down, ops::gelu(ops::linear(w.up, normed)));
        }

        for (std::size_t i = 0; i < C.dim; ++i) {
            stream.storage[(t * C.dim) + i] += projected.storage[i];
        }
    }
}

template <typename T, ModelConfig C, std::size_t Capacity>
[[nodiscard]] constexpr auto forward(Transformer<T, C> const& model,
                                     TokenSequence<Capacity> const& input) -> Vector<T, C.vocab> {
    std::size_t const count = input.size() < C.max_seq ? input.size() : C.max_seq;

    Matrix<T, C.max_seq, C.dim> stream{};
    for (std::size_t t = 0; t < count; ++t) {
        set_row(stream, t, row(model.embedding, input[t]));
    }

    for (std::size_t layer = 0; layer < C.layers; ++layer) {
        auto const& block = model.blocks[layer];
        attend<T, C>(block.attn, stream, block.attn_norm, count);
        feed_forward<T, C>(block.ffn, stream, block.ffn_norm, count);
    }

    auto const last =
        normalize(model.final_norm, row(stream, count - 1), static_cast<T>(C.norm_eps));
    return ops::linear(model.lm_head, last);
}

template <std::size_t MaxNew, typename T, ModelConfig C, std::size_t Capacity>
[[nodiscard]] constexpr auto
generate(Transformer<T, C> const& model,
         TokenSequence<Capacity> const& prompt) -> TokenSequence<Capacity> {
    TokenSequence<Capacity> sequence = prompt;
    for (std::size_t step = 0; step < MaxNew; ++step) {
        if (sequence.full() || sequence.size() >= C.max_seq) {
            break;
        }
        sequence.push(ops::argmax(forward(model, sequence)));
    }
    return sequence;
}

} // namespace poyea::shrimp
