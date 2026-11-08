#pragma once

#include "shrimp/core/math.hpp"
#include "shrimp/core/tensor.hpp"
#include "shrimp/model/transformer.hpp"
#include "shrimp/ops/norm_grad.hpp"

#include <array>
#include <type_traits>
#include <cstddef>

namespace poyea::shrimp {

// Mirrors the weight structs but zero initialised, and a distinct type so a
// gradient cannot be passed where a model is expected.

template <typename T, std::size_t N, Norm K>
struct NormGradients {
    Vector<T, N> gain{};
};

template <typename T, std::size_t N>
struct NormGradients<T, N, Norm::Layer> {
    Vector<T, N> gain{};
    Vector<T, N> bias{};
};

template <typename T, ModelConfig C>
struct AttentionGradients {
    Matrix<T, C.dim, C.dim> wq{};
    Matrix<T, C.kv_dim(), C.dim> wk{};
    Matrix<T, C.kv_dim(), C.dim> wv{};
    Matrix<T, C.dim, C.dim> wo{};
};

template <typename T, ModelConfig C, Act A>
struct FeedForwardGradientsFor {
    Matrix<T, C.hidden, C.dim> gate{};
    Matrix<T, C.hidden, C.dim> up{};
    Matrix<T, C.dim, C.hidden> down{};
};

template <typename T, ModelConfig C>
struct FeedForwardGradientsFor<T, C, Act::GELU> {
    Matrix<T, C.hidden, C.dim> up{};
    Matrix<T, C.dim, C.hidden> down{};
};

template <typename T, ModelConfig C>
using FeedForwardGradients = FeedForwardGradientsFor<T, C, C.act>;

template <typename T, ModelConfig C>
struct BlockGradients {
    NormGradients<T, C.dim, C.norm> attn_norm{};
    AttentionGradients<T, C> attn{};
    NormGradients<T, C.dim, C.norm> ffn_norm{};
    FeedForwardGradients<T, C> ffn{};
};

template <typename T, ModelConfig C>
    requires(C.valid())
struct Gradients {
    static constexpr ModelConfig config = C;
    using value_type = T;

    Matrix<T, C.vocab, C.dim> embedding{};
    std::array<BlockGradients<T, C>, C.layers> blocks{};
    NormGradients<T, C.dim, C.norm> final_norm{};
    Matrix<T, C.vocab, C.dim> lm_head{};
};

template <typename T, std::size_t... Dims>
constexpr void accumulate(Tensor<T, Dims...>& into, Tensor<T, Dims...> const& delta) noexcept {
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        into.storage[i] += delta.storage[i];
    }
}

template <typename T, std::size_t... Dims>
constexpr void scale_into(Tensor<T, Dims...>& t, T factor) noexcept {
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        t.storage[i] *= factor;
    }
}

template <typename T, std::size_t... Dims>
constexpr void descend(Tensor<T, Dims...>& weight, Tensor<T, Dims...> const& grad,
                       T rate) noexcept {
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        weight.storage[i] -= rate * grad.storage[i];
    }
}

template <typename T, std::size_t... Dims>
[[nodiscard]] constexpr auto sum_squares(Tensor<T, Dims...> const& t) noexcept -> T {
    T total{};
    for (std::size_t i = 0; i < Tensor<T, Dims...>::size; ++i) {
        total += t.storage[i] * t.storage[i];
    }
    return total;
}

template <typename T, ModelConfig C, typename Fn>
constexpr void for_each_tensor(Gradients<T, C>& g, Fn&& fn) {
    fn(g.embedding);
    for (std::size_t l = 0; l < C.layers; ++l) {
        auto& b = g.blocks[l];
        fn(b.attn_norm.gain);
        if constexpr (C.norm == Norm::Layer) {
            fn(b.attn_norm.bias);
        }
        fn(b.attn.wq);
        fn(b.attn.wk);
        fn(b.attn.wv);
        fn(b.attn.wo);
        fn(b.ffn_norm.gain);
        if constexpr (C.norm == Norm::Layer) {
            fn(b.ffn_norm.bias);
        }
        if constexpr (C.act == Act::SwiGLU) {
            fn(b.ffn.gate);
        }
        fn(b.ffn.up);
        fn(b.ffn.down);
    }
    fn(g.final_norm.gain);
    if constexpr (C.norm == Norm::Layer) {
        fn(g.final_norm.bias);
    }
    fn(g.lm_head);
}

template <typename T, ModelConfig C>
constexpr void zero_gradients(Gradients<T, C>& g) {
    for_each_tensor(g, [](auto& t) {
        for (std::size_t i = 0; i < std::remove_reference_t<decltype(t)>::size; ++i) {
            t.storage[i] = typename std::remove_reference_t<decltype(t)>::value_type{};
        }
    });
}

template <typename T, ModelConfig C>
constexpr void scale_gradients(Gradients<T, C>& g, T factor) {
    for_each_tensor(g, [factor](auto& t) { scale_into(t, factor); });
}

template <typename T, ModelConfig C>
[[nodiscard]] constexpr auto gradient_norm(Gradients<T, C>& g) -> T {
    T total{};
    for_each_tensor(g, [&total](auto const& t) { total += sum_squares(t); });
    return cmath::sqrt(total);
}

template <typename T, ModelConfig C>
constexpr void sgd_step(Transformer<T, C>& model, Gradients<T, C> const& g, T rate) {
    descend(model.embedding, g.embedding, rate);
    for (std::size_t l = 0; l < C.layers; ++l) {
        auto& w = model.blocks[l];
        auto const& d = g.blocks[l];
        descend(w.attn_norm.gain, d.attn_norm.gain, rate);
        if constexpr (C.norm == Norm::Layer) {
            descend(w.attn_norm.bias, d.attn_norm.bias, rate);
        }
        descend(w.attn.wq, d.attn.wq, rate);
        descend(w.attn.wk, d.attn.wk, rate);
        descend(w.attn.wv, d.attn.wv, rate);
        descend(w.attn.wo, d.attn.wo, rate);
        descend(w.ffn_norm.gain, d.ffn_norm.gain, rate);
        if constexpr (C.norm == Norm::Layer) {
            descend(w.ffn_norm.bias, d.ffn_norm.bias, rate);
        }
        if constexpr (C.act == Act::SwiGLU) {
            descend(w.ffn.gate, d.ffn.gate, rate);
        }
        descend(w.ffn.up, d.ffn.up, rate);
        descend(w.ffn.down, d.ffn.down, rate);
    }
    descend(model.final_norm.gain, g.final_norm.gain, rate);
    if constexpr (C.norm == Norm::Layer) {
        descend(model.final_norm.bias, g.final_norm.bias, rate);
    }
    descend(model.lm_head, g.lm_head, rate);
}

template <typename T, ModelConfig C>
[[nodiscard]] constexpr auto norm_backward_into(NormWeights<T, C.dim, C.norm> const& norm,
                                                Vector<T, C.dim> const& x,
                                                Vector<T, C.dim> const& dn,
                                                NormGradients<T, C.dim, C.norm>& dnorm,
                                                T epsilon) -> Vector<T, C.dim> {
    if constexpr (C.norm == Norm::Layer) {
        auto const grad = ops::layernorm_backward(x, norm.gain, epsilon, dn);
        accumulate(dnorm.gain, grad.dgain);
        accumulate(dnorm.bias, grad.dbias);
        return grad.dx;
    } else if constexpr (C.norm == Norm::RMS) {
        auto const grad = ops::rmsnorm_backward(x, norm.gain, epsilon, dn);
        accumulate(dnorm.gain, grad.dgain);
        return grad.dx;
    } else {
        static_assert(detail::dependent_false<T>, "unhandled Norm enumerator");
    }
}

} // namespace poyea::shrimp
