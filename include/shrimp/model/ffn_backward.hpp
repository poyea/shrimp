#pragma once

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

template <typename T, ModelConfig C>
constexpr void feed_forward_backward(FeedForward<T, C> const& w,
                                     Matrix<T, C.max_seq, C.dim> const& stream_in,
                                     NormWeights<T, C.dim, C.norm> const& norm,
                                     Matrix<T, C.max_seq, C.dim>& dstream,
                                     FeedForwardGradients<T, C>& dw,
                                     NormGradients<T, C.dim, C.norm>& dnorm, std::size_t count) {
    constexpr T eps = static_cast<T>(C.norm_eps);

    for (std::size_t t = 0; t < count; ++t) {
        auto const x = row(stream_in, t);
        auto const dout = row(dstream, t);
        auto const n = normalize(norm, x, eps);

        Vector<T, C.dim> dn{};
        if constexpr (C.act == Act::SwiGLU) {
            auto const z_gate = ops::linear(w.gate, n);
            auto const gated = ops::silu(z_gate);
            auto const lifted = ops::linear(w.up, n);
            auto const mixed = ops::hadamard(gated, lifted);

            auto const down = ops::linear_backward(w.down, mixed, dout);
            accumulate(dw.down, down.dw);

            auto const dz_gate = ops::silu_backward(z_gate, ops::hadamard(down.dx, lifted));
            auto const dlifted = ops::hadamard(down.dx, gated);

            auto const from_gate = ops::linear_backward(w.gate, n, dz_gate);
            auto const from_up = ops::linear_backward(w.up, n, dlifted);
            accumulate(dw.gate, from_gate.dw);
            accumulate(dw.up, from_up.dw);
            dn = ops::add(from_gate.dx, from_up.dx);
        } else if constexpr (C.act == Act::GELU) {
            auto const z = ops::linear(w.up, n);
            auto const activated = ops::gelu(z);

            auto const down = ops::linear_backward(w.down, activated, dout);
            accumulate(dw.down, down.dw);

            auto const dz = ops::gelu_backward(z, down.dx);
            auto const up = ops::linear_backward(w.up, n, dz);
            accumulate(dw.up, up.dw);
            dn = up.dx;
        } else {
            static_assert(detail::dependent_false<T>, "unhandled Act enumerator");
        }

        add_to_row(dstream, t, norm_backward_into<T, C>(norm, x, dn, dnorm, eps));
    }
}

} // namespace poyea::shrimp
