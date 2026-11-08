# shrimp

A transformer that runs entirely at compile time.

```cpp
constexpr auto model = shrimp::induction::build<double>();

static_assert(shrimp::generate<3>(model, tokens<12>(0, 1, 2, 3, 1)) ==
              tokens<12>(0, 1, 2, 3, 1, 2, 3, 1));
```

The compiler does the matmuls, the softmaxes and the decode loop.
The binary does nothing.

## Model

```cpp
constexpr shrimp::ModelConfig cfg{
    .vocab = 8, .dim = 16, .heads = 1,
    .layers = 2, .max_seq = 12, .hidden = 1,
    .rope_theta = 1e6, .norm_eps = 1e-5};

shrimp::Transformer<double, cfg> model{};
```

Pre-norm RMSNorm, split-half RoPE, SwiGLU, greedy decode.
Shapes are template parameters, so a mismatch fails to compile.
Weights come from `induction::build()` (closed form) or `random_transformer(seed)`.

Ceiling is ~10^5 parameters: constant evaluation is interpreted and step-limited.

## Layout

```
include/shrimp/core/math.hpp          exp log sqrt tanh sigmoid pow sin cos
include/shrimp/core/tensor.hpp        Tensor<T, Dims...>
include/shrimp/core/random.hpp        SplitMix64, uniform, normal
include/shrimp/ops/linalg.hpp         dot matmul linear transpose argmax
include/shrimp/ops/nn.hpp             softmax rmsnorm layernorm gelu silu rope
include/shrimp/model/transformer.hpp  forward, generate
include/shrimp/model/induction.hpp    closed-form induction head
include/shrimp/model/init.hpp         seeded init
tests/                                static_assert suites
```

## Build

Compiling the suites runs them.

```bash
bash scripts/constexpr_check.sh g++-14
```

```bash
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
```

## License

MIT
