# shrimp

A transformer that runs entirely at compile time.

```cpp
constexpr auto model = poyea::shrimp::induction::build<double>();

static_assert(poyea::shrimp::generate<3>(model, tokens<12>(0, 1, 2, 3, 1)) ==
              tokens<12>(0, 1, 2, 3, 1, 2, 3, 1));
```

The compiler does the matmuls, the softmaxes and the decode loop.
The binary does nothing.

## Model

```cpp
constexpr poyea::shrimp::ModelConfig cfg{
    .vocab = 8, .dim = 16, .heads = 1, .kv_heads = 1,
    .layers = 2, .max_seq = 12, .hidden = 1,
    .norm = Norm::RMS, .act = Act::SwiGLU,
    .pos = Pos::RoPE, .bias = false, .tie_embeddings = false,
    .rope_theta = 1e6, .norm_eps = 1e-5};

poyea::shrimp::Transformer<double, cfg> model{};
```

Pre-norm, split-half RoPE, greedy decode.
`norm` takes RMS or Layer, `act` takes SwiGLU or GELU, `kv_heads` gives GQA.
`pos` takes RoPE or a learned table, `bias` adds a bias to every projection, and `tie_embeddings` makes the embedding the output map.
Shapes are template parameters, so a mismatch fails to compile.
Weights come from `induction::build()` (closed form) or `random_transformer(seed)`.

Ceiling is ~10^5 parameters: constant evaluation is interpreted and step-limited.

## GPT-2

`gpt2::config` is GPT-2's architecture at a shape that fits: learned absolute positions, LayerNorm, GELU, a bias on every projection, output weights tied to the embedding.
`gpt2::small`, `medium`, `large` and `xl` are the four published shapes, as configs rather than models.
Their `parameters()` reproduce the released counts (124,439,808 for small), which the suite asserts.
Instantiating one is a different matter: the weights live in the compiler during constant evaluation, so the real shapes are far past the ceiling.

## Layout

```
include/shrimp/core/math.hpp          exp log sqrt tanh sigmoid pow sin cos
include/shrimp/core/tensor.hpp        Tensor<T, Dims...>
include/shrimp/core/random.hpp        SplitMix64, uniform, normal
include/shrimp/ops/linalg.hpp         dot matmul linear transpose argmax
include/shrimp/ops/nn.hpp             softmax rmsnorm layernorm gelu silu rope
include/shrimp/model/transformer.hpp  forward, generate
include/shrimp/model/induction.hpp    closed-form induction head
include/shrimp/model/gpt2.hpp         GPT-2 architecture and shapes
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
