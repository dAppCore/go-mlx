---
title: Architecture
description: CGO binding layer, lazy evaluation, memory model, and internal structure of go-mlx.
---

# Architecture

go-mlx is a Go package that wraps Apple's MLX framework via the mlx-c C API. It runs LLM inference and LoRA fine-tuning on Apple Silicon GPUs (M1-M4) using Metal compute shaders.

## Layer Diagram

```
Go Application
    |
    v
inference.TextModel / inference.TrainableModel   <-- go-inference interfaces
mlx.LoadModel / mlx.NewSession                   <-- direct root APIs
    |
    v
register_metal.go (metalAdapter)                  <-- Backend registration + type conversion
    |
    v
internal/metal/                                   <-- All CGO code
    |
    +-- generate.go    Model, Generate, Chat, batch inference
    +-- metal_kernel.go Custom Metal kernel dispatch for frame compute
    +-- gemma3.go      Gemma 3 decoder
    +-- qwen3.go       Qwen 2/3 and Llama 3 decoder
    +-- tokenizer.go   BPE tokeniser (SentencePiece + GPT-2)
    +-- cache.go       KVCache + RotatingKVCache
    +-- sample.go      Sampling chain: greedy, temperature, TopK, TopP, MinP
    +-- nn.go          Linear, Embedding, RMSNormModule
    +-- ops.go         Element-wise, reduction, matrix, shape operations
    +-- fast.go        Fused Metal kernels: RMSNorm, RoPE, SDPA
    +-- grad.go        VJP, JVP, ValueAndGrad, Checkpoint, loss functions
    +-- lora.go        LoRA adapters, random normal init, safetensors save
    +-- optim.go       AdamW optimiser
    +-- array.go       Array type, creation, data access
    +-- io.go          Safetensors load/save iterators
    +-- metal.go       Init, error handler, Eval, Materialize
    |
    v
mlx-c v0.4.1                                     <-- C API (fetched by CMake)
    |
    v
Apple MLX / Metal / Accelerate                    <-- GPU compute
```

## CGO Binding

### Build Chain

mlx-c is fetched and built by CMake via `go generate ./...`. The `CMakeLists.txt` at the module root pulls mlx-c v0.4.1 from GitHub:

```cmake
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG "v0.4.1"
)
```

After the CMake build, headers land in `dist/include/` and shared libraries in `dist/lib/`. The `#cgo` directives in `internal/metal/metal.go` reference these paths:

```
CPPFLAGS: -I${SRCDIR}/../../dist/include
LDFLAGS:  -L${SRCDIR}/../../dist/lib -lmlxc -lmlx
darwin:   -framework Foundation -framework Metal -framework Accelerate
          -Wl,-rpath,${SRCDIR}/../../dist/lib
```

Every Go source file in `internal/metal/` carries `//go:build darwin && arm64`. The root package compiles on all platforms; the blank import `_ "dappco.re/go/mlx"` only triggers Metal backend registration on supported hardware.

### Error Handling

mlx-c reports errors through a registered C callback. The handler stores the error string atomically using C11 `atomic_store_explicit` with release ordering. `lastError()` reads and clears it with acquire ordering, returning a Go error.

- `Eval(...*Array) error` -- synchronous GPU evaluation, returns errors
- `EvalAsync(...*Array) error` -- queues arrays for asynchronous evaluation
- `Materialize(...*Array)` -- synchronous evaluation, logs errors (does not return them)

Callers on the hot path (generation loop, training) use `Eval()` for error propagation. `Materialize()` is used in weight loading and test helpers where errors are non-recoverable.

## Lazy Evaluation

MLX uses lazy evaluation: operations build a computation graph without executing. The graph is dispatched to the Metal GPU only when `Eval()` or `EvalAsync()` is called. This enables MLX to optimise and fuse operations across the graph before execution.

After evaluation, `Detach()` breaks an array's graph connections, allowing Metal memory from parent operations to be freed. The generation loop calls `Detach()` on logits and KV cache arrays after each step to prevent memory accumulation from the retained computation graph.

## Array Type

`Array` wraps an `mlx_array` C handle. Arrays are reference-counted on the C side; Go uses `runtime.SetFinalizer` to call `mlx_array_free` when the Go object is collected.

### Creation

```go
// Scalar
a := metal.FromValue(3.14)

// From Go slice with shape
b := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2) // [2, 2]

// Zeros
c := metal.Zeros([]int32{4, 4}, metal.DTypeFloat32)
```

### Data Access

Data extraction methods (`Floats()`, `DataInt32()`, `Ints()`, `Iter()`) automatically handle non-contiguous arrays. Views created by `Transpose`, `BroadcastTo`, or `SliceAxis` are made row-contiguous via `ensureContiguous()` before reading.

```go
data := array.Floats()    // []float32
ints := array.DataInt32() // []int32
val  := array.Float()     // scalar float64
```

### Explicit Cleanup

While GC finalisers handle cleanup, `Free()` releases C handles immediately without waiting for collection:

```go
mlx.Free(a, b, c) // release immediately
```

This is used throughout the generation loop and training code to keep GPU memory bounded.

## Memory Model

The Metal allocator (separate from system RAM) is controlled via functions exposed at the root package level. See [index.md](index.md) for the full function table.

Key points:

- `Model.Close()` deterministically frees all weight arrays without relying on GC. Tied output weights (shared with the embedding table) are detected and skipped to prevent double-free.
- Each `Generate()` call allocates fresh KV caches that are released to GC when the iterator completes.
- Call `ClearCache()` between multi-turn chat turns for prompt memory reclaim rather than waiting for GC.

## Fused Metal Kernels

`internal/metal/fast.go` wraps four mlx-c fused kernels that bypass the general computation graph and dispatch directly to optimised Metal compute shaders:

| Kernel | Go Function | Usage |
|--------|-------------|-------|
| `mlx_fast_rms_norm` | `RMSNorm(x, weight, eps)` | Pre-/post-attention normalisation |
| `mlx_fast_layer_norm` | `LayerNorm(x, weight, bias, eps)` | Standard layer normalisation |
| `mlx_fast_rope` | `RoPE(x, dims, traditional, base, scale, offset)` | Rotary position embeddings |
| `mlx_fast_scaled_dot_product_attention` | `ScaledDotProductAttention(...)` | Causal or explicit-mask attention |

## Attention Mechanism

### Virtual Transpose

Linear projections produce `[B, L, H*D]`. The reshape to `[B, H, L, D]` for attention is implemented via `AsStrided` -- a zero-copy stride manipulation:

```
shape:   [B,    H,   L,   D]
strides: [L*H*D, D, H*D, 1]
```

This avoids a physical copy while reordering dimensions for the SDPA call.

### RoPE

Applied via the fused `mlx_fast_rope` Metal kernel. The `offset` parameter is the current KV cache offset, enabling correct position encoding for continuation from cached positions. Gemma 3 uses per-layer theta (10000 for sliding, 1000000 for global); Qwen and Llama use a single theta.

### SDPA

Two variants:

- **Causal** -- `ScaledDotProductAttention(q, k, v, scale, true)` for autoregressive generation
- **Masked** -- `ScaledDotProductAttentionWithMask(q, k, v, mask, scale)` with explicit additive mask (0 = attend, -inf = ignore) for batched inference with padding

Scale is `1/sqrt(head_dim)`, precomputed at model load time.

## KV Cache

The `Cache` interface manages key-value state for transformer attention layers:

```go
type Cache interface {
    Update(k, v *Array, seqLen int) (*Array, *Array)
    Offset() int
    Len() int
    State() []*Array
    Reset()
    Detach()
}
```

### KVCache (Unbounded)

Pre-allocates in 256-token chunks, growing as needed. Each decode step writes new K/V via `SliceUpdateInplace` and returns a slice view of the valid region. This amortises allocation cost.

### RotatingKVCache (Sliding Window)

Bounded to `maxSize` tokens with two update paths:

- **Prefill** (`seqLen > 1`): concatenate, then trim leading tokens that fall outside the window
- **Decode** (`seqLen == 1`): write in-place at circular index `idx % maxSize`

Used for Gemma 3 sliding-window attention layers. When `ContextLen` is set via `inference.WithContextLen()`, all unbounded caches are replaced with rotating caches.

## Sampling Chain

`newSampler(temp, topP, minP, topK)` builds a composable pipeline:

```
TopP -> MinP -> TopK -> Temperature -> RandomCategorical
```

If `temp == 0`, the chain collapses to greedy (argmax).

- **Greedy** -- `Argmax(logits, -1)`
- **Temperature** -- multiply logits by `1/temp`
- **TopK** -- mask all but the K highest logits with `-inf`
- **TopP (nucleus)** -- keep the smallest set with cumulative probability exceeding `p`
- **MinP** -- mask tokens below `min_p * max_probability`

Full sampling chain (TopP + MinP + TopK) adds approximately 560 us over greedy per token.

## Public APIs

go-mlx exposes two public surfaces:

- the `go-inference` backend registered by `register_metal.go`
- the direct root-package APIs in `api_*.go`, `training*.go`, and `compute*.go`

`register_metal.go` auto-registers `metalBackend` via `init()` on darwin/arm64. The `metalAdapter` converts between `inference.*` types and `metal.*` types, implementing: `Generate`, `Chat`, `Classify`, `BatchGenerate`, `Metrics`, `Info`, `InspectAttention`, `Close`, and the `TrainableModel` interface (`ApplyLoRA`, `Encode`, `Decode`, `NumLayers`).

Consumer pattern:

```go
import (
    "dappco.re/go/core/inference"
    _ "dappco.re/go/mlx"
)

m, err := inference.LoadModel("/path/to/model/")
for tok := range m.Generate(ctx, "prompt", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
```

The root package also exposes direct inference and training APIs:

```go
import mlx "dappco.re/go/mlx"

model, err := mlx.LoadModel("/path/to/model", mlx.WithContextLength(8192))
session, err := mlx.NewSession()
```

### Load Options

Options from `inference.LoadConfig` understood by the Metal backend:

- `ContextLen` -- replaces unbounded `KVCache` with `RotatingKVCache(contextLen)` for all layers
- `AdapterPath` -- loads a trained LoRA adapter from disk at model load time
- `GPULayers` -- logged as a warning if set to 0 (Metal always uses full GPU offload)

## mlxlm Subprocess Backend

`mlxlm/` provides a second backend (`"mlx_lm"`) that spawns a Python 3 process running an embedded `bridge.py` script. Communication is over JSON Lines (stdin/stdout). This backend requires no CGO but depends on Python 3 and the `mlx-lm` package.

Use it when CGO is not available or when you need model architectures not yet implemented natively:

```go
import _ "dappco.re/go/mlx/mlxlm"

m, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
```

Limitations:

- `Classify` and `BatchGenerate` are not supported
- No inference metrics
- Build tag `nomlxlm` removes the backend entirely
