---
title: Model ↔ Runtime SDK
description: The public boundary a model package (pkg/metal/model/{family}) uses, so models are pure-Go and metal owns all cgo/Metal/runtime.
---

# Model ↔ Runtime SDK

A model family lives in its own package under `pkg/metal/model/{family}` (e.g.
`pkg/metal/model/gemma4`). The package is **pure Go**: it imports `metal` and
depends only on the public SDK described here. It contains no cgo, names no
private metal symbol, and touches no metal struct field directly.

`metal` owns everything below the SDK line — the cgo bindings, Metal compute
shaders, the lazy-eval graph, the KV-cache implementations, sampling, and
quantisation. A model package describes *what* its architecture computes; `metal`
provides the primitives and kernels that compute it.

The boundary exists because cgo C types are package-private: a model package
cannot construct or pass a `metal.C.mlx_array`, so any code that crosses the
Go↔C line for MLX must live in `metal`. The SDK is the set of Go-typed surfaces
that let a model package stay on the Go side of that line.

## Boundary

```
pkg/metal/model/gemma4   (package gemma4, pure Go)
    |  implements metal.InternalModel
    |  uses: primitive surface · cache accessors · native-kernel requests
    v
pkg/metal                (package metal — cgo, Metal, runtime)
```

The model→runtime entry point is the existing `metal.InternalModel` interface
(`Forward`, `ForwardMasked`, `NewCache`, `NumLayers`, `Tokenizer`, `ModelType`,
`ApplyLoRA`, plus the optional capability interfaces). `metal`'s generate/decode
loop drives a model through it. A model package self-registers its loader from
`init()` via `metal.RegisterModelLoader(arch, fn)`; a blank import of the model
package (from `cmd/mlx`) triggers registration. `metal` never names a concrete
model type.

The SDK adds three categories on top of that entry point.

## Category 1 — Primitive surface

The tensor and model-building operations a model's `Forward` legitimately needs,
exposed as curated public API: tensor ops (`Matmul`, `Add`, `SDPA`, `RMSNorm`,
…), sampling, quantised mat-vec, activation helpers (`Gelu*`), weight loading and
resolution (`LoadModelWeights`, `ResolveModelRoot`), and cache length/capacity
reads (`CacheLen`, `CacheCapacity`).

The surface is **curated, not a dump**. The rule:

- **Exported** — genuine model-author primitives: an operation a model performs,
  a value it reads, a loader it calls.
- **Internal** — runtime plumbing that has no place in a model: C-handle
  marshalling (`cArray`), the cgo error sink (`lastError`), scratch pools
  (`suppressIDsScratch`), trace-event buffers. These never cross the boundary;
  where a model appears to need them, it is reaching into the runtime and the
  need is met by Category 2 or 3 instead.

## Category 2 — Cache accessors

KV-cache implementations (`KVCache`, `RotatingKVCache`, `FixedKVCache`,
`PagedKVCache`, `QuantizedKVCache`) expose their state through methods rather
than fields, so a model package never touches cache internals:

```go
// read surface (illustrative)
func (c *KVCache) Keys() *Array
func (c *KVCache) Values() *Array
func (c *KVCache) Offset() int
func (c *KVCache) Step() int
func (c *KVCache) MaxSize() int
// fixed/paged/quantised add PageSize(), Bits(), capacity reads
```

Construction that a model needs (wrapping existing key/value tensors into a
cache for a custom layout) is offered through exported constructors, not struct
literals. The model reads and builds caches only through this surface.

## Category 3 — Native-kernel requests

Fused Metal decode kernels are cgo and model-shape-specific (a gemma4 fused
layer differs from a qwen3 one), so the kernels **stay in `metal`**, beside the
C types and `decode_bridge.h` they use. `metal` exposes each kernel through a
**request struct** whose fields are `*metal.Array` and scalars. The model fills
a request from its own types and calls the kernel:

```go
// metal side
type Gemma4DecodeLayerRequest struct {
    Hidden, Residual, KeyCache, ValueCache, Offset, FixedMask *Array
    QProjWeight, QProjScales, QProjBiases *Array
    // … the projection / norm / router arrays the kernel reads …
    NumAttentionHeads, NumKVHeads, HeadDim, RopeDims int32
    RopeBase, RMSNormEps                             float32
}

func NativeGemma4DecodeLayer(req Gemma4DecodeLayerRequest) (out, newKeys, newValues *Array, ok bool, err error)
```

```go
// model side (pure Go) — fills the request from its own structs
out, nk, nv, ok, err := metal.NativeGemma4DecodeLayer(metal.Gemma4DecodeLayerRequest{
    Hidden: h, Residual: residual, KeyCache: kc, ValueCache: vc, /* … */
    QProjWeight: attn.QProj.Weight, /* … */
    NumAttentionHeads: cfg.NumAttentionHeads, /* … */
})
```

The model passes **data**, never model types into `metal`, and never opens a cgo
context of its own. `metal` builds the C struct from the request internally and
keeps the C-type boundary on its side.

Each model's fused kernels follow this convention; the *pattern* is the SDK, the
specific request structs are per-model and live in `metal`.

## Layering

Categories 1 and 2 are the **baseline**: they are sufficient to compile and run a
model's generic `Forward` path against `metal`'s portable operations. Category 3
restores the gated fused-kernel fast path. The fused path is an optional
acceleration — a model is correct and complete on Categories 1+2 alone, and opts
into Category 3 where a fused kernel exists and its runtime gate is enabled.

Categories 1 and 2 are reusable as-is across model families. Category 3 is a
repeated pattern: a new family adds its own request structs and kernels in
`metal` and calls them the same way.
