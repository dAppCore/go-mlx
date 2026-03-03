# Metal Memory Leak in Forward Pass — Intermediate Tensor Accumulation

**Severity:** Critical — OOM on all models during extended generation (1B slower, 4B instant)
**Repo:** go-mlx (`forge.lthn.ai/core/go-mlx`)
**Observed:** 23 Feb 2026 — Gemma3 4B distill hits 147GB, Gemma3 1B also leaks

## Problem

During token generation, the forward pass creates ~20+ intermediate `*Array` objects per layer per decode step. None are explicitly freed. Go's garbage collector uses `runtime.SetFinalizer` to call `mlx_array_free()`, but **Go doesn't know about Metal memory** — it only sees the tiny Go struct (~48 bytes). Go GC rarely triggers because Go heap pressure is low, so Metal buffers accumulate unboundedly.

**Math:** 34 layers × ~20 intermediates × 3072 max_tokens = ~2 million unreleased Metal buffers per generation. Each holds a C `mlx_array` handle that pins a Metal buffer.

## Root Cause

### 1. Array lifecycle via Go finalizers (array.go)

```go
func newArray(name string, inputs ...*Array) *Array {
    t := &Array{name: name}
    runtime.SetFinalizer(t, finalizeArray)  // GC-dependent!
    return t
}

func finalizeArray(t *Array) {
    if t != nil && t.ctx.ctx != nil {
        C.mlx_array_free(t.ctx)   // Only called when Go GC runs
        t.ctx.ctx = nil
    }
}
```

Go GC doesn't track Metal memory. With millions of tiny Go structs (~48 bytes each) accumulating, Go sees ~100MB heap — no pressure — no GC trigger. But Metal has 50GB+ pinned.

### 2. Forward pass creates intermediates that become unreachable but not freed (gemma3.go)

```go
func (m *GemmaModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
    h := m.EmbedTokens.Forward(tokens)
    h = MulScalar(h, ...)           // ← old h unreachable, not freed
    for i, layer := range m.Layers {
        h = layer.forward(h, ...)   // ← old h unreachable, not freed (×34 layers)
    }
    return m.Output.Forward(RMSNorm(h, ...))  // ← h, normed unreachable, not freed
}

func (l *DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *TextConfig) *Array {
    normed := RMSNorm(x, ...)               // intermediate
    attnOut := l.Attention.forward(normed, ...)  // creates ~10 intermediates inside
    attnOut = RMSNorm(attnOut, ...)          // ← old attnOut unreachable
    h := Add(x, attnOut)                     // ← attnOut unreachable
    normed = RMSNorm(h, ...)                 // ← old normed unreachable
    mlpOut := l.MLP.forward(normed)          // intermediates inside
    mlpOut = RMSNorm(mlpOut, ...)            // ← old mlpOut unreachable
    return Add(h, mlpOut)                    // ← h, mlpOut unreachable
}

func (a *Attention) forward(x *Array, c Cache, B, L int32, isSliding bool, mask *Array, cfg *TextConfig) *Array {
    q := a.QProj.Forward(x)
    k := a.KProj.Forward(x)
    v := a.VProj.Forward(x)
    q = AsStrided(q, ...)                    // ← view — CANNOT free old q (shared buffer)
    k = AsStrided(k, ...)                    // ← view — CANNOT free old k
    v = AsStrided(v, ...)                    // ← view — CANNOT free old v
    q = RMSNorm(q, ...)                      // ← old q (strided view) unreachable
    k = RMSNorm(k, ...)                      // ← old k unreachable
    q = RoPE(q, ...)                         // ← old q unreachable
    k = RoPE(k, ...)                         // ← old k unreachable
    k, v = c.Update(k, v, ...)               // ← old k, v unreachable
    k = RepeatKV(k, ...)                     // ← old k unreachable (if GQA)
    v = RepeatKV(v, ...)                     // ← old v unreachable
    out = ScaledDotProductAttention(q, k, v, ...)  // ← q, k, v unreachable
    out = Reshape(Transpose(out, ...), ...)  // ← old out, transpose unreachable
    return a.OProj.Forward(out)              // ← out unreachable
}
```

Qwen3's forward pass has the identical pattern.

### 3. Generate loop correctly frees its own intermediates (generate.go)

```go
for i := 0; i < cfg.MaxTokens; i++ {
    l1 := SliceAxis(logits, ...)
    lastPos := Reshape(l1, ...)
    Free(l1)                        // ✓ freed
    // ...
    Free(lastPos)                   // ✓ freed
    Free(next)                      // ✓ freed

    oldLogits := logits
    logits = m.model.Forward(nextInput, caches)  // ← Forward() leaks ~20*34 intermediates
    Free(nextInput, oldLogits)      // ✓ freed

    Eval(logits)                    // ✓ materialises result, but intermediates still held by Go
}
```

The generate loop is disciplined about freeing. The forward pass is not.

## Constraints

1. **`AsStrided` creates views** — freeing the source array invalidates the view. Must not free before the view is consumed.
2. **Compile ops** (`getCompiledGELU()`) — compiled functions may have different ownership rules.
3. **`c.Update(k, v, ...)` takes ownership** — the cache stores k/v. Must not free the arrays the cache now owns.
4. **Model weights are persistent** — don't free `QProj.Weight`, `EmbedTokens.Weight`, etc. Only free *results* of operations.
5. **Thread safety** — Generate may be called from multiple goroutines (unlikely in distill, but API supports it).

## What We Need

Add explicit `Free()` calls for intermediate tensors in:
1. `GemmaModel.ForwardMasked()`
2. `DecoderLayer.forward()`
3. `Attention.forward()`
4. `MLP.forward()`
5. `Qwen3Model.ForwardMasked()` (same pattern)
6. `Qwen3DecoderLayer.forward()`
7. `Qwen3Attention.forward()`
8. `Qwen3MLP.forward()`

The tricky parts:
- **AsStrided views**: After `q = AsStrided(q_proj_result, ...)`, the original `q_proj_result` may share memory. Is it safe to free the pre-strided array after AsStrided? Or does AsStrided reference the same C buffer?
- **Eval timing**: Should we `Eval()` intermediates at layer boundaries to force materialization, or does Free + lazy eval handle it?
- **RMSNorm/RoPE pattern**: `q = RMSNorm(q, ...)` — the old q was a strided view. Does RMSNorm create a new buffer? If so, the strided view (and its source) can be freed after.

## Alternative Approaches Considered

1. **`runtime.GC()` every N steps** — Hack. Costs CPU, non-deterministic, doesn't guarantee Metal buffers are released promptly.
2. **Custom allocator with arena** — Heavy refactor, MLX-C doesn't support it.
3. **`runtime.SetMemoryLimit()` including Metal** — Not possible, Go doesn't track foreign memory.
4. **Explicit Free in forward pass** — The right fix. Matches how generate.go already works.

## Files

- `internal/metal/gemma3.go` — Gemma3 forward pass (lines 323-400)
- `internal/metal/qwen3.go` — Qwen3 forward pass (lines 270-332)
- `internal/metal/generate.go` — Generate loop (lines 131-252, correct)
- `internal/metal/array.go` — Array type, Free(), finalizer (lines 25-296)
- `internal/metal/cache.go` — KVCache, RotatingKVCache
- `internal/metal/ops.go` — AsStrided, RMSNorm, RoPE, etc.

## Reproduction

```bash
cd /Users/snider/Code/LEM
go run . gen distill --model gemma3/4b --probes core --runs 1
# Observe: memory grows ~1GB/sec during generation, hits 50GB+ within 60s
# Also affects 1B, just slower growth rate
```
