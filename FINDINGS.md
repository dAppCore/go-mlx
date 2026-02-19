# FINDINGS.md — go-mlx Research & Discovery

Record findings, gaps, and architectural decisions here as work progresses.

---

## 2026-02-19: Split from go-ai (Virgil)

### Origin

This package was extracted from `forge.lthn.ai/core/go-ai/mlx/`. The split was motivated by:

1. **Platform isolation** — mlx is darwin/arm64 only with CGO + CMake build. Keeping it in go-ai forces the entire AI package to deal with platform-specific build complexity.
2. **Dependency chain** — go-i18n Phase 2a needs MLX inference for Gemma3-1B domain classification. A standalone go-mlx module can be imported directly without pulling in all of go-ai (DuckDB, Parquet, gRPC, Ollama, etc.).
3. **Build tag simplicity** — Every file is `//go:build darwin && arm64`. As a standalone module, this is clean. Inside go-ai, it was a special case that required careful handling.

### What Was Extracted

| Directory | Files | LOC | Purpose |
|-----------|-------|-----|---------|
| Root (`mlx/`) | 16 | ~2,500 | Core MLX bindings, ops, training |
| `model/` | 3 | ~800 | Gemma3, Qwen3 model implementations |
| `tokenizer/` | 1 | ~324 | BPE tokenizer |
| `sample/` | 1 | ~150 | Sampling strategies |
| `cache/` | 1 | ~201 | KV cache for inference |
| **Total** | **22** | **~4,354** | |

### Import Path Changes

All internal imports rewritten:
- `forge.lthn.ai/core/go-ai/mlx` → `forge.lthn.ai/core/go-mlx`
- `forge.lthn.ai/core/go-ai/mlx/cache` → `forge.lthn.ai/core/go-mlx/cache`
- `forge.lthn.ai/core/go-ai/mlx/tokenizer` → `forge.lthn.ai/core/go-mlx/tokenizer`
- `forge.lthn.ai/core/go-ai/mlx/model` → `forge.lthn.ai/core/go-mlx/model`
- `forge.lthn.ai/core/go-ai/mlx/sample` → `forge.lthn.ai/core/go-mlx/sample`

### Upstream Consumer

`go-ai/ml/backend_mlx.go` is the only file outside mlx/ that imports it. After split, go-ai needs either:
- A `replace` directive: `replace forge.lthn.ai/core/go-mlx => ../go-mlx`
- Or a published module version

### What Stayed in go-ai

- `ml/backend_mlx.go` (253 LOC) — the Backend adapter that calls go-mlx. This stays in go-ai because it implements the go-ai-specific `Backend` interface.
- `test-mlx.go` — integration test utility (go-ai root). Needs updating to import from go-mlx.
- `TEST-RESULTS.md` — comprehensive test report (stays as historical record).

---

## 2026-02-19: Test Coverage Assessment

### Tested (3 test files)

| File | Tests | Coverage |
|------|-------|---------|
| `grad_test.go` | VJP/gradient computation | Good — tests forward+backward pass |
| `lora_test.go` | LoRA adapter | Good — tests apply/merge/save |
| `optim_test.go` | AdamW optimiser | Good — tests step/state |

### Not Tested (critical gaps)

| File | LOC | Risk | Notes |
|------|-----|------|-------|
| `ops.go` | 353 | **High** | MatMul, Softmax, element-wise ops — core of everything |
| `array.go` | 261 | **High** | Array creation, reshape, data access — foundational |
| `nn.go` | ~150 | Medium | Linear, Embedding, RMSNorm layers |
| `fast.go` | ~100 | Medium | Fused Metal kernels (RoPE, ScaledDotProduct) |
| `model/*.go` | ~800 | **High** | No tests for Gemma3/Qwen3 forward pass |
| `tokenizer/` | 324 | **High** | No BPE encode/decode tests |
| `sample/` | ~150 | Medium | No sampling tests |
| `cache/` | 201 | Medium | No KV cache tests |
| `io.go` | ~100 | Medium | No safetensors load tests |

### Error Handling

The error handler in `mlx.go` stores the last error in a C static variable and logs it via `slog.Error`. This is **not propagated to Go callers**. Functions like `MatMul`, `Softmax`, etc. return `*Array` with no error — if the C operation fails, the caller gets a nil/invalid array with no indication why.

### Memory Management

Arrays use `runtime.SetFinalizer` for C-side deallocation. Under sustained inference (1000+ tokens), this relies on GC pressure to trigger finalizers. No explicit `Close()` or `Free()` method exists on Array — could leak under high throughput if GC doesn't keep up.

---

## 2026-02-19: Dependency Chain

```
go-i18n (Phase 2a: domain classification)
    └── needs Gemma3-1B inference
        └── go-mlx (this package)
            └── mlx-c v0.4.1 (CMake, fetched from GitHub)
                └── Apple MLX (Metal GPU compute)

go-ai/ml/backend_mlx.go
    └── imports go-mlx
        └── implements go-ai Backend interface
```

### LEM Lab Connection

LEM Lab (the native MLX chat UI at `localhost:8090`) also uses this code path. Currently working with Qwen3-8B streaming. The model/ directory supports both Gemma3 and Qwen3.

---

## 2026-02-19: Hardware Test Results (from go-ai TEST-RESULTS.md)

Tested on Mac Studio M3 Ultra (32-core CPU, 60-core GPU, 96GB unified memory):
- All 84 go-ai tests pass (including 3 mlx tests)
- MLX grad, lora, optim tests all pass
- Go 1.25.7, mlx-c v0.4.1

### Model Inventory (safetensors)

Available on `/Volumes/Data/lem/safetensors/`:
- Gemma3-1B, Gemma3-4B, Gemma3-27B
- Qwen3-8B (used by LEM Lab)

---

## 2026-02-19: Go 1.26 Impact Assessment

Source: https://go.dev/doc/go1.26

### High Impact (free performance, no code changes)

**CGO call overhead reduced ~30%**
Every MLX operation (MatMul, Add, Softmax, RoPE, etc.) crosses the CGO boundary. The runtime previously used a dedicated syscall P state for cgo calls; Go 1.26 removes that and checks goroutine status instead. This is a direct, automatic performance win for the entire package.

**Green Tea GC now default (10-40% less GC overhead)**
Critical for go-mlx because `Array` objects use `runtime.SetFinalizer` for C-side deallocation via `mlx_*_free()`. Reduced GC overhead means:
- More timely finaliser execution during sustained inference
- Less memory pressure from stale Array objects waiting for GC
- The FINDINGS.md concern about "GC not keeping up under high throughput" is partially mitigated
- Opt-out: `GOEXPERIMENT=nogreenteagc` (temporary, removed in 1.27)

### Medium Impact

**Slice stack allocation in more situations**
The compiler can now allocate slice backing stores on the stack more often. Benefits small temporary slices in `Collect()`, shape manipulation, and internal ops helpers. Debug: `-compile=variablemakehash` flag.

**`testing.B.Loop` inlining fix**
When we add benchmarks (Phase 1), `b.Loop()` style now properly inlines loop bodies. Important for micro-benchmarks of small ops like Add, Multiply.

**Heap base address randomisation (64-bit)**
Security improvement for CGO programs. Randomises heap base at startup. Disable: `GOEXPERIMENT=norandomizedheapbase64`.

### Clarification on Range-over-func

Virgil's Phase 6 TODO mentions "if 1.26 stabilises range-over-func". **Range-over-func has been stable since Go 1.23** and the `iter` package was added in 1.23. Since go.mod is already at Go 1.25.5, `Array.Iter() iter.Seq[float32]` can be implemented today without a version bump. Go 1.26 adds no new iterator features beyond what 1.23-1.25 provide.

### Recommendation

No Go version bump needed for the performance wins — they're automatic at runtime. The only code-level Go 1.26 feature that matters is `testing.ArtifactDir()` for benchmark result storage, which is minor. Focus remains on Phase 1 hardening.

---

## 2026-02-19: go-ai Split Context

Virgil is splitting go-ai into sub-packages, with go-ai becoming a meta/catch-all for ML features. go-mlx was the first extraction. This means:
- More packages will follow the go-mlx pattern (standalone module, own build, own tests)
- go-ai will eventually be a thin layer importing sub-packages
- The `replace` directive approach works for development; published modules for releases

---

## 2026-02-19: Floats()/DataInt32() Unsafe on Non-Contiguous Arrays

**Discovery**: `Array.Floats()` and `Array.DataInt32()` read `Size()` elements from the raw C data pointer (`mlx_array_data_float32`). For non-contiguous arrays (transpose, broadcast, slice views), the physical memory layout doesn't match the logical layout. Reading `Size()` contiguous elements returns incorrect data or reads past the physical buffer.

**Affected operations**: `Transpose()`, `BroadcastTo()`, `SliceAxis()`, `Slice()`, `AsStrided()` — any operation that creates a view rather than a copy.

**Workaround**: `Reshape(arr, totalSize)` forces a contiguous copy before reading flat data. All tests use this pattern for view operations.

**Fix needed (Phase 4)**: Either:
1. Add a `Contiguous()` method that wraps `mlx_contiguous` (if available in mlx-c)
2. Or have `Floats()`/`DataInt32()` automatically force contiguity before reading
3. Document the behaviour clearly if views are intentionally lazy

This is a data correctness issue — silent wrong results, not a crash.

---

## 2026-02-19: Backend Abstraction — API Breaking Change

**Design doc:** `docs/plans/2026-02-19-backend-abstraction-design.md`

### What's changing

The entire public API is being replaced. All CGO code moves to `internal/metal/`. The root package becomes a clean interface layer:

```go
m, _ := mlx.LoadModel("/path/to/model/")
defer m.Close()
for tok := range m.Generate("prompt", mlx.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
```

The old API (`Array`, `MatMul`, `model.LoadModel`, `model.Model`, etc.) will no longer be public.

### Impact on go-ai

`backend_mlx.go` currently imports root-level Array, ops, model types. These move to `internal/metal/` and become inaccessible. Migration: replace direct tensor manipulation with `mlx.LoadModel()` + `mlx.TextModel.Generate()`.

### Impact on go-i18n

The API for Gemma3-1B domain classification will be:
```go
m, _ := mlx.LoadModel("/path/to/gemma-3-1b/")
for tok := range m.Generate(sentence, mlx.WithMaxTokens(32)) { ... }
```
Streaming via `iter.Seq[Token]`. No tokenisation or sampling to handle.

### Memory leak fix included

The refactor includes deterministic memory management — `TextModel.Close()` for model weights and per-step intermediate cleanup during generation. This addresses the current production blocker.
