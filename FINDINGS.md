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

## 2026-02-19: Backend Abstraction — COMPLETED

**Design doc:** `docs/plans/2026-02-19-backend-abstraction-design.md`
**Implementation plan:** `docs/plans/2026-02-19-backend-abstraction-plan.md`

### What changed

The entire public API has been replaced. All CGO code is now in `internal/metal/`. The root package is a clean interface layer:

```go
m, _ := mlx.LoadModel("/path/to/model/")
defer m.Close()
ctx := context.Background()
for tok := range m.Generate(ctx, "prompt", mlx.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
if err := m.Err(); err != nil { log.Fatal(err) }
```

The old API (`Array`, `MatMul`, `model.LoadModel`, `model.Model`, sub-packages `model/`, `tokenizer/`, `sample/`, `cache/`) is no longer public. All moved to `internal/metal/`.

### Architecture note: import cycle resolution

`internal/metal/` cannot import the root package (circular dependency). Solution: internal/metal defines its own concrete types (`metal.Token`, `metal.GenerateConfig`, `metal.Model`), and `register_metal.go` in root provides a thin adapter (`metalAdapter`) that converts between root types (`mlx.Token`) and metal types.

### Impact on go-ml

`backend_mlx.go` must migrate from direct tensor manipulation to:
```go
m, _ := mlx.LoadModel(path)
ctx := context.Background()
for tok := range m.Generate(ctx, prompt, mlx.WithMaxTokens(n)) { ... }
if err := m.Err(); err != nil { ... }
```
253 LOC → ~60 LOC. Memory controls: `mlx.SetCacheLimit()`, `mlx.ClearCache()`, etc.

### Impact on go-i18n

```go
m, _ := mlx.LoadModel("/path/to/gemma-3-1b/")
ctx := context.Background()
for tok := range m.Generate(ctx, sentence, mlx.WithMaxTokens(32)) { ... }
```

### Memory management status

`Close()` stub is in place but does not yet explicitly free model weights. Per-step intermediate cleanup (`ClearCache()` per decode step) is implemented in the generate loop. Full deterministic cleanup awaits CLion Claude research on `mlx_array_free` safety (see `cpp/TODO.md`).

### Test results

- 148 existing tests moved to `internal/metal/` — all pass
- 7 new integration tests for public API — all pass
- Total: 155 tests passing

---

## 2026-02-19: Migration to go-inference Shared Interfaces

### What changed

go-mlx no longer defines its own `TextModel`, `Backend`, `Token`, `Message`, `GenerateConfig`, `GenerateOption`, `LoadConfig`, `LoadOption` types. These are now provided by `forge.lthn.ai/core/go-inference`, a zero-dependency shared interface package.

### Files removed

- `textmodel.go` — `Token`, `Message`, `TextModel` now in go-inference
- `options.go` — `GenerateConfig`, `GenerateOption`, `LoadConfig`, `LoadOption` now in go-inference
- `backend.go` — `Backend`, `Register`, `Get`, `Default`, `LoadModel` now in go-inference

### Files updated

- `register_metal.go` — implements `inference.Backend` (added `Available() bool`), adapts `inference.Token`/`inference.Message`
- `mlx_test.go` — all tests use `inference.*` types, added `TestListBackends`, `TestLoadOptions`, `TestLoadOptionsDefaults`
- `mlx.go` — package doc updated to show go-inference import pattern
- `go.mod` — added `forge.lthn.ai/core/go-inference` dependency (replace directive for local dev)
- `internal/metal/generate.go` — `GenerateConfig` gained `RepeatPenalty float32`

### What go-mlx still exports

- `MetalAvailable() bool` — convenience check
- `SetCacheLimit`, `SetMemoryLimit`, `GetActiveMemory`, `GetPeakMemory`, `ClearCache` — Metal-specific memory controls
- Side-effect import (`_ "forge.lthn.ai/core/go-mlx"`) registers the `"metal"` backend into go-inference's registry

### Consumer migration

Before:
```go
import "forge.lthn.ai/core/go-mlx"
m, _ := mlx.LoadModel(path)
for tok := range m.Generate(ctx, prompt, mlx.WithMaxTokens(128)) { ... }
```

After:
```go
import (
    "forge.lthn.ai/core/go-inference"
    _ "forge.lthn.ai/core/go-mlx" // register Metal backend
)
m, _ := inference.LoadModel(path)
for tok := range m.Generate(ctx, prompt, inference.WithMaxTokens(128)) { ... }
```

### New go-inference features available

- `inference.List()` — returns all registered backend names
- `inference.Backend.Available()` — hardware availability check
- `inference.WithRepeatPenalty(p)` — repetition penalty option
- `inference.WithContextLen(n)` — context window size
- `inference.WithGPULayers(n)` — GPU layer offload control (-1 = all)
- `inference.LoadConfig.GPULayers` defaults to -1 (full GPU offload)

### Test results

- 148 internal/metal tests — all pass
- 11 root integration tests — all pass
- Total: 159 tests passing

---

## 2026-02-19: CLion Claude Research Applied

### Contiguous Array Fix (data correctness bug)

`Floats()`, `DataInt32()`, and `Ints()` now automatically handle non-contiguous arrays. Previously, reading data from view arrays (Transpose, BroadcastTo, SliceAxis) returned silently wrong results.

**Fix**: Bound `mlx_contiguous` and `_mlx_array_is_row_contiguous` from mlx-c. The `ensureContiguous()` helper checks `IsRowContiguous()` and makes a contiguous copy when needed before accessing the raw data pointer.

The old workaround of `Reshape(arr, totalSize)` to force contiguity is no longer needed.

### TopP (Nucleus) Sampling Implemented

Was a stub that passed logits through unchanged. Now fully implemented:
1. Softmax to get probabilities
2. Argsort descending to get sorted indices
3. CumSum of sorted probabilities
4. Mask tokens where cumulative probability (excluding current) exceeds threshold
5. Scatter mask back to original positions via PutAlongAxis + argsort indices

### MinP Sampling Implemented

Was a stub. Now masks tokens whose probability is below `min_p * max_prob`. Uses MaxAxis to find the peak probability per position.

### New Bindings

| Function | Header | Purpose |
|----------|--------|---------|
| `mlx_contiguous` | ops.h | Force row-major contiguous layout |
| `_mlx_array_is_row_contiguous` | array.h | Check contiguity without copying |
| `mlx_cumsum` | ops.h | Cumulative sum (forward/reverse, inclusive/exclusive) |
| `mlx_sort_axis` | ops.h | Sort along axis |
| `mlx_argsort_axis` | ops.h | Indices that would sort |
| `mlx_greater` | ops.h | Element-wise comparison |
| `mlx_max_axis` | ops.h | Maximum along axis |
| `mlx_get_cache_memory` | memory.h | Current allocator cache size |
| `mlx_reset_peak_memory` | memory.h | Reset peak memory tracking |
| `mlx_set_wired_limit` | memory.h | Wired memory limit control |
| `mlx_metal_device_info` | metal.h | GPU hardware info |

### Test results

- 165 internal/metal tests — all pass
- 11 root integration tests — all pass
- Total: 176 tests passing

---

## 2026-02-19: Benchmark Baseline — M3 Ultra

29 benchmarks in `internal/metal/bench_test.go`. All times in ns/op, measured with `go test -bench=. -benchtime=2s`.

### Matrix Multiply

| Shape | ns/op | Notes |
|-------|------:|-------|
| 128×128 | 194,467 | CGO overhead dominates at small sizes |
| 512×512 | 255,288 | GPU starting to amortise |
| 1024×1024 | 474,900 | Sweet spot for Metal throughput |
| 2048×2048 | 4,173,797 | ~4ms — good for decode step |
| 4096×4096 | 10,715,051 | ~10.7ms — large context attention |
| 1×2048 → 32000 (token proj) | 626,087 | Output projection per token |

### Fused Metal Kernels

| Operation | Shape | ns/op |
|-----------|-------|------:|
| RMSNorm | 1×2048 | 156,696 |
| RMSNorm | 32×2048 | 225,164 |
| LayerNorm | 32×2048 | 184,514 |
| RoPE | 1×1×32×128 (decode) | 176,605 |
| RoPE | 1×32×512×128 (prefill) | 1,443,803 |
| SDPA causal | 1 head, seq=32 | 200,926 |
| SDPA causal | 32 heads, seq=128 | 515,477 |
| SDPA causal | 32 heads, seq=512 | 1,815,073 |

### Softmax & Reductions

| Operation | Shape | ns/op |
|-----------|-------|------:|
| Softmax | 1×1024 | 173,811 |
| Softmax | 32×32000 | 948,660 |
| Softmax | 1×128000 | 270,022 |
| Sum | 1M elements | 175,204 |
| Argmax | 1×32000 | 171,327 |

### Element-wise (1M elements)

| Operation | ns/op |
|-----------|------:|
| Add | 651,687 |
| Mul | 394,941 |
| SiLU | 1,192,843 |

### Layers

| Operation | Shape | ns/op |
|-----------|-------|------:|
| Linear | 1×2048 → 2048 | 181,417 |
| Linear | 32×2048 → 8192 | 471,038 |
| Embedding | 32 tokens, 32K vocab, 2048 dim | 219,154 |

### Sampling (vocab=32000)

| Strategy | ns/op |
|----------|------:|
| Greedy (argmax) | 172,698 |
| TopK=50, temp=1.0 | 542,635 |
| TopP=0.9, temp=1.0 | 713,538 |
| Full (TopP+MinP+TopK) | 731,118 |

### Key Observations

1. **CGO floor ~170μs**: All operations have a ~170μs minimum (greedy sample, RMSNorm single row, Sum 1M). This is the CGO call + Metal command buffer overhead.
2. **MatMul scales well**: 128² → 4096² is only ~55× slower for 1024× more work, showing good GPU utilisation.
3. **SDPA efficient**: 32-head seq=512 attention at 1.8ms is practical for real-time inference.
4. **Sampling overhead**: Full chain (TopP+MinP+TopK) adds ~560μs over greedy — acceptable per token.
5. **Linear layer**: Single-token forward through 2048→2048 at 181μs suggests ~5500 layers/sec ceiling for per-token decode.
