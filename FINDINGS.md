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
