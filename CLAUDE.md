# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Native Apple Metal GPU inference via mlx-c CGO bindings. Module: `forge.lthn.ai/core/go-mlx`

Implements the `inference.Backend` interface from `forge.lthn.ai/core/go-inference` for Apple Silicon (M1-M4) GPUs using Metal compute shaders via the mlx-c C API. Supports Gemma 3, Qwen 2/3, and Llama 3 architectures from HuggingFace safetensors format.

## Platform

**darwin/arm64 only.** All CGO files carry `//go:build darwin && arm64`. A stub (`mlx_stub.go`) provides `MetalAvailable() bool` returning false on other platforms.

## Build & Test

```bash
# Build mlx-c C library (required on fresh checkout, ~2min on M3 Ultra)
go generate ./...

# Run all tests
go test ./...

# Run a single test
go test -run TestRMSNorm_Good ./internal/metal/

# Run benchmarks
go test -bench=. -benchtime=2s ./internal/metal/

# Lint
golangci-lint run ./...

# Clean rebuild (if dist/ is stale)
rm -rf build dist && go generate ./...
```

The compiled libraries (`dist/lib/`) are gitignored and must be rebuilt on each fresh checkout. Headers in `dist/include/` are committed for Go module consumers.

## Go Workspace

go-mlx uses a `replace` directive for local development alongside go-inference:
```
replace forge.lthn.ai/core/go-inference => ../go-inference
```
After adding modules or changing dependencies: `go work sync`

## Architecture

Three-layer design:

1. **Root package** (`mlx.go`, `register_metal.go`, `training.go`) — public API surface. `init()` auto-registers the `"metal"` backend with go-inference. `metalAdapter` converts between `inference.*` and `metal.*` types. Training type aliases (`Array`, `LoRAAdapter`, `GradFn`, `AdamW`) are re-exported for direct use by downstream `go-ml`.

2. **`internal/metal/`** — all CGO code lives here. Key files:
   - `metal.go` — init, error handler (atomic C callback), `Eval`/`Materialize`
   - `generate.go` — `Model`, `Generate`, `Chat`, batch inference
   - `gemma3.go`, `qwen3.go` — model architecture decoders implementing `InternalModel`
   - `tokenizer.go` — BPE tokeniser (SentencePiece + GPT-2)
   - `cache.go` — `KVCache` (unbounded, 256-token chunks) + `RotatingKVCache` (sliding window)
   - `fast.go` — fused Metal kernels: RMSNorm, LayerNorm, RoPE, SDPA
   - `grad.go`, `lora.go`, `optim.go` — training: autodiff, LoRA adapters, AdamW

3. **`mlxlm/`** — CGO-free Python subprocess backend (`"mlx_lm"`). Spawns `bridge.py` communicating over JSON Lines. Build tag `nomlxlm` removes it. Tests use `testdata/mock_bridge.py` — no GPU or Python ML deps needed.

MLX uses **lazy evaluation**: operations build a computation graph dispatched to Metal only on `Eval()`. `Detach()` breaks graph connections to free GPU memory between generation steps. `Array` wraps `mlx_array` C handles with `runtime.SetFinalizer` calling `mlx_array_free`; explicit `Free()` releases immediately.

See `docs/architecture.md` for full details (attention, sampling chain, memory model).

## Documentation

- `docs/architecture.md` — CGO binding, model architectures, weight loading, KV cache, attention, batch inference, training
- `docs/development.md` — prerequisites, CGO flags, test patterns, benchmarks, adding new ops/architectures
- `docs/history.md` — completed phases with commit hashes, known limitations
- `docs/plans/` — design and implementation plans (preserved, do not delete)

## Coding Standards

- UK English (colour, organisation, centre, initialise, behaviour)
- `go test ./...` must pass before commit
- Conventional commits: `type(scope): description` — scopes: `metal`, `api`, `mlxlm`, `cpp`, `docs`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`

## Test Patterns

Tests use `_Good`, `_Bad`, `_Ugly` suffix convention:
- `_Good` — happy path
- `_Bad` — expected error conditions
- `_Ugly` — panic / edge cases

Tests requiring model files on disk use `t.Skip()` when the path is absent. Model path: `/Volumes/Data/lem/safetensors/`

## Model Format

**Safetensors** (HuggingFace format). NOT GGUF. Models must include matching `tokenizer.json`.

## Downstream Consumers

- `forge.lthn.ai/core/go-ml` — imports go-inference + go-mlx for Metal backend
- `forge.lthn.ai/core/go-i18n` — needs Gemma3-1B inference for domain classification
- `forge.lthn.ai/core/go-rocm` — sibling backend for AMD GPUs, same go-inference interfaces
