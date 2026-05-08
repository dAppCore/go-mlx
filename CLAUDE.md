# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Native Apple Metal GPU inference and research-grade training pipeline via mlx-c CGO bindings. Module: `dappco.re/go/mlx`

Implements the `inference.Backend` and `inference.TextModel` interfaces from `dappco.re/go/inference` for Apple Silicon (M1-M4) GPUs using Metal compute shaders via the mlx-c C API. Supports Gemma 3, Gemma 4 (dense and MoE, with paged KV), Qwen 2/3, and Llama 3 from HuggingFace safetensors directories and GGUF checkpoints. The root package also exposes a non-LLM frame-compute API (pixel buffers, kernels) for emulator and image workloads.

## Platform

**darwin/arm64 only.** All CGO files carry `//go:build darwin && arm64`. A stub provides `MetalAvailable() bool` returning false on every other platform. Files that need the native MLX runtime use the build tag; non-Darwin builds compile against `*_stub.go` files.

## Build & Test

```bash
# Build mlx-c C library (required on fresh checkout, ~2min on M3 Ultra)
git submodule update --init --recursive
go generate ./...

# Run all tests
go test ./...

# Run a single test
go test -run TestRMSNorm_Good ./go/internal/metal/

# Run benchmarks
go test -bench=. -benchtime=2s ./go/internal/metal/

# Lint
golangci-lint run ./...

# Clean rebuild (if dist/ is stale)
rm -rf build dist && go generate ./...
```

The compiled libraries (`dist/lib/`) are gitignored and must be rebuilt on each fresh checkout. Headers in `dist/include/` are committed for Go module consumers. On sandboxed systems, set `GOCACHE` to a writable directory such as `/tmp/codex-go-mlx-cache`.

## Repository Layout

After Mantis #1241, all Go code lives under `go/`:

```
go/                          Go module root (dappco.re/go/mlx)
  *.go                       Public root API: model, tokenizer, compute, training, eval, distill, GRPO, hf-fit, merge, gguf-quantize, kv-snapshot, lora-fuse
  cmd/violet/                Unix-socket sidecar daemon
  internal/metal/            All CGO code (mlx-c bindings)
  mlxlm/                     CGO-free Python subprocess backend
  pkg/daemon/                Daemon implementation
  pkg/memvid/                Memvid storage CLI
  tests/                     Integration tests
cpp/                         C++ side (CLion-side companion)
docs/                        Markdown documentation
examples/                    Per-feature usage examples (markdown)
external/                    Vendored core libraries
lib/mlx/                     Upstream mlx submodule (pinned at v0.30.1)
patches/                     Local patches to lib/mlx (not auto-applied)
```

## Architecture

Three-layer design:

1. **Root package `go/` (`mlx.go`, `register_metal.go`, `training.go`, etc.)** — public API surface. `init()` auto-registers the `"metal"` backend with go-inference. `metalAdapter` converts between `inference.*` and `metal.*` types. Training type aliases (`Array`, `LoRAAdapter`, `GradFn`, `AdamW`) are re-exported for downstream `go-ml`. The root package also owns the new research-grade pipeline surface: distillation, GRPO, eval reporting, LoRA fusion, model merging, GGUF quantisation, KV snapshots, HF Hub metadata.

2. **`go/internal/metal/`** — all CGO code. Key files:
   - `metal.go` — init, error handler (atomic C callback), `Eval`/`Materialize`
   - `generate.go` — `Model`, `Generate`, `Chat`, batch inference
   - `gemma3.go`, `gemma4.go`, `qwen3.go`, `llama.go` — model decoders implementing `InternalModel`
   - `tokenizer.go` — BPE tokeniser (SentencePiece + GPT-2)
   - `cache.go` — `KVCache` (256-token chunks), `RotatingKVCache` (sliding window), `PagedKVCache` (block-oriented for Gemma 4)
   - `fast.go` — fused Metal kernels: RMSNorm, LayerNorm, RoPE, SDPA
   - `grad.go`, `lora.go`, `optim.go` — autodiff, LoRA adapters, AdamW

3. **`go/mlxlm/`** — CGO-free Python subprocess backend (`"mlx_lm"`). Spawns `bridge.py` over JSON Lines. Build tag `nomlxlm` removes it. Tests use `testdata/mock_bridge.py` (no GPU or Python ML deps required).

MLX uses **lazy evaluation**: operations build a computation graph dispatched to Metal only on `Eval()`. `Detach()` breaks graph connections to free GPU memory between generation steps. `Array` wraps `mlx_array` C handles with `runtime.SetFinalizer` calling `mlx_array_free`; explicit `Free()` releases immediately.

See `docs/architecture.md` for full details (attention, sampling chain, memory model).

## Documentation

| Document | Topic |
|----------|-------|
| `docs/index.md` | Top-level guide, feature matrix, supported models, perf baseline |
| `docs/architecture.md` | CGO binding, model architectures, weight loading, KV cache, attention |
| `docs/compute.md` | Frame-oriented Metal compute sessions, pixel buffers, kernels |
| `docs/models.md` | Model loading, supported architectures, tokenisation, chat templates |
| `docs/training.md` | LoRA fine-tuning, AdamW, gradient computation, loss functions, LoRA fusion |
| `docs/distillation.md` | Knowledge distillation (KL, soft cross-entropy) |
| `docs/grpo.md` | Group-relative policy optimisation for RL |
| `docs/eval.md` | Dataset-native perplexity, quality probes, eval reports |
| `docs/model-operations.md` | Model merge (Linear/SLERP/TIES/DARE), GGUF quantise, KV snapshot, HF fit |
| `docs/development.md` | Prerequisites, CGO flags, test patterns, benchmarks |
| `docs/build.md` | Build pipeline, CMake, build tags |
| `docs/history.md` | Completed phases, commit hashes, known limitations |
| `docs/model-state-roadmap.md` | Native session restore, state bundles, training runner, model packs |
| `docs/plans/` | Design and implementation plans (preserved, do not delete) |
| `examples/` | Per-feature usage examples organised by type (inference, training, model-ops, compute, eval, daemon) |

## Coding Standards

- UK English (colour, organisation, centre, initialise, behaviour)
- `go test ./...` must pass before commit
- Conventional commits: `type(scope): description` — scopes include `metal`, `api`, `mlxlm`, `cpp`, `docs`, `repo`, `deps`
- Co-Author trailer: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`
- Use `dappco.re/go` core helpers for fmt, errors, JSON, filesystem, path, env, byte buffers, and string ops — do not reach for stdlib equivalents covered by the wrapper policy

## Test Patterns

Tests use `_Good`, `_Bad`, `_Ugly` suffix convention:
- `_Good` — happy path
- `_Bad` — expected error conditions
- `_Ugly` — panic / edge cases

Public functions in `foo.go` have their Good/Bad/Ugly triplets in `foo_test.go`, and runnable examples in `foo_example_test.go`. Tests requiring model files on disk use `t.Skip()` when the path is absent. Model path: `/Volumes/Data/lem/safetensors/`.

## Model Formats

- **HuggingFace safetensors** directory packs (`config.json`, `tokenizer.json`, one or more `*.safetensors` shards)
- **GGUF** single-file checkpoints (auto-detected)

Architecture is detected from `config.json` (`model_type`) for safetensors and from GGUF metadata for GGUF.

## Submodule Patches

`lib/mlx` is pinned at upstream tag `v0.30.1`. Local patches that we do not upstream live in `patches/` as standalone diff files (e.g. `patches/mlx-metallib-path.patch` for the `MLX_METALLIB_PATH` env-var override). Patches are not auto-applied — run them inside the submodule manually when their function is needed:

```bash
git -C lib/mlx apply ../../patches/mlx-metallib-path.patch
```

## Downstream Consumers

| Package | Role |
|---------|------|
| `dappco.re/go/core/ml` | Imports go-inference + go-mlx for the Metal backend training loop |
| `dappco.re/go/core/i18n` | Gemma3-1B domain classification (Phase 2a) |
| `dappco.re/go/core/rocm` | Sibling AMD GPU backend, same go-inference interfaces |
