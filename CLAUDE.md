# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Native Apple Metal GPU inference and research-grade training pipeline via mlx-c CGO bindings. Module: `dappco.re/go/mlx`

Implements the `inference.Backend` and `inference.TextModel` interfaces from `dappco.re/go/inference` for Apple Silicon (M1-M4) GPUs using Metal compute shaders via the mlx-c C API. Supports Gemma 3, Gemma 4 (dense and MoE, with paged KV), Qwen 2/3, and Llama 3 from HuggingFace safetensors directories and GGUF checkpoints. The root package also exposes a non-LLM frame-compute API (pixel buffers, kernels) for emulator and image workloads.

## Platform

**darwin/arm64 only.** All CGO files carry `//go:build darwin && arm64`. A stub provides `MetalAvailable() bool` returning false on every other platform. Files that need the native MLX runtime use the build tag; non-Darwin builds compile against `*_stub.go` files.

## How to use this repo — drive `task`, never hand-roll the build

**This repo is driven by [Taskfile.yml](Taskfile.yml) (`go-task`). Run `task <target>` — do NOT reconstruct the build with bare `go build` / `go test` and manual env exports.** The Taskfile already bakes in everything the build needs:

- `GOCACHE` (default `/private/tmp/codex-go-mlx-cache` — the repo's shared build cache; not a "codex env", just the default)
- `GO_DARWIN_LDFLAGS = -extldflags=-mmacosx-version-min=26.0` (Metal 4 floor — the build fails without it)
- `MLX_METALLIB_PATH = {{.ROOT_DIR}}/dist/lib/mlx.metallib`
- `-tags metal_runtime` on the test path

Hand-exporting those yourself is how you get a subtly-wrong build and waste a session. `task fmt; go-task --list` shows every target. The Go module lives in `go/`; the Taskfile `dir: go` handles the `cd` for you.

```bash
# Build the binary (self-contained — embeds the gzipped metallib). Output: bin/lthn-mlx
task build:lthn

# Compile the native engine's OWN fused Metal kernels (router-topk, q4 lm-head argmax, bf16)
#   -> dist/lib/lthn_kernels.metallib, loaded beside mlx.metallib. Build after editing pkg/native/kernels/*.metal.
task build:kernels

# Run the whole Go suite (metal_runtime tag + ldflags, on the GPU)
task test

# fmt + vet + test
task qa

# Real coverage figure (metal_runtime + model_eval tags) -> /tmp/go-mlx-coverage.out
task cov

# C++ side (standalone lib/mlx build + the kernel-bridge tests). First run cold-builds MLX ~15 min.
task test:cpp        # vendored MLX suite
task test:cpp:kernels  # go-mlx's own Metal-kernel bridges

# Clean
task clean
```

**Fresh checkout:** `git submodule update --init --recursive`, then build the MLX C library + metallib (`go generate ./...` from `go/`, or the CMake path in `docs/build.md`) so `dist/lib/mlx.metallib` exists before `task build:lthn` can embed it. `dist/lib/` is gitignored (rebuilt per checkout); `dist/include/` headers are committed for module consumers.

### The two engines, and trying them

| Engine | Path | How to run |
|--------|------|-----------|
| **cgo metal** (mature: MTP, paged KV, cache modes) | `go/internal/metal/` | `./bin/lthn-mlx generate <model-dir>` / `serve` |
| **no-cgo native** (`pkg/native` + `pkg/model` — the contract engine, current focus) | `go/pkg/native/` | add `-native`: `./bin/lthn-mlx generate -native …` |

```bash
# Try the native engine end-to-end (greedy, deterministic). Model = a gemma4 HF/MLX snapshot dir.
./bin/lthn-mlx generate -native -temp 0 -max-tokens 64 -prompt "Explain RoPE in one sentence." \
  ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/*/

# Serve it over the OpenAI/Anthropic/Ollama API, then curl /v1/chat/completions
./bin/lthn-mlx serve -native <model-dir>          # default :11434
```

`generate -native` does NOT yet support `-trace` or `-state` (cgo-engine only); those exit 2. A run-quick smoke of the served engine also exists as the `lethean-lem` skill (`lem.sh smoke e2b`), but for *driving the repo itself* prefer `task` + the binary directly.

## Repository Layout

After Mantis #1241, all Go code lives under `go/`:

```
go/                          Go module root (dappco.re/go/mlx)
  *.go                       Public root API: model, tokenizer, compute, training, eval, distill, GRPO, hf-fit, merge, gguf-quantize, kv-snapshot, lora-fuse
  cmd/mlx/                   CLI tool (built with `-o core-mlx`; consumers rename: lthn-mlx)
  cmd/violet/                Unix-socket sidecar daemon
  internal/metal/            All CGO code (mlx-c bindings)
  mlxlm/                     CGO-free Python subprocess backend
  pkg/daemon/                Daemon implementation
  pkg/memvid/                Deprecated State codec compatibility shim
  tests/                     Integration tests
cpp/                         C++ side (CLion-side companion)
docs/                        Markdown documentation
examples/                    Per-feature usage examples (markdown)
external/                    Vendored core libraries
lib/mlx/                     Upstream mlx submodule (pinned at v0.31.1)
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

`lib/mlx` is pinned at upstream tag `v0.31.1`. Local patches that we do not upstream live in `patches/` as standalone diff files (e.g. `patches/mlx-metallib-path.patch` for the `MLX_METALLIB_PATH` env-var override). Patches are not auto-applied — run them inside the submodule manually when their function is needed:

```bash
git -C lib/mlx apply ../../patches/mlx-metallib-path.patch
```

## Downstream Consumers

| Package | Role |
|---------|------|
| `dappco.re/go/core/ml` | Imports go-inference + go-mlx for the Metal backend training loop |
| `dappco.re/go/core/i18n` | Gemma3-1B domain classification (Phase 2a) |
| `dappco.re/go/core/rocm` | Sibling AMD GPU backend, same go-inference interfaces |
