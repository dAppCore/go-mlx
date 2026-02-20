# CLAUDE.md

## What This Is

Native Apple Metal GPU inference via mlx-c bindings. Module: `forge.lthn.ai/core/go-mlx`

Implements the `inference.Backend` interface from [`forge.lthn.ai/core/go-inference`](https://forge.lthn.ai/core/go-inference) for Apple Silicon (M1-M4) GPUs using Metal compute shaders via the [mlx-c](https://github.com/ml-explore/mlx-c) C API.

## Platform

**darwin/arm64 only.** All CGO files carry `//go:build darwin && arm64`. A stub (`mlx_stub.go`) provides `MetalAvailable() bool` returning false on other platforms.

## Build

```bash
# Step 1: Build mlx-c C library via CMake (fetches mlx-c v0.4.1)
go generate ./...

# Step 2: Run tests (must be on Apple Silicon)
go test ./...
```

## Architecture

See `docs/architecture.md` for the full architecture reference.

## Documentation

- `docs/architecture.md` — CGO binding, model architectures, weight loading, tokenisation, KV cache, attention, batch inference, training, mlxlm backend, go-inference integration
- `docs/development.md` — Prerequisites, build/test, CGO flags, test patterns, benchmarks, coding standards
- `docs/history.md` — Completed phases with commit hashes, known limitations, future considerations
- `docs/plans/` — Design and implementation plans (preserved, do not delete)

## Coding Standards

- UK English (colour, organisation, centre)
- `go test ./...` must pass before commit
- Conventional commits: `type(scope): description`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`

## Test Patterns

Tests use `_Good`, `_Bad`, `_Ugly` suffix convention. Tests requiring model files on disk use `t.Skip()` when the path is absent.

## Model Format

**Safetensors** (HuggingFace format). NOT GGUF.
- Example: `/Volumes/Data/lem/safetensors/gemma-3/`
- Models must be in safetensors format with matching `tokenizer.json`

## Downstream Consumers

- `forge.lthn.ai/core/go-ml` — imports go-inference + go-mlx for Metal backend
- `forge.lthn.ai/core/go-i18n` — Phase 2a needs Gemma3-1B inference for domain classification
- `forge.lthn.ai/core/go-rocm` — sibling backend for AMD GPUs, same go-inference interfaces
