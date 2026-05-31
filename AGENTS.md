# go-mlx Agent Guide

This repository provides Go bindings, model loaders, and a research-grade
training pipeline for MLX on Apple Silicon. Module: `dappco.re/go/mlx`.

## Layout (post Mantis #1241)

All Go code lives under `go/`:

- `go/` — root package: public model, tokenizer, compute, training, eval,
  distill, GRPO, hf-fit, merge, gguf-quantize, kv-snapshot, lora-fuse APIs
- `go/internal/metal/` — CGO boundary to `mlx-c`; do not move CGO code out
- `go/mlxlm/` — subprocess backend for Python `mlx-lm` (CGO-free; build tag
  `nomlxlm` removes it)
- `go/cmd/violet/` and `go/pkg/daemon/` — local Violet Unix-socket sidecar
- `cpp/` — C++ side companion (CLion-side worktree)
- `lib/mlx/` — upstream MLX submodule pinned at `v0.31.1`
- `patches/` — local patches against `lib/mlx` (manual apply only)
- `docs/`, `examples/` — markdown documentation and per-feature usage examples

## Platform Boundaries

Files that need the native MLX runtime use `//go:build darwin && arm64`.
Unsupported builds compile against the `*_stub.go` files and a stub
`MetalAvailable() bool` that returns false. Do not move CGO code out of
`go/internal/metal/`.

The native path targets [macOS Tahoe 26.0+](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes)
on Apple Silicon. The floor is intentional: the Metal 4 API generation this
runner is built around shipped with macOS 26, including lower-overhead command
encoding, explicit compilation control, tensor resources, and machine-learning
passes. Keep build and test invocations aligned with that floor by passing
`-ldflags "-extldflags=-mmacosx-version-min=26.0"` when compiling native code.
See `docs/operator/deployment.md` and `docs/operator/metallib-and-variants.md`
for the full reference chain.

## Conventions

- UK English in code, comments, and docs (colour, organisation, behaviour)
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`
- Conventional commits: `type(scope): description` — scopes include `metal`,
  `api`, `mlxlm`, `cpp`, `docs`, `repo`, `deps`
- Co-Author trailer: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Use `dappco.re/go` core helpers for fmt, errors, JSON, filesystem, path,
  env, byte buffers, and string ops — do not import the wrapped stdlib
  packages directly

## Test Patterns

Tests are file-aware. Public functions and methods in `foo.go` have their
Good, Bad, and Ugly triplets in `foo_test.go`, and runnable examples in
`foo_example_test.go`. Native tests skip only when the local machine lacks
the required Metal runtime or test model assets. Keep examples small and
checkable so they document the public API without requiring heavyweight
model downloads.

## Sandboxing Notes

Before handing off, run the repository gates from the checked-in workspace; do
not use `GOWORK=off` unless the user explicitly asks for an isolated module
check. On sandboxed systems, set `GOCACHE` to a writable directory such as
`/tmp/codex-go-mlx-cache` so Go can compile without touching the user cache.
If the sandbox cannot resolve the bundled `mlx.metallib`, apply
`patches/mlx-metallib-path.patch` inside `lib/mlx` to enable the
`MLX_METALLIB_PATH` env-var override (not auto-applied).

## What's Inside the Public Surface

Beyond the inference path, the root package owns the research-grade
pipeline: knowledge distillation (`RunKnowledgeDistillation`), GRPO
(`RunGRPOReasoningTraining`), dataset eval (`RunModelEval`), LoRA fusion
(`FuseLoRAIntoModelPack`), model merging (`MergeModelPacks`), native
GGUF quantisation (`QuantizeModelPackToGGUF`), KV snapshots
(`KVSnapshot.Save` / `LoadKVSnapshot`), and HuggingFace Hub metadata
(`HuggingFaceModelSource`). See `docs/` and `examples/` for full
walkthroughs.
