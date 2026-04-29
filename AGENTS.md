# go-mlx Agent Guide

This repository provides Go bindings and adapter layers for MLX on Apple
Silicon. The root `mlx` package exposes the public model, tokenizer, compute,
and LoRA APIs; `internal/metal` owns the cgo boundary to `mlx-c`; `mlxlm`
provides the subprocess backend for Python `mlx-lm`; and `pkg/daemon` contains
the local Violet Unix-socket sidecar.

Keep platform boundaries explicit. Files that need the native MLX runtime use
`darwin && arm64` build tags, while unsupported builds use the root stub files.
Do not move cgo code out of `internal/metal`, and do not add direct stdlib
imports covered by the core wrapper policy. Use `dappco.re/go` helpers for
formatting, errors, JSON, filesystem, path, environment, byte buffers, and
string operations.

Tests are file-aware. Public functions and methods in `foo.go` have their
Good, Bad, and Ugly triplets in `foo_test.go`, and runnable examples in
`foo_example_test.go`. Native tests must skip only when the local machine lacks
the required Metal runtime or test model assets. Keep examples small and
checkable so they document the public API without requiring heavyweight model
downloads.

Before handing off, run the repository gates from the brief with
`GOWORK=off`. On sandboxed systems, set `GOCACHE` to a writable directory such
as `/tmp/codex-go-mlx-cache` so Go can compile without touching the user cache.
