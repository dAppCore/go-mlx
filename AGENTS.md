Top rule: keep `GOAL.md` and tracker files as compact worklists only. Do not add, preserve, or update progress, proof, benchmark, savings, changelog, or status-diary notes there unless the user explicitly asks for tracking changes; remove completed task lines and report evidence in the turn or commit instead.

# go-mlx Agent Guide

- Tracker hygiene: do not add progress, proof, benchmark, savings, or per-slice status notes to `GOAL.md` or tracker files. Keep them as compact contracts/worklists; when a task is done, remove the task line and report evidence in the turn or commit.

Module `dappco.re/go/mlx`; Go lives in `go/`.

- Native: `go/pkg/native` stays CGO-free; CGO stays in `go/internal/metal`; `darwin && arm64`, macOS 26+.
- Test env: repo `go.work`; `GOCACHE=/private/tmp/codex-go-mlx-cache`; `MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib`; `-ldflags "-extldflags=-mmacosx-version-min=26.0"`.
- Style/tests: UK English; EUPL SPDX on new files; use `dappco.re/go` helpers; file-aware tests; native skips only for missing runtime/assets.
