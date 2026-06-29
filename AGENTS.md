Rule -1: Do not add tracking, status notes, benchmark/savings notes, or "what got faster" writeups to `GOAL.md` or tracker files during normal implementation work. Do not read those files to reorient at the start of a turn; only open or edit them when the user explicitly asks for a tracking change.

Rule 0: `GOAL.md` and tracker files are compact task queues only. Do not read them at the start of each turn, do not use them as progress logs, and do not add, preserve, or update proof, benchmark, savings, changelog, status-diary, or "what got faster" notes there unless the user explicitly asks for tracking changes. When a tracked task is done, remove only that task line and report evidence in the turn or commit instead.

# go-mlx Agent Guide

Module `dappco.re/go/mlx`; Go lives in `go/`.

- Native: `go/pkg/native` stays CGO-free; CGO stays in `go/internal/metal`; `darwin && arm64`, macOS 26+.
- Test env: repo `go.work`; `GOCACHE=/private/tmp/codex-go-mlx-cache`; `MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib`; `-ldflags "-extldflags=-mmacosx-version-min=26.0"`.
- Style/tests: UK English; EUPL SPDX on new files; use `dappco.re/go` helpers; file-aware tests; native skips only for missing runtime/assets.
