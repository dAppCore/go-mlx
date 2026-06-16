#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
COVERPROFILE=${COVERPROFILE:-/tmp/go-mlx-coverage.out}

export GOCACHE=${GOCACHE:-/tmp/codex-go-mlx-cache}

cd "$ROOT/go"

# go-mlx IS the Apple/Metal engine: pkg/metal and the model trunks only compile
# behind the `metal_runtime` tag, linked against the built metallib, on darwin.
# Running coverage WITHOUT the tag silently drops ~40% of statements (the entire
# engine) from the denominator and prints a flattering, meaningless figure — the
# no-tag number reads ~69% where the real one is ~80%. So: measure the real thing
# where we can, and refuse to pass a partial run off as the project figure where
# we can't, rather than hide the gap behind an exclusion list.
METALLIB="${MLX_METALLIB_PATH:-$ROOT/dist/lib/mlx.metallib}"

if [ "$(uname)" = "Darwin" ] && [ -f "$METALLIB" ]; then
	export MLX_METALLIB_PATH="$METALLIB"
	echo "coverage: full metal_runtime build (engine included) via $METALLIB"
	# -p 1 serialises packages: the metal tests share one GPU, so parallel
	# packages would contend and can OOM. The suite uses synthetic fixtures —
	# it never loads real model weights.
	go test -tags metal_runtime -p 1 \
		-ldflags "-extldflags=-mmacosx-version-min=26.0" \
		./... -coverprofile="$COVERPROFILE" -covermode=atomic
else
	echo "WARNING: not darwin, or metallib missing at $METALLIB."
	echo "WARNING: the metal engine (~40% of statements) is NOT compiled or"
	echo "WARNING: measured in this run — this profile is PARTIAL. Do not treat"
	echo "WARNING: it as the project coverage figure. Measure on darwin/arm64"
	echo "WARNING: with the metallib built (task build / lib/mlx -> dist/lib)."
	go test ./... -coverprofile="$COVERPROFILE" -covermode=atomic
fi

echo "=== go-mlx total coverage ==="
go tool cover -func="$COVERPROFILE" | tail -1
