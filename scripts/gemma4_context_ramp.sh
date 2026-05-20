#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2

set -euo pipefail

ROOT="${GO_MLX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${GO_MLX_BIN:-$ROOT/bin/lthn-mlx}"
MODEL="${GO_MLX_MODEL:-/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd}"
MODEL_LABEL="${GO_MLX_MODEL_LABEL:-gemma4-e2b-4bit}"
PROMPT_FILE="${GO_MLX_PROMPT_FILE:-$ROOT/README.md}"
PROMPT_SUFFIX="${GO_MLX_PROMPT_SUFFIX:-}"
PROMPT_SUFFIX_FILE="${GO_MLX_PROMPT_SUFFIX_FILE:-}"
OUT_DIR="${GO_MLX_OUT_DIR:-$ROOT/docs/runtime}"
GOWORK_PATH="${GO_MLX_GOWORK:-$ROOT/go.work}"
GOCACHE_PATH="${GOCACHE:-/private/tmp/codex-go-mlx-cache}"
METALLIB_PATH="${MLX_METALLIB_PATH:-$ROOT/dist/lib/mlx.metallib}"
POWER_WATTS="${GO_MLX_POWER_WATTS:-100}"
MAX_TOKENS="${GO_MLX_RAMP_MAX_TOKENS:-128}"
RUNS="${GO_MLX_RAMP_RUNS:-3}"
DATE_STAMP="${GO_MLX_DATE_STAMP:-$(date +%F)}"
STEPS="${GO_MLX_RAMP_STEPS:-1:4096 4:16384 8:32768 13:32768 24:65536 46:131072}"

mkdir -p "$OUT_DIR" "$GOCACHE_PATH"

if [[ ! -x "$BIN" ]]; then
  echo "missing executable: $BIN" >&2
  echo "build it with: (cd $ROOT/go && env GOWORK=$GOWORK_PATH GOCACHE=$GOCACHE_PATH MLX_METALLIB_PATH=$METALLIB_PATH go build -trimpath -o ../bin/lthn-mlx ./cmd/mlx/)" >&2
  exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 2
fi

prompt_suffix_args=()
if [[ -n "$PROMPT_SUFFIX_FILE" ]]; then
  if [[ ! -f "$PROMPT_SUFFIX_FILE" ]]; then
    echo "missing prompt suffix file: $PROMPT_SUFFIX_FILE" >&2
    exit 2
  fi
  prompt_suffix_args=(-prompt-suffix-file "$PROMPT_SUFFIX_FILE")
elif [[ -n "$PROMPT_SUFFIX" ]]; then
  prompt_suffix_args=(-prompt-suffix "$PROMPT_SUFFIX")
fi

for step in $STEPS; do
  repeat="${step%%:*}"
  context="${step#*:}"
  artifact="$OUT_DIR/${DATE_STAMP}-go-mlx-${MODEL_LABEL}-fast-gemma4-lane-context-ramp-repeat${repeat}-ctx${context}-g${MAX_TOKENS}-r${RUNS}-energy${POWER_WATTS}w.json"
  stderr_artifact="${artifact%.json}.stderr"

  echo "context ramp: repeat=$repeat context=$context max_tokens=$MAX_TOKENS runs=$RUNS"
  env \
    GOWORK="$GOWORK_PATH" \
    GOCACHE="$GOCACHE_PATH" \
    MLX_METALLIB_PATH="$METALLIB_PATH" \
    "$BIN" driver-profile \
      -report-file "$artifact" \
      -fast-gemma4-lane \
      -prompt-file "$PROMPT_FILE" \
      -prompt-repeat "$repeat" \
      "${prompt_suffix_args[@]}" \
      -context "$context" \
      -max-tokens "$MAX_TOKENS" \
      -runs "$RUNS" \
      -estimate-power-watts "$POWER_WATTS" \
      -include-output=false \
      "$MODEL" 2>"$stderr_artifact"

  if command -v jq >/dev/null 2>&1; then
    jq '{prompt_repeat, max_tokens, requested_runs, load, summary, estimated_energy, error}' "$artifact"
  fi
done
