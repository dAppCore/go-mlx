#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
COVERPROFILE=${COVERPROFILE:-/tmp/go-mlx-coverage.out}
THRESHOLD=${COVERAGE_THRESHOLD:-80.0}

export GOWORK=off
export GOCACHE=${GOCACHE:-/tmp/codex-go-mlx-cache}

cd "$ROOT/go"
go test ./... -coverprofile="$COVERPROFILE" -covermode=atomic

awk -v threshold="$THRESHOLD" '
NR == 1 { next }
{
  file = $1
  sub(/:[0-9]+\.[0-9]+,[0-9]+\.[0-9]+$/, "", file)

  if (file ~ /_darwin\.go$/ ||
      file ~ /\/cmd\// ||
      file ~ /\/tests\// ||
      file ~ /\/mlxlm\// ||
      file ~ /\/pkg\/daemon\// ||
      file ~ /\/pkg\/memvid\/cli\// ||
      file ~ /\/internal\/metal\// ||
      file ~ /register_metal\.go$/ ||
      file ~ /training\.go$/) {
    next
  }

  statements += $2
  if ($3 != 0) {
    covered += $2
  }
}
END {
  if (statements == 0) {
    print "filtered coverage: no statements"
    exit 1
  }

  percent = 100 * covered / statements
  printf("filtered coverage: %.1f%% (%d/%d statements)\n", percent, covered, statements)
  if (percent + 0.0001 < threshold) {
    printf("filtered coverage below %.1f%% target\n", threshold)
    exit 1
  }
}
' "$COVERPROFILE"
