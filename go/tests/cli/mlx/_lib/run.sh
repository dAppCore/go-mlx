#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# AX-10 (RFC-CORE-008 §10) CLI-test helpers for the lthn-mlx engine binary.
# Each command test sources this, builds the binary, runs it in its OWN process,
# and asserts on exit-code + output. Assertions use jq / grep / exit-code only —
# NO python (keeps the repo Sonar-clean). Ported from
# core/agent/tests/cli/_lib/run.sh plus mlx build/model resolution helpers.
#
# Why a separate process per command: model_eval tests crammed into one shared
# `go test` binary pollute global registries (a prior test left state that broke
# the real gemma4 weight load — the serve-turn `reshape size 4 -> (3,3)` bug).
# An isolated `mlx <command>` invocation cannot be poisoned by a sibling test.

run_capture_stdout() {
	local expected_status="$1"
	local output_file="$2"
	shift 2
	set +e
	"$@" >"$output_file"
	local status=$?
	set -e
	if [[ "$status" -ne "$expected_status" ]]; then
		printf 'expected exit %s, got %s\n' "$expected_status" "$status" >&2
		[[ -s "$output_file" ]] && { printf 'stdout:\n' >&2; cat "$output_file" >&2; }
		return 1
	fi
}

run_capture_all() {
	local expected_status="$1"
	local output_file="$2"
	shift 2
	set +e
	"$@" >"$output_file" 2>&1
	local status=$?
	set -e
	if [[ "$status" -ne "$expected_status" ]]; then
		printf 'expected exit %s, got %s\n' "$expected_status" "$status" >&2
		[[ -s "$output_file" ]] && { printf 'output:\n' >&2; cat "$output_file" >&2; }
		return 1
	fi
}

assert_jq() { jq -e "$1" "$2" >/dev/null; }
assert_contains() { grep -Fq "$1" "$2"; }

# go_root echoes the go-mlx module root (go/) from a command-leaf dir
# (tests/cli/mlx/<command>/). The depth is fixed by AX-10's path = command map.
go_root() { ( cd "${1:-.}/../../../.." && pwd ); }

# build_mlx builds the metal-tagged lthn-mlx binary into tests/cli/mlx/bin/mlx
# (incremental after the first cgo compile) and echoes its absolute path. Build
# chatter goes to stderr so the echoed path stays clean for command substitution.
build_mlx() {
	local root="$1"
	local bin="$root/tests/cli/mlx/bin/mlx"
	( cd "$root" && CGO_ENABLED=1 go build -tags metal_runtime \
		-ldflags "-extldflags=-mmacosx-version-min=26.0" -o "$bin" ./cmd/mlx ) >&2 || return 1
	printf '%s\n' "$bin"
}

# metallib_path echoes the metallib the metal binary loads at runtime.
metallib_path() { printf '%s\n' "$1/../dist/lib/mlx.metallib"; }

# hf_model_path resolves an HF repo to its cached snapshot dir, mirroring
# internal/metaltest.HFModelPath. Returns non-zero when the model is not cached
# so the caller skips (a checkout without weights stays green, not red).
hf_model_path() {
	local repo="$1"
	local snaps="$HOME/.cache/huggingface/hub/models--${repo//\//--}/snapshots"
	[[ -d "$snaps" ]] || return 2
	local snap
	snap="$(find "$snaps" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
	[[ -n "$snap" ]] || return 2
	printf '%s\n' "$snap"
}
