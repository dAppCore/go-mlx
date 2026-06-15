#!/bin/sh
# scripts/cpp-coverage.sh — clang source-based coverage for the native MLX C++.
#
# Companion to scripts/coverage.sh (the Go analog). The Go script measures
# go/... ; this one measures the patched native sources under lib/mlx/mlx/.
#
# Driven by `task cov:cpp`, which first configures + builds build/cppcov with
# -fprofile-instr-generate -fcoverage-mapping baked into CMAKE_CXX_FLAGS (so
# every mlx-library translation unit the `tests` binary links is instrumented).
# This script then:
#   1. runs the `tests` (+ test_teardown) binaries DIRECTLY (not via ctest —
#      ctest does not propagate LLVM_PROFILE_FILE to the per-case children),
#      writing one .profraw per process, and excludes scheduler_tests.cpp so the
#      3 known stream-in-threads reds don't SIGABRT before clang flushes,
#   2. merges with `xcrun llvm-profdata merge --failure-mode=warn` (tolerant of
#      a truncated/empty .profraw; `any` would mean the opposite),
#   3. emits `xcrun llvm-cov report` (overall regions/functions/lines/branches %)
#      plus a per-file table sorted to surface the biggest uncovered native
#      sources under lib/mlx/mlx/.
#
# Usage (normally via `task cov:cpp`):
#   scripts/cpp-coverage.sh <build-dir>
# Env:
#   COVERAGE_TOP   how many biggest-uncovered files to list (default 30)
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD=${1:-$ROOT/build/cppcov}
TOP=${COVERAGE_TOP:-30}

PROFDIR="$BUILD/profraw"
MERGED="$BUILD/cpp.profdata"
TESTBIN="$BUILD/tests/tests"
TEARDOWNBIN="$BUILD/tests/test_teardown"

PROFDATA=$(xcrun --find llvm-profdata)
COV=$(xcrun --find llvm-cov)

# Restrict the report to the patched native sources we care about; the doctest
# headers, fetched deps and test .cpp files would otherwise drown the ranking.
SRC_REGEX="$ROOT/lib/mlx/mlx/"

if [ ! -x "$TESTBIN" ]; then
  echo "cpp-coverage: test binary not found at $TESTBIN — build build/cppcov first (task cov:cpp)" >&2
  exit 1
fi

rm -rf "$PROFDIR"
mkdir -p "$PROFDIR"

# We do NOT drive coverage through ctest. ctest re-invokes the `tests` binary
# once per discovered case but does not propagate LLVM_PROFILE_FILE to those
# children, so a ctest run yields zero usable .profraw. Worse, three scheduler
# stream-in-threads cases abort the process (SIGABRT) — and a clang profile is
# flushed at exit, so an aborted run flushes nothing. Instead we invoke the test
# binaries DIRECTLY and exclude the aborting source so `tests` exits cleanly and
# flushes its full profile.
#
# Excluded: tests/scheduler_tests.cpp — the 3 known reds (ctest #221 "default
# stream in threads", #224 "thread local stream", #245 the aggregate that
# inherits the abort). They are an intentional-divergence patch family
# (per-thread vs process-canonical default stream); see docs/cpp-test-status.md.
# Coverage of scheduler.cpp itself still comes from every other test that
# schedules work — only the threads-specific cases are skipped.
EXCLUDE_SRC='*scheduler_tests*'
echo "==> running C++ test suite directly under coverage instrumentation"
echo "    (excluding $EXCLUDE_SRC — the 3 aborting stream-in-threads reds; see docs/cpp-test-status.md)"
set +e
LLVM_PROFILE_FILE="$PROFDIR/tests-%p.profraw" \
  "$TESTBIN" --source-file-exclude="$EXCLUDE_SRC"
TESTS_RC=$?
if [ -x "$TEARDOWNBIN" ]; then
  LLVM_PROFILE_FILE="$PROFDIR/teardown-%p.profraw" "$TEARDOWNBIN" >/dev/null 2>&1
fi
set -e
echo "==> tests exit code: $TESTS_RC (expect 0 with the reds excluded)"

RAW_COUNT=$(find "$PROFDIR" -name '*.profraw' -size +0c | wc -l | tr -d ' ')
echo "==> collected $RAW_COUNT non-empty .profraw files"
if [ "$RAW_COUNT" -eq 0 ]; then
  echo "cpp-coverage: no non-empty .profraw produced — coverage flags not in the build, or every run aborted before flush." >&2
  exit 1
fi

# --failure-mode=warn: do NOT fail the merge on a bad/truncated profile (note:
# `any` means the OPPOSITE — fail if any profile is invalid). A 0-byte .profraw
# from an aborted process is skipped with a warning instead of aborting.
echo "==> merging profiles (llvm-profdata merge --failure-mode=warn)"
find "$PROFDIR" -name '*.profraw' -size +0c -print0 \
  | xargs -0 "$PROFDATA" merge --failure-mode=warn --sparse -o "$MERGED"
if [ ! -s "$MERGED" ]; then
  echo "cpp-coverage: merged profile $MERGED is empty — merge failed." >&2
  exit 1
fi

# Report against the `tests` binary: as a statically-linked executable it carries
# the coverage records for every mlx TU it references, so the test-exercised
# slice of lib/mlx/mlx/ surfaces. CAVEAT (measured, not theoretical): mlx is a
# static archive (libmlx.a) — a .cpp no test references is dropped by the linker
# and is ABSENT here, not shown as 0%. On this Metal build that absent set is
# ~51/163 native .cpp and is almost entirely INACTIVE backends: backend/cuda/*,
# the no_cpu/no_gpu/no_metal/no_gguf/no_safetensors disabled-stub TUs, and the
# distributed mpi/nccl/ring/jaccl transports — i.e. code not compiled into this
# config, not untested Metal/CPU sources. We report against the binary (clean,
# no hash-mismatch noise) and print the linked/total count below so the scope is
# explicit. To force an unreferenced TU into the report, add its .o as -object;
# mixing libmlx .o files with the binary profile emits "mismatched data"
# warnings (template/inline hash drift), so we don't do it by default.
OBJ_ARGS="-object $TESTBIN"

# Keep ONLY the native mlx sources (lib/mlx/mlx/...). Coverage records store a
# mix of absolute (/.../lib/mlx/mlx/...) and build-relative (build/.../mlx/...)
# paths, so we EXCLUDE everything that is not native rather than positionally
# filter a directory (a positional dir makes llvm-cov try to load it as a
# profile → "Is a directory"). Excludes: doctest/fetched deps, the 3rdparty
# vendored headers (pocketfft), the test and benchmark .cpp, the toolchain/SDK
# headers, and lib/mlx top-level non-mlx dirs.
IGNORE='(/tests/|/benchmarks/|/_deps/|doctest|/3rdparty/|/lib/mlx/[^m/]|/usr/|/Applications/|/Library/)'

# Quantify the static-archive scope: how many native .cpp are linked (measured)
# vs how many exist on disk.
LINKED_CPP=$("$COV" report $OBJ_ARGS -instr-profile="$MERGED" \
  -ignore-filename-regex="$IGNORE" 2>/dev/null \
  | grep -oE '[A-Za-z0-9_/]+\.cpp' | sed 's#.*/mlx/#mlx/#' | sort -u | wc -l | tr -d ' ')
DISK_CPP=$(find "$ROOT/lib/mlx/mlx" -name '*.cpp' | wc -l | tr -d ' ')

echo
echo "==================== OVERALL NATIVE C++ COVERAGE ===================="
echo "  scope: lib/mlx/mlx/ (patched native sources), test-linked surface only"
echo "  linked native .cpp: $LINKED_CPP of $DISK_CPP on disk"
echo "    (the rest = inactive backends: cuda/*, no_* stubs, distributed transports)"
echo "  columns: Regions  Functions  Lines  Branches"
# No 2>/dev/null here — a load/merge error must be visible, not read as 0%.
# shellcheck disable=SC2086
"$COV" report $OBJ_ARGS -instr-profile="$MERGED" --show-region-summary \
  -ignore-filename-regex="$IGNORE" \
  | grep -E '^(Filename|-----|TOTAL)'

echo
echo "============ BIGGEST UNCOVERED NATIVE SOURCES (top $TOP) ============"
echo "  (regions = branch/expression coverage; ranked by uncovered-region count)"
# With --show-region-summary, llvm-cov report emits one row per file as fixed
# FRONT-anchored columns:
#   $1 file
#   $2 RegTotal  $3 RegMissed  $4 Reg%
#   $5 FnTotal   $6 FnMissed   $7 Fn%(Executed)
#   $8 LineTotal $9 LineMissed $10 Line%
#   $11 BrTotal  $12 BrMissed  $13 Br%   (may be "-" when no branches)
# Rank by uncovered regions ($3). File paths come abs or build-relative; we trim
# both prefixes to a stable lib/mlx/mlx/... display.
# shellcheck disable=SC2086
"$COV" report $OBJ_ARGS -instr-profile="$MERGED" --show-region-summary \
  -ignore-filename-regex="$IGNORE" 2>/dev/null \
  | awk -v root="$ROOT/" '
      $1 ~ /\.(cpp|h|cc|hpp|cuh)$/ {
        f=$1
        sub(root, "", f)               # strip /Users/.../go-mlx/ if absolute
        sub(/^build\/[^\/]+\//, "", f) # strip build/cppcov/ if build-relative
        printf "%8d uncov-reg  | reg %7s | fns %7s | lines %7s  %s\n", $3+0, $4, $7, $10, f
      }' \
  | sort -rn \
  | head -n "$TOP"

echo
echo "merged profile: $MERGED"
echo "to drill into one file:  $COV show $TESTBIN -instr-profile=$MERGED <path>"
