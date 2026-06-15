#!/bin/sh
# scripts/cpp-kernel-coverage.sh — clang source-based coverage for go-mlx's OWN
# Metal-kernel bridges (go/pkg/metal/*_bridge.cpp), NOT the vendored MLX.
#
# Companion to scripts/cpp-coverage.sh, which measures the vendored native MLX
# under lib/mlx/mlx/. That script CANNOT see the go-mlx bridges: it reports
# against the lib/mlx `tests` binary (which never links go/pkg/metal) and its
# SRC_REGEX is hard-locked to lib/mlx/mlx/. The go-mlx custom kernels — the
# fused/compiled single-token decode (#65/#90/#91/#93) and the fused quantised
# lm-head + top-k (#95) — therefore showed 0% C++ coverage. This script closes
# that gap.
#
# It builds a SEPARATE test executable (tests/cpp, see tests/cpp/CMakeLists.txt)
# that compiles the bridge sources WITH coverage instrumentation and links them
# against the libmlx.a + metallib an existing MLX build already produced, runs
# the doctest suite under LLVM_PROFILE_FILE, then reports llvm-cov FILTERED to
# the bridge files only — so the number is OUR kernels, not the vendor.
#
# Usage (normally via `task cov:cpp:kernels`):
#   scripts/cpp-kernel-coverage.sh <kernel-build-dir> <mlx-build-dir>
# Defaults: build/cppkernels  build/cppcov
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
KBUILD=${1:-$ROOT/build/cppkernels}
MLXBUILD=${2:-$ROOT/build/cppcov}

TESTBIN="$KBUILD/gomlx_kernel_tests"
PROFDIR="$KBUILD/profraw"
MERGED="$KBUILD/kernel.profdata"
METALLIB="$MLXBUILD/mlx/backend/metal/kernels/mlx.metallib"

PROFDATA=$(xcrun --find llvm-profdata)
COV=$(xcrun --find llvm-cov)

if [ ! -x "$TESTBIN" ]; then
  echo "cpp-kernel-coverage: test binary not found at $TESTBIN — build it first (task cov:cpp:kernels)" >&2
  exit 1
fi
if [ ! -f "$METALLIB" ]; then
  echo "cpp-kernel-coverage: metallib not found at $METALLIB — build the MLX tree first (task cov:cpp)" >&2
  exit 1
fi

rm -rf "$PROFDIR"
mkdir -p "$PROFDIR"

echo "==> running go-mlx kernel test suite under coverage instrumentation"
set +e
MLX_METALLIB_PATH="$METALLIB" \
  LLVM_PROFILE_FILE="$PROFDIR/kernel-%p.profraw" \
  "$TESTBIN"
TESTS_RC=$?
set -e
echo "==> tests exit code: $TESTS_RC (expect 0)"
if [ "$TESTS_RC" -ne 0 ]; then
  echo "cpp-kernel-coverage: tests failed — fix before trusting coverage." >&2
  exit "$TESTS_RC"
fi

RAW_COUNT=$(find "$PROFDIR" -name '*.profraw' -size +0c | wc -l | tr -d ' ')
if [ "$RAW_COUNT" -eq 0 ]; then
  echo "cpp-kernel-coverage: no non-empty .profraw — coverage flags not in the build." >&2
  exit 1
fi

echo "==> merging profiles"
find "$PROFDIR" -name '*.profraw' -size +0c -print0 \
  | xargs -0 "$PROFDATA" merge --failure-mode=warn --sparse -o "$MERGED"

# Report ONLY the go-mlx bridge sources. Everything else linked into the binary
# (vendored mlx, mlx-c, fmt, doctest, the test .cpp themselves) is excluded so
# the number is unambiguously OUR custom Metal kernels.
IGNORE='(/tests/cpp/|doctest|/usr/|/Applications/|/Library/|/lib/mlx/|/lib/mlx-c/|/lib/fmt/|/_deps/|metal_cpp)'

echo
echo "============== go-mlx KERNEL BRIDGE COVERAGE (our parts only) =============="
echo "  scope: go/pkg/metal/*_bridge.cpp — the custom kernels go-mlx adds on MLX"
echo "  columns: Regions  Functions  Lines  Branches"
"$COV" report "$TESTBIN" -instr-profile="$MERGED" --show-region-summary \
  -ignore-filename-regex="$IGNORE" \
  | grep -E '^(Filename|-----|/Users|go/pkg/metal|TOTAL)' || true

echo
echo "merged profile: $MERGED"
echo "to drill into a bridge:  $COV show $TESTBIN -instr-profile=$MERGED go/pkg/metal/decode_bridge.cpp"
