# SPDX-Licence-Identifier: EUPL-1.2
#
# go-mlx native-kernel C++ test target.
#
# This builds a SEPARATE test executable for go-mlx's OWN Metal-kernel bridges
# (go/pkg/metal/*_bridge.cpp) — the custom fused kernels go-mlx layers on top of
# upstream MLX. It is deliberately NOT part of the vendored lib/mlx build:
#
#   * lib/mlx is a vendored submodule (ml-explore/mlx + a few lthn patches); its
#     CMake and tests/ are upstream's and must not be edited here.
#   * The bridges live under go/pkg/metal and are normally compiled by cgo, so
#     the lib/mlx `tests` binary never links them — `task cov:cpp` (scoped to
#     lib/mlx/mlx/) reports ZERO coverage for them. This target closes that gap.
#
# It reuses the artefacts the cov:cpp / test:cpp build already produces:
#   * an instrumented (or plain) libmlx.a + mlx.metallib under build/cppcov
#     (or build/cpptest),
#   * the fetched doctest + metal_cpp sources under that build's _deps,
#   * libmlxc.a from the cgo install tree (dist/lib).
#
# The bridge sources are compiled WITH coverage flags so llvm-cov can attribute
# region/line coverage to go/pkg/metal/*_bridge.cpp specifically (see
# scripts/cpp-kernel-coverage.sh).
#
# Driven by `task test:cpp:kernels` / `task cov:cpp:kernels`.

cmake_minimum_required(VERSION 3.24)
project(go-mlx-kernel-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Repo root (this file lives at <root>/tests/cpp).
get_filename_component(GOMLX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Which already-built MLX tree to link against. Defaults to the coverage build
# (build/cppcov); set -DGOMLX_MLX_BUILD=build/cpptest for the plain test build.
set(GOMLX_MLX_BUILD "build/cppcov" CACHE STRING
    "MLX build dir (relative to repo root) supplying libmlx.a + metallib + _deps")
set(MLX_BUILD_DIR "${GOMLX_ROOT}/${GOMLX_MLX_BUILD}")

# Instrument the bridge sources for coverage by default (the whole point).
option(GOMLX_KERNEL_COVERAGE "Build bridge sources with clang coverage" ON)

# --- locate the pieces the MLX build already produced -----------------------
find_library(MLX_LIB        NAMES mlx        PATHS "${MLX_BUILD_DIR}"           NO_DEFAULT_PATH REQUIRED)
find_library(GGUF_LIB       NAMES gguflib    PATHS "${MLX_BUILD_DIR}/mlx/io"    NO_DEFAULT_PATH REQUIRED)
find_library(MLXC_LIB       NAMES mlxc       PATHS "${GOMLX_ROOT}/dist/lib"     NO_DEFAULT_PATH REQUIRED)

set(DOCTEST_INC  "${MLX_BUILD_DIR}/_deps/doctest-src")
set(METAL_CPP_INC "${MLX_BUILD_DIR}/_deps/metal_cpp-src")
set(METALLIB_PATH "${MLX_BUILD_DIR}/mlx/backend/metal/kernels/mlx.metallib")

if(NOT EXISTS "${DOCTEST_INC}/doctest/doctest.h")
  message(FATAL_ERROR "doctest not found under ${DOCTEST_INC} — build ${GOMLX_MLX_BUILD} first (task cov:cpp or test:cpp)")
endif()

# --- the go-mlx bridges under test (OUR Metal kernels) ----------------------
# Only the go-mlx-authored kernel bridges; the mlx_*/mlxc_* shims are generated
# vendor glue and the gguf/detach/pinned bridges are not kernels.
set(GOMLX_BRIDGE_SOURCES
    "${GOMLX_ROOT}/go/pkg/metal/lm_head_topk_bridge.cpp"
    "${GOMLX_ROOT}/go/pkg/metal/decode_bridge.cpp"
    "${GOMLX_ROOT}/go/pkg/metal/activation_bridge.cpp")

# doctest auto-registers TEST_CASEs at static init, so adding a test .cpp here is
# all that is needed to run it — tests_main.cpp does not list them explicitly.
set(GOMLX_TEST_SOURCES
    tests_main.cpp
    lm_head_topk_bridge_tests.cpp
    decode_bridge_tests.cpp
    activation_bridge_tests.cpp)

add_executable(gomlx_kernel_tests ${GOMLX_TEST_SOURCES} ${GOMLX_BRIDGE_SOURCES})

target_include_directories(gomlx_kernel_tests PRIVATE
    "${GOMLX_ROOT}/lib/mlx"
    "${GOMLX_ROOT}/lib/mlx-c"
    "${GOMLX_ROOT}/lib/fmt/include"
    "${GOMLX_ROOT}/go/pkg/metal"
    "${METAL_CPP_INC}"
    "${DOCTEST_INC}")

target_compile_definitions(gomlx_kernel_tests PRIVATE
    MLX_STATIC
    ACCELERATE_NEW_LAPACK
    MLX_USE_ACCELERATE
    FMT_HEADER_ONLY)

# Coverage instrumentation on the bridge sources only (the test/doctest TUs
# don't need it — we measure OUR kernels, not the harness). Applying it to the
# whole target is simplest and the report is filtered to the bridge files
# anyway; the linker also needs the profile runtime.
if(GOMLX_KERNEL_COVERAGE)
  target_compile_options(gomlx_kernel_tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(gomlx_kernel_tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

# Static-archive link order: bridges (in the executable objects) reference mlxc
# + mlx; mlxc references mlx; mlx references gguf. List most-dependent first.
target_link_libraries(gomlx_kernel_tests PRIVATE
    "${MLXC_LIB}"
    "${MLX_LIB}"
    "${GGUF_LIB}"
    "-framework Foundation"
    "-framework Metal"
    "-framework Accelerate"
    "-framework QuartzCore")

# Bake the metallib path so the harness finds MLX's compiled kernels without an
# external env var (this is exactly the MLX_METALLIB_PATH override the lthn
# device.cpp patch added; we set it via the env at run time in the test command
# and also expose it here for `ctest`).
set_property(TARGET gomlx_kernel_tests PROPERTY GOMLX_METALLIB_PATH "${METALLIB_PATH}")

enable_testing()
add_test(NAME gomlx_kernel_tests COMMAND gomlx_kernel_tests)
set_tests_properties(gomlx_kernel_tests PROPERTIES
    ENVIRONMENT "MLX_METALLIB_PATH=${METALLIB_PATH}")
