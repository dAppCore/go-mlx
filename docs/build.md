---
title: Build Guide
description: Build requirements, CMake setup, build tags, and testing for go-mlx.
---

# Build Guide

go-mlx requires CGO and Apple's Metal framework. All CGO source files carry `//go:build darwin && arm64` -- the package compiles on other platforms but provides only a `MetalAvailable() bool` stub returning false.

## Prerequisites

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| macOS | 26.0+ on Apple Silicon (M1+) | Metal 4 GPU compute |
| Go | 1.26.0+ | Module toolchain |
| CMake | 3.24+ | Builds mlx-c from source |
| AppleClang | 17.0+ | C/C++ compiler for mlx-c |
| macOS SDK | 26.2+ | Metal framework headers |
| Xcode Command Line Tools | Current | Provides `xcrun`, frameworks |

Install CMake if absent:

```bash
brew install cmake
```

## Build Steps

### Step 1: Build mlx-c

From the module root:

```bash
git submodule update --init --recursive
go generate ./...
```

This executes the `//go:generate` directives in `mlx.go`:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build
```

The submodule initialisation is required because `internal/metal/` contains
forwarding translation units that include sources from `lib/mlx`, `lib/mlx-c`,
and `lib/generated`.

CMake fetches mlx-c v0.6.0 from GitHub and builds it against the local
patched `lib/mlx` submodule with:

- `MLX_BUILD_SAFETENSORS=ON` -- required for model loading
- `MLX_BUILD_GGUF=ON` -- enables GGUF load/save support
- `BUILD_SHARED_LIBS=OFF` -- static archives only (cgo doesn't link these; see below)
- `CMAKE_OSX_DEPLOYMENT_TARGET=26.0`

Headers install to `dist/include/`, the precompiled Metal shader library lands at `dist/lib/mlx.metallib`. The MLX C++ implementation is vendored in-tree at `go/internal/metal/` (187 `mlx_*.cpp` files) and cgo compiles it inline — the CMake-side static archives are configuration scaffolding, not runtime link artefacts. Build time is approximately 2 minutes on M3 Ultra.

The `dist/` directory is gitignored and must be rebuilt on each fresh checkout.

### Step 2: Run Tests

```bash
go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./...
```

Tests that require model files on disk (e.g. `/Volumes/Data/lem/safetensors/...`) are skipped automatically when the paths are absent. CI runs without model files.

## CGO Flags

The `#cgo` directives in `internal/metal/metal.go` set all required flags automatically:

```c
#cgo CXXFLAGS: -std=gnu++23 -mmacosx-version-min=26.0 -O2 -DNDEBUG ...
#cgo CFLAGS: -mmacosx-version-min=26.0
#cgo darwin CFLAGS: -x objective-c
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/mlx -I${SRCDIR}/../../../lib/mlx-c
#cgo CPPFLAGS: -I${SRCDIR}/../../../dist/include
#cgo darwin LDFLAGS: -mmacosx-version-min=26.0 -framework Foundation -framework Metal -framework Accelerate -framework QuartzCore
```

`${SRCDIR}` is the directory containing `metal.go` at build time (`internal/metal/`). The full file at `go/internal/metal/metal.go` has the complete set. Notably absent: any `-L` or `-l` for libmlx/libmlxc — the implementation `.cpp` files sit alongside `metal.go` and cgo picks them up directly.

The final Go executable/test link also needs the macOS 26.0 floor because the
native path is aligned to the Metal 4 API generation shipped with macOS Tahoe
26. Apple's Metal 4 pages document the API family used for lower-overhead
command encoding, explicit compilation, native tensors, and machine-learning
passes; the macOS 26 release notes are the operating-system boundary for that
Metal 4 support. The canonical Taskfile passes this automatically:

```bash
task build:lthn
task test
```

When invoking Go directly, pass the same external linker floor:

```bash
go build -trimpath -ldflags "-extldflags=-mmacosx-version-min=26.0" -o ../bin/lthn-mlx ./cmd/mlx
go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./...
```

Reference links:

- [macOS Tahoe 26 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes)
- [What's new in macOS 26](https://developer.apple.com/macos/whats-new/)
- [What's new in Metal](https://developer.apple.com/metal/whats-new/)
- [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api)
- [Metal machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes)

## Build Tags

| Tag | File | Effect |
|-----|------|--------|
| `darwin && arm64` | `register_metal.go`, all `internal/metal/*.go` | Enables native Metal backend |
| `!(darwin && arm64)` | `mlx_stub.go` | Provides `MetalAvailable() = false` |
| `!nomlxlm` | `mlxlm/backend.go` | Includes the mlx-lm subprocess backend (default) |
| `nomlxlm` | -- | Excludes the mlxlm subprocess backend |

To build without the subprocess backend:

```bash
go build -tags nomlxlm ./...
```

## Go Workspace

go-mlx participates in a Go workspace alongside go-inference and other forge modules. The workspace file (`~/Code/go.work`) overrides version resolution for local development.

After adding modules or changing dependencies:

```bash
go work sync
```

## Clean Rebuild

To force a complete rebuild of mlx-c:

```bash
rm -rf build dist
go generate ./...
```

## CMake Configuration

`CMakeLists.txt` at the module root:

```cmake
cmake_minimum_required(VERSION 3.24)
project(mlx)

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
set(MLX_BUILD_GGUF ON CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_RPATH "@loader_path")

include(FetchContent)
set(MLX_C_GIT_TAG "v0.6.0" CACHE STRING "")
set(FETCHCONTENT_SOURCE_DIR_MLX "${CMAKE_CURRENT_SOURCE_DIR}/lib/mlx" CACHE PATH "Local patched MLX source")
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG ${MLX_C_GIT_TAG}
)
FetchContent_MakeAvailable(mlx-c)
```

The `CMAKE_INSTALL_RPATH` of `@loader_path` is legacy from when the CMake build produced shared libraries that cgo linked against; with `BUILD_SHARED_LIBS=OFF` and cgo compiling the C++ tree inline, the rpath setting is inert. It is retained for future contributors who may use the standalone `cpp/` CLion build that still links against the static archives.

## Testing

### Running All Tests

```bash
go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./...
```

### Running a Single Test

```bash
go test -run TestRMSNorm_Good ./internal/metal/
```

### Running with Race Detector

```bash
go test -race ./...
```

### Test Naming Convention

Tests use the `_Good`, `_Bad`, `_Ugly` suffix pattern:

| Suffix | Meaning |
|--------|---------|
| `_Good` | Happy path -- expected to succeed |
| `_Bad` | Expected error conditions |
| `_Ugly` | Panic and edge cases |

### Model-Dependent Tests

Integration tests that load real models use `t.Skip()` when the model path is absent:

```go
func gemma3ModelPath(t *testing.T) string {
    paths := []string{
        "/Volumes/Data/lem/gemma-3-1b-it-base",
        "/Volumes/Data/lem/safetensors/gemma-3/",
    }
    for _, p := range paths {
        if _, err := os.Stat(p); err == nil {
            return p
        }
    }
    t.Skip("no Gemma3 model available")
    return ""
}
```

These tests run locally when models are present but are safely skipped in CI.

### mlxlm Backend Tests

The `mlxlm/` package has no CGO dependency. Tests use `testdata/mock_bridge.py` instead of the real bridge, so no `mlx-lm` installation is required:

```bash
go test ./mlxlm/
```

## Benchmarks

29 benchmarks in `internal/metal/bench_test.go`:

```bash
go test -bench=. -benchtime=2s ./internal/metal/
```

| Benchmark Group | What It Measures |
|----------------|-----------------|
| `BenchmarkMatmul_*` | Matrix multiply at 128^2 through 4096^2 |
| `BenchmarkSoftmax_*` | Softmax at 1K through 128K vocab |
| `BenchmarkElementWise_*` | Add, Mul, SiLU at 1M elements |
| `BenchmarkRMSNorm_*` | Fused RMSNorm at decode and prefill shapes |
| `BenchmarkRoPE_*` | RoPE at single-token and 512-token shapes |
| `BenchmarkSDPA_*` | Scaled dot-product attention at 1, 32, 512 seq lengths |
| `BenchmarkLinear_*` | Linear layer forward at decode and prefill shapes |
| `BenchmarkSampler_*` | Greedy, TopK, TopP, and full chain on 32K vocab |

CGO call overhead floors at approximately 170 us per operation (Metal command buffer + CGO boundary). MatMul scales well: 128^2 to 4096^2 is roughly 55x slower for 1024x more work.

## Dependency Graph

```
go-mlx
+-- forge.lthn.ai/core/go-inference  (shared interfaces, zero dependencies)
+-- mlx-c v0.6.0                     (CMake, fetched at go generate time)
    +-- Apple MLX v0.31.1             (local patched lib/mlx submodule)
        +-- Foundation, Metal, Accelerate frameworks
```

The root package and `mlxlm/` have no CGO dependency. Only `internal/metal/` links against mlx-c.

## Coding Standards

- **UK English** throughout: colour, organisation, centre, initialise
- **EUPL-1.2 licence** -- every new file must carry `// SPDX-Licence-Identifier: EUPL-1.2`
- **Conventional commits**: `type(scope): description` (scopes: metal, api, mlxlm, cpp, docs)
- **Tests must pass**: `go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./...` before every commit
- **Co-Author**: `Co-Authored-By: Virgil <virgil@lethean.io>`
