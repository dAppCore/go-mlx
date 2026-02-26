# Development Guide

Module: `forge.lthn.ai/core/go-mlx`

---

## Prerequisites

### Platform

**macOS on Apple Silicon only.** All CGO source files carry `//go:build darwin && arm64`. The package will not build for native Metal inference on any other platform; a stub (`mlx_stub.go`) provides `MetalAvailable() bool` returning false elsewhere.

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Go | 1.25.5+ | Module toolchain |
| CMake | 3.24+ | Builds mlx-c from source |
| AppleClang | 17.0+ | C/C++ compiler for mlx-c |
| macOS SDK | 26.2+ | Metal framework headers |
| Xcode Command Line Tools | Current | Provides `xcrun`, frameworks |

Install CMake if absent:

```bash
brew install cmake
```

### Go Workspace

go-mlx participates in a Go workspace alongside go-inference. The `go.mod` uses a `replace` directive for local development:

```
replace forge.lthn.ai/core/go-inference => ../go-inference
```

After adding modules or changing dependencies: `go work sync`

---

## Build

### Step 1: Build mlx-c

Run from the module root:

```bash
go generate ./...
```

This executes the `//go:generate` directives in `mlx.go`:

```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build
```

CMake fetches mlx-c v0.4.1 from GitHub, builds it with:
- `MLX_BUILD_SAFETENSORS=ON` (model loading)
- `MLX_BUILD_GGUF=OFF`
- `BUILD_SHARED_LIBS=ON`
- macOS deployment target: 13.3 (minimum required by MLX)

The built library installs to `dist/include/` and `dist/lib/`. Build time is approximately 2 minutes on M3 Ultra.

The `dist/` directory is gitignored and must be rebuilt on each fresh checkout.

### Step 2: Run Tests

```bash
go test ./...
```

Tests require a working mlx-c build. Integration tests that load model files are skipped automatically when model paths are absent (`/Volumes/Data/lem/safetensors/...`).

---

## CGO Flags

The `#cgo` directives in `internal/metal/metal.go` set all required flags automatically when building on darwin/arm64:

```c
#cgo CXXFLAGS: -std=c++17
#cgo CFLAGS: -mmacosx-version-min=26.0
#cgo CPPFLAGS: -I${SRCDIR}/../../dist/include
#cgo LDFLAGS: -L${SRCDIR}/../../dist/lib -lmlxc -lmlx
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../../dist/lib
```

`${SRCDIR}` is the directory containing `metal.go` at build time (`internal/metal/`), so the `../../dist/` path resolves to the module root `dist/`.

No manual environment variables are needed for `go build` or `go test`.

---

## Test Patterns

Tests use the `_Good`, `_Bad`, `_Ugly` suffix convention:

| Suffix | Meaning |
|--------|---------|
| `_Good` | Happy path; expected to succeed |
| `_Bad` | Expected error conditions |
| `_Ugly` | Panic / edge cases |

Example:

```go
func TestMatmul_Good(t *testing.T) { ... }
func TestMatmul_Bad(t *testing.T) { ... }
```

Tests that require model files on disk use `t.Skip()` when the path is absent:

```go
const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"
if _, err := os.Stat(modelPath); err != nil {
    t.Skip("model not available:", modelPath)
}
```

All 180+ tests in `internal/metal/` are unit or integration tests that exercise the CGO layer directly. The 11 tests in the root package (`mlx_test.go`) exercise the public API via go-inference.

### Running a Single Test

```bash
go test -run TestRMSNorm_Good ./internal/metal/
```

### Running with Race Detector

```bash
go test -race ./...
```

---

## Benchmarks

29 benchmarks in `internal/metal/bench_test.go`. Run with:

```bash
go test -bench=. -benchtime=2s ./internal/metal/
```

Key benchmarks:

| Benchmark group | What it measures |
|----------------|-----------------|
| `BenchmarkMatmul_*` | Matrix multiply at 128² through 4096², plus token projection |
| `BenchmarkSoftmax_*` | Softmax at 1K through 128K vocab |
| `BenchmarkElementWise_*` | Add, Mul, SiLU at 1M elements |
| `BenchmarkRMSNorm_*` | Fused RMSNorm at decode and prefill shapes |
| `BenchmarkRoPE_*` | RoPE at single-token and 512-token shapes |
| `BenchmarkSDPA_*` | Scaled dot-product attention at 1, 32, 512 sequence lengths |
| `BenchmarkLinear_*` | Linear layer forward at decode and prefill shapes |
| `BenchmarkSampler_*` | Greedy, TopK, TopP, and full chain on 32K vocab |

Model-level benchmarks (`model.Forward`, tokenizer) require model files on disk and are not included in the automated suite.

---

## Code Structure

### Adding a New Operation

1. Add the C binding to the appropriate file in `internal/metal/`:
   - `ops.go` — element-wise, reduction, matrix, shape operations
   - `fast.go` — fused Metal kernel wrappers
   - `slice.go` — slicing and scatter operations
2. Follow the `newArray("OP_NAME", inputs...)` pattern for tracking
3. Add tests in the corresponding `_test.go` file using `_Good`/`_Bad` suffixes
4. Add a benchmark in `bench_test.go` for any operation on the hot path

### Adding a New Model Architecture

1. Read `config.json` `model_type` and add a case in `model.go`:`loadModel`
2. Create `architecture.go` in `internal/metal/` implementing `InternalModel`
3. Add `ApplyLoRA` to the new model
4. Add a `close*` helper in `close.go` for deterministic resource cleanup
5. Add `formatXyzChat` in `generate.go` for the chat template
6. Add tokeniser BOS/EOS detection in `tokenizer.go`:`LoadTokenizer`
7. Write tests: config parsing, missing weights, end-to-end inference

---

## Coding Standards

### Language

UK English throughout: colour, organisation, centre, initialise, behaviour. Never American spellings.

### Go Style

- `declare(strict_types=1)` equivalent: all parameters and return types must be explicitly typed
- PSR-12 equivalent: `gofmt` + `goimports`; run before committing
- `go test ./...` must pass before every commit; no red tests in main

### Licence Header

Every new source file must carry the EUPL-1.2 licence identifier:

```go
// SPDX-Licence-Identifier: EUPL-1.2
```

### Conventional Commits

Format: `type(scope): description`

Types:
- `feat` — new capability
- `fix` — bug fix
- `test` — test additions or changes
- `bench` — benchmark additions or changes
- `refactor` — code restructuring without behaviour change
- `docs` — documentation only
- `chore` — maintenance (gitignore, go.mod, CMake)

Scopes: `metal`, `api`, `mlxlm`, `cpp`, `docs`

Examples:
```
feat(metal): add TopP nucleus sampling
fix(metal): auto-contiguous data access for non-contiguous arrays
test(metal): add model loading robustness tests
bench(metal): add 29 benchmarks baselined on M3 Ultra
```

### Co-Author

All commits must include:

```
Co-Authored-By: Virgil <virgil@lethean.io>
```

### Build Tags

- All CGO files: `//go:build darwin && arm64`
- Stub file: `//go:build !darwin || !arm64`
- mlxlm opt-out: `//go:build !nomlxlm`

---

## CMake Configuration

`CMakeLists.txt` at the module root. Key settings:

```cmake
set(MLX_BUILD_SAFETENSORS ON)   # Required for model loading
set(MLX_BUILD_GGUF OFF)         # GGUF not supported
set(BUILD_SHARED_LIBS ON)       # Shared .dylib for rpath loading
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.3)  # MLX minimum
```

To force a clean rebuild:

```bash
rm -rf build dist
go generate ./...
```

---

## mlxlm Backend Development

The `mlxlm/` package has no CGO dependency and tests run on any platform where Python 3 is available. Tests use `testdata/mock_bridge.py` instead of the real `bridge.py`, so no `mlx-lm` installation is required.

Run mlxlm tests:

```bash
go test ./mlxlm/
```

The mock bridge responds to all commands with fixed fake data, enabling full subprocess protocol testing without GPU or Python ML dependencies.

To opt out of building the mlxlm backend:

```bash
go build -tags nomlxlm ./...
```

---

## Dependency Graph

```
go-mlx
├── forge.lthn.ai/core/go-inference  (shared interfaces, zero dependencies)
└── mlx-c v0.4.1                     (CMake, fetched from GitHub at generate time)
    └── Apple MLX (Metal GPU compute)
        └── Foundation, Metal, Accelerate frameworks
```

The root package and `mlxlm/` have no CGO dependency. Only `internal/metal/` links against mlx-c.
