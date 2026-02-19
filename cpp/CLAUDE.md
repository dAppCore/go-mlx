# CLAUDE.md — go-mlx C++ / mlx-c Side

## What This Is

You are the C++ specialist for `forge.lthn.ai/core/go-mlx`. This Go package wraps Apple's MLX framework through the **mlx-c** C API using CGO. You handle C/C++ work; a separate GoLand Claude handles the Go bindings.

## Your Role

- Inspect and understand the mlx-c API surface (headers, source)
- When Virgil (the core/go orchestration agent) or the GoLand Claude needs a C-level change, you implement it
- Debug C-level issues (segfaults, memory leaks, API mismatches)
- Research mlx-c capabilities for new Go bindings
- You do NOT modify Go files — relay Go-side changes to the GoLand Claude

## Project Layout

```
go-mlx/                          ← CMakeLists.txt lives here (CLion project root)
├── CMakeLists.txt               ← Fetches mlx-c v0.4.1, builds to dist/
├── cpp/                         ← YOUR workspace docs (this directory)
│   ├── CLAUDE.md                ← This file
│   ├── TODO.md                  ← C++ task queue
│   └── FINDINGS.md              ← C++ research notes
│
├── build/                       ← CMake build directory (gitignored)
│   └── _deps/
│       ├── mlx-c-src/           ← mlx-c v0.4.1 source (24 .cpp, 27 .h)
│       │   └── mlx/c/           ← The C API headers + implementations
│       └── mlx-src/             ← MLX C++ framework source (Metal shaders, ops)
│
├── dist/                        ← Installed artifacts (gitignored)
│   ├── include/mlx/c/           ← Headers used by Go CGO
│   └── lib/                     ← libmlxc.dylib, libmlx.dylib
│
└── *.go                         ← Go bindings (GoLand Claude's domain)
```

## Key Files

### mlx-c Headers (the API surface Go binds to)
- `build/_deps/mlx-c-src/mlx/c/array.h` — Array creation, data access, shape
- `build/_deps/mlx-c-src/mlx/c/ops.h` — Element-wise and reduction operations
- `build/_deps/mlx-c-src/mlx/c/fast.h` — Fused Metal kernels (RMSNorm, RoPE, SDPA)
- `build/_deps/mlx-c-src/mlx/c/transforms.h` — eval, compile, VJP/JVP
- `build/_deps/mlx-c-src/mlx/c/io.h` — Safetensors load/save
- `build/_deps/mlx-c-src/mlx/c/random.h` — Random number generation
- `build/_deps/mlx-c-src/mlx/c/stream.h` — Metal stream/queue
- `build/_deps/mlx-c-src/mlx/c/metal.h` — Metal device availability
- `build/_deps/mlx-c-src/mlx/c/error.h` — Error handler registration
- `build/_deps/mlx-c-src/mlx/c/memory.h` — Memory management

### CMake
- `go-mlx/CMakeLists.txt` — Top-level, fetches mlx-c v0.4.1 from GitHub
- `build/_deps/mlx-c-src/CMakeLists.txt` — mlx-c's own build config

## Build

```bash
# From go-mlx root:
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build

# Or via Go:
go generate ./...
```

### CMake Settings
- `MLX_BUILD_SAFETENSORS=ON`
- `MLX_BUILD_GGUF=OFF`
- `BUILD_SHARED_LIBS=ON`
- macOS deployment target: 26.0

## How Go Binds to mlx-c

Go files use CGO with `#include "mlx/c/mlx.h"` and link via:
```
#cgo CPPFLAGS: -I${SRCDIR}/dist/include
#cgo LDFLAGS: -L${SRCDIR}/dist/lib -lmlxc -lmlx
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
```

Each Go function calls the corresponding `mlx_*` C function. The pattern is:
1. Go allocates a C output handle (`mlx_array_new()`)
2. Calls the C operation (`mlx_add()`, `mlx_matmul()`, etc.)
3. Wraps result in Go `*Array` with `runtime.SetFinalizer` → `mlx_array_free()`

## Communication Protocol

- **Receiving work**: Tasks appear in `cpp/TODO.md`, written by the GoLand Claude or Virgil
- **Reporting findings**: Write to `cpp/FINDINGS.md`
- **Requesting Go changes**: Describe what the GoLand Claude needs to change in FINDINGS.md
- **Completing tasks**: Mark `[x]` in TODO.md with notes

## Platform

- **darwin/arm64 only** — Apple Silicon (M1-M4)
- Tested on: Mac Studio M3 Ultra (32-core CPU, 60-core GPU, 96GB)
- Xcode Command Line Tools required
- CMake 3.24+

## Coding Standards

- UK English in documentation
- Conventional commits: `type(scope): description`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2
