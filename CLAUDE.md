# CLAUDE.md

## What This Is

Native Apple Metal GPU inference via mlx-c bindings. Module: `forge.lthn.ai/core/go-mlx`

Implements the `inference.Backend` interface from [`forge.lthn.ai/core/go-inference`](https://forge.lthn.ai/core/go-inference) for Apple Silicon (M1-M4) GPUs using Metal compute shaders via the [mlx-c](https://github.com/ml-explore/mlx-c) C API.

## Platform

**darwin/arm64 only.** All CGO files carry `//go:build darwin && arm64`. A stub (`mlx_stub.go`) provides `MetalAvailable() bool` returning false on other platforms.

## Build

```bash
# Step 1: Build mlx-c C library via CMake (fetches mlx-c v0.4.1)
go generate ./...

# Step 2: Run tests (must be on Apple Silicon)
go test ./...
```

### CGO Flags (auto-set via go:generate)

CMake installs to `dist/` inside the package directory. The `#cgo` directives in `internal/metal/metal.go` reference:
- `CPPFLAGS: -I${SRCDIR}/../../dist/include`
- `LDFLAGS: -L${SRCDIR}/../../dist/lib -lmlxc -lmlx`
- Frameworks: Foundation, Metal, Accelerate

### CMake Config

`CMakeLists.txt` fetches mlx-c v0.4.1 from GitHub. Key settings:
- `MLX_BUILD_SAFETENSORS=ON` (model loading)
- `MLX_BUILD_GGUF=OFF`
- `BUILD_SHARED_LIBS=ON`
- macOS deployment target: 26.0

## Architecture

```
go-mlx/
├── Public API (compiles on all platforms)
│   ├── mlx.go              — Package doc + go:generate CMake directives
│   ├── register_metal.go   — //go:build darwin && arm64 — auto-registers metal backend
│   ├── mlx_stub.go         — //go:build !darwin || !arm64 — MetalAvailable() false
│   └── mlx_test.go         — Integration tests (public API via go-inference)
│
├── internal/metal/          — All CGO code (darwin/arm64 only)
│   ├── metal.go            — Init, Materialize, error handler, stream
│   ├── array.go            — Array type, creation, data access
│   ├── dtype.go            — DType constants
│   ├── stream.go           — Metal stream/queue, memory controls
│   ├── ops.go              — Element-wise, reduction, shape ops
│   ├── fast.go             — Fused Metal kernels (RMSNorm, RoPE, SDPA)
│   ├── nn.go               — Linear, Embedding, RMSNormModule
│   ├── compile.go          — CompiledFunc
│   ├── slice.go            — Array slicing
│   ├── random.go           — RandomCategorical, RandomUniform
│   ├── io.go               — Safetensors loading
│   ├── model.go            — InternalModel interface + architecture dispatch
│   ├── gemma3.go           — Gemma3 decoder
│   ├── qwen3.go            — Qwen3 decoder
│   ├── cache.go            — KVCache + RotatingKVCache
│   ├── sample.go           — Sampling chain (greedy, temp, topK, topP)
│   ├── tokenizer.go        — BPE tokenizer
│   ├── grad.go             — VJP
│   ├── lora.go             — LoRA adapters
│   ├── optim.go            — AdamW
│   ├── generate.go         — Autoregressive generation loop + chat templates
│   └── backend.go          — LoadAndInit entry point
│
├── cpp/                     — CLion Claude workspace (C++ research)
│   ├── CMakeLists.txt
│   ├── CLAUDE.md
│   ├── TODO.md
│   └── FINDINGS.md
│
└── docs/plans/              — Design and implementation docs
```

## Usage

```go
import (
    "forge.lthn.ai/core/go-inference"
    _ "forge.lthn.ai/core/go-mlx" // register Metal backend
)

// Load and generate — types come from go-inference
m, err := inference.LoadModel("/path/to/model/")
if err != nil { log.Fatal(err) }
defer m.Close()

ctx := context.Background()
for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
if err := m.Err(); err != nil { log.Fatal(err) }

// Chat with template formatting
for tok := range m.Chat(ctx, []inference.Message{
    {Role: "user", Content: "Hello"},
}, inference.WithMaxTokens(64)) {
    fmt.Print(tok.Text)
}

// Metal-specific memory controls (from mlx package directly)
mlx.SetCacheLimit(4 * 1024 * 1024 * 1024) // 4GB
mlx.ClearCache()
```

## Dependencies

- **[`forge.lthn.ai/core/go-inference`](https://forge.lthn.ai/core/go-inference)** — shared Backend/TextModel interfaces
- **mlx-c v0.4.1** (fetched by CMake at build time)
- **Apple frameworks**: Foundation, Metal, Accelerate

## Downstream Consumers

- `forge.lthn.ai/core/go-ml` — imports go-inference + go-mlx for Metal backend
- `forge.lthn.ai/core/go-i18n` — Phase 2a needs Gemma3-1B inference for domain classification
- `forge.lthn.ai/core/go-rocm` — sibling backend for AMD GPUs, same go-inference interfaces

Import `_ "forge.lthn.ai/core/go-mlx"` to register the Metal backend. All types (`TextModel`, `Token`, `Backend`, options) come from `go-inference`.

## Model Format

**Safetensors** (HuggingFace format). NOT GGUF.
- Example: `/Volumes/Data/lem/safetensors/gemma-3/`
- Models must be in safetensors format with matching tokenizer config

## Coding Standards

- UK English (colour, organisation, centre)
- `go test ./...` must pass before commit
- Conventional commits: `type(scope): description`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Task Queue

See `TODO.md` for prioritised work.
See `FINDINGS.md` for research notes.
See the [wiki](https://forge.lthn.ai/core/go-mlx/wiki) for architecture docs.
