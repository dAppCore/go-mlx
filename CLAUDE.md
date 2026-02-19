# CLAUDE.md

## What This Is

Native Apple Metal GPU inference via mlx-c bindings. Module: `forge.lthn.ai/core/go-mlx`

Pure Go + CGO package that wraps Apple's [MLX framework](https://github.com/ml-explore/mlx) through the [mlx-c](https://github.com/ml-explore/mlx-c) C API. Runs LLM inference on Apple Silicon GPUs (M1-M4) using Metal compute shaders.

## Platform

**darwin/arm64 only.** All files carry `//go:build darwin && arm64`. A stub (`mlx_stub.go`) provides `MetalAvailable() bool` returning false on other platforms.

## Build

```bash
# Step 1: Build mlx-c C library via CMake (fetches mlx-c v0.4.1)
cd /tmp/core-go-mlx && go generate ./...

# Step 2: Run tests (must be on Apple Silicon)
go test ./...

# Step 3: Use in another module
# Import "forge.lthn.ai/core/go-mlx" and "forge.lthn.ai/core/go-mlx/model"
```

### CGO Flags (auto-set via go:generate)

CMake installs to `dist/` inside the package directory. The `#cgo` directives in `mlx.go` reference:
- `CPPFLAGS: -I${SRCDIR}/dist/include`
- `LDFLAGS: -L${SRCDIR}/dist/lib -lmlxc -lmlx`
- Frameworks: Foundation, Metal, Accelerate

### CMake Config

`CMakeLists.txt` fetches mlx-c v0.4.1 from GitHub. Key settings:
- `MLX_BUILD_SAFETENSORS=ON` (model loading)
- `MLX_BUILD_GGUF=OFF` (GGUF is for llama.cpp backend in go-ai/ml/)
- `BUILD_SHARED_LIBS=ON`
- macOS deployment target: 26.0

## Architecture

```
go-mlx/
├── Core Layer (Metal GPU)
│   ├── mlx.go              — Init, Materialize, MetalAvailable (CGO bridge)
│   ├── mlx_stub.go         — Non-Apple fallback
│   ├── array.go            — MLX array wrapper (create, reshape, data access)
│   ├── dtype.go            — Data types (Float16, Float32, BFloat16, Int32, etc.)
│   ├── stream.go           — Metal stream/queue management
│   └── CMakeLists.txt      — Fetches mlx-c v0.4.1
│
├── Operations
│   ├── ops.go              — Element-wise: Add, Multiply, MatMul, Softmax, etc.
│   ├── fast.go             — Fused Metal kernels: RMSNorm, RoPE, ScaledDotProductAttention
│   ├── nn.go               — Neural network layers: Linear, Embedding, RMSNorm
│   ├── compile.go          — Compiled function closures for kernel fusion
│   ├── slice.go            — Array slicing/indexing
│   └── random.go           — Categorical sampling
│
├── Training
│   ├── grad.go             — VJP (Vector-Jacobian Product) gradient computation
│   ├── lora.go             — LoRA adapter (rank decomposition fine-tuning)
│   └── optim.go            — AdamW optimiser
│
├── Model Support
│   ├── model/
│   │   ├── model.go        — Base model interface (LoadModel, Generate)
│   │   ├── gemma3.go       — Google Gemma3 decoder
│   │   └── qwen3.go        — Alibaba Qwen3 decoder
│   ├── tokenizer/
│   │   └── tokenizer.go    — BPE tokenizer (sentencepiece format)
│   ├── sample/
│   │   └── sample.go       — Sampling strategies (temperature, top-k, top-p)
│   └── cache/
│       └── cache.go        — KV cache for autoregressive inference
│
└── I/O
    └── io.go               — Safetensors model loading
```

## Key Interfaces

```go
// model/model.go — base interface for all models
type Model interface {
    Forward(x *mlx.Array, cache *cache.KVCache) *mlx.Array
}

// Top-level — GPU materialisation
mlx.Materialize(outputs...)      // Sync GPU eval
mlx.MaterializeAsync(outputs...) // Async GPU eval
mlx.MetalAvailable() bool        // Check Metal GPU

// Array operations
mlx.NewArray(data, shape, dtype)
mlx.MatMul(a, b)
mlx.Softmax(x, axis)
mlx.Add(a, b), mlx.Multiply(a, b), mlx.Divide(a, b)
```

## Dependencies

- **mlx-c v0.4.1** (fetched by CMake at build time)
- **Apple frameworks**: Foundation, Metal, Accelerate
- **Go stdlib only** — no external Go dependencies

## Downstream Consumers

- `forge.lthn.ai/core/go-ai/ml` — `backend_mlx.go` uses this for Metal inference
- `forge.lthn.ai/core/go-i18n` — Phase 2a needs Gemma3-1B inference for domain classification (blocked on this package being importable)

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
