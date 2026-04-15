---
title: go-mlx
description: Native Metal GPU inference and training for Go on Apple Silicon.
---

# go-mlx

`dappco.re/go/mlx` provides native Apple Metal GPU inference and LoRA fine-tuning for Go. It wraps Apple's [MLX](https://github.com/ml-explore/mlx) framework through the [mlx-c](https://github.com/ml-explore/mlx-c) C API, implementing the `inference.Backend` interface from `dappco.re/go/core/inference` and an RFC-style direct root-package API.

**Platform:** darwin/arm64 only (Apple Silicon M1-M4). A stub provides `MetalAvailable() bool` returning false on all other platforms.

## Quick Start

```go
import (
    "context"
    "fmt"

    "dappco.re/go/core/inference"
    _ "dappco.re/go/mlx" // registers "metal" backend via init()
)

func main() {
    m, err := inference.LoadModel("/path/to/model/")
    if err != nil {
        panic(err)
    }
    defer m.Close()

    ctx := context.Background()
    for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
        fmt.Print(tok.Text)
    }
    if err := m.Err(); err != nil {
        panic(err)
    }
}
```

The blank import (`_ "dappco.re/go/mlx"`) auto-registers the Metal backend. You can use either the `go-inference` interfaces or the direct root API:

```go
import (
    "fmt"

    mlx "dappco.re/go/mlx"
)

model, err := mlx.LoadModel("/path/to/model/",
    mlx.WithContextLength(8192),
    mlx.WithDevice("cpu"), // "gpu" or "cpu"
)
if err != nil {
    panic(err)
}
defer model.Close()

text, err := model.Generate("What is 2+2?", mlx.WithMaxTokens(64))
if err != nil {
    panic(err)
}
fmt.Println(text)
```

## Features

- **Streaming inference** -- token-by-token generation via `iter.Seq[Token]` (range-over-func)
- **Multi-turn chat** -- native chat templates for Gemma 3/4, Qwen 2/3, and Llama 3
- **Batch inference** -- `Classify` (prefill-only) and `BatchGenerate` (autoregressive) for multiple prompts
- **LoRA fine-tuning** -- low-rank adaptation with AdamW optimiser and gradient checkpointing
- **Quantisation** -- transparent support for 4-bit and 8-bit quantised models via `QuantizedMatmul`
- **Attention inspection** -- extract post-RoPE K vectors from the KV cache for analysis
- **Performance metrics** -- prefill/decode tokens per second, GPU memory usage

## Supported Models

Models may be loaded from **HuggingFace safetensors shards** or **GGUF checkpoints**. Architecture is auto-detected from `config.json`:

| Architecture | `model_type` values | Tested sizes |
|-------------|---------------------|-------------|
| Gemma 3 | `gemma3`, `gemma3_text`, `gemma2` | 1B, 4B, 27B |
| Gemma 4 | `gemma4`, `gemma4_text` | E2B, E4B, 26B MoE, 31B |
| Qwen 3 | `qwen3`, `qwen2` | 8B+ |
| Llama 3 | `llama` | 8B+ |

## Package Layout

| Package | Purpose |
|---------|---------|
| Root (`mlx`) | Public API: backend registration, direct model API, memory controls, training type exports |
| `internal/metal/` | All CGO code: array ops, model loaders, generation, training primitives |
| `mlxlm/` | Alternative subprocess backend via Python's mlx-lm (no CGO required) |

## Metal Memory Controls

These control the Metal allocator directly, not individual models:

```go
import mlx "dappco.re/go/mlx"

mlx.SetCacheLimit(4 << 30)   // 4 GB cache limit
mlx.SetMemoryLimit(32 << 30) // 32 GB hard limit
mlx.ClearCache()              // release cached memory between chat turns

fmt.Printf("active: %d MB, peak: %d MB\n",
    mlx.GetActiveMemory()/1024/1024,
    mlx.GetPeakMemory()/1024/1024)
```

| Function | Purpose |
|----------|---------|
| `SetCacheLimit(bytes)` | Soft limit on the allocator cache |
| `SetMemoryLimit(bytes)` | Hard ceiling on Metal memory |
| `SetWiredLimit(bytes)` | Wired memory limit |
| `GetActiveMemory()` | Current live allocations in bytes |
| `GetPeakMemory()` | High-water mark since last reset |
| `GetCacheMemory()` | Cached (not yet freed) memory |
| `ClearCache()` | Release cached memory to the OS |
| `ResetPeakMemory()` | Reset the high-water mark |
| `GetDeviceInfo()` | Metal GPU hardware information |

## Performance Baseline

Measured on M3 Ultra (60-core GPU, 96 GB unified memory):

| Operation | Throughput |
|-----------|-----------|
| Gemma3-1B 4-bit prefill | 246 tok/s |
| Gemma3-1B 4-bit decode | 82 tok/s |
| Gemma3-1B 4-bit classify (4 prompts) | 152 prompts/s |
| DeepSeek R1 7B 4-bit decode | 27 tok/s |
| Llama 3.1 8B 4-bit decode | 30 tok/s |

## Documentation

- [Architecture](architecture.md) -- CGO binding layer, lazy evaluation, memory model, attention, KV cache
- [Models](models.md) -- model loading, supported architectures, tokenisation, chat templates
- [Training](training.md) -- LoRA fine-tuning, gradient computation, AdamW optimiser, loss functions
- [Build Guide](build.md) -- prerequisites, CMake setup, build tags, testing

## Downstream Consumers

| Package | Role |
|---------|------|
| `dappco.re/go/core/ml` | Imports go-inference + go-mlx for the Metal backend training loop |
| `dappco.re/go/core/i18n` | Gemma3-1B domain classification (Phase 2a) |
| `dappco.re/go/core/rocm` | Sibling AMD GPU backend, same go-inference interfaces |

## Licence

EUPL-1.2
