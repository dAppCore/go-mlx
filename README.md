[![Go Reference](https://pkg.go.dev/badge/dappco.re/go/core/mlx.svg)](https://pkg.go.dev/dappco.re/go/core/mlx)
[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL--1.2-blue.svg)](LICENSE.md)
[![Go Version](https://img.shields.io/badge/Go-1.26-00ADD8?style=flat&logo=go)](go.mod)

# go-mlx

Native Apple Metal GPU inference via mlx-c CGO bindings, implementing the `inference.Backend` and `inference.TextModel` interfaces from go-inference for Apple Silicon (M1-M4). Supports Gemma 3, Gemma 4 (dense and MoE), Qwen 2/3, and Llama 3 architectures from HuggingFace safetensors directories and GGUF checkpoints, with fused Metal kernels for RMSNorm, RoPE, scaled dot-product attention, KV cache management, LoRA fine-tuning with AdamW, and batch inference. The root package also exposes an RFC-style direct API (`mlx.LoadModel`, `model.Generate`, `model.GenerateStream`) with `gpu` and `cpu` device selection. A Python subprocess backend (`mlxlm`) is provided as a CGO-free alternative. Platform-restricted: `darwin/arm64` only; a no-op stub compiles on all other platforms.

**Module**: `dappco.re/go/core/mlx`
**Licence**: EUPL-1.2
**Language**: Go 1.25

## Quick Start

```go
import (
    "dappco.re/go/core/inference"
    _ "dappco.re/go/core/mlx"  // registers "metal" backend via init()
)

model, err := inference.LoadModel("/Volumes/Data/lem/safetensors/gemma-3-1b/")
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

## Documentation

- [Architecture](docs/architecture.md) — CGO binding, model architectures, weight loading, KV cache, attention, batch inference, LoRA training, mlxlm backend
- [Development Guide](docs/development.md) — prerequisites (mlx-c CMake build), CGO flags, test patterns, benchmarks
- [Project History](docs/history.md) — completed phases, commit hashes, known limitations

## Build & Test

```bash
go generate ./...        # builds mlx-c C library (required first time)
go test ./...
go build ./...
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
