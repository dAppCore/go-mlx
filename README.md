[![Go Reference](https://pkg.go.dev/badge/dappco.re/go/mlx.svg)](https://pkg.go.dev/dappco.re/go/mlx)
[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL--1.2-blue.svg)](LICENSE.md)
[![Go Version](https://img.shields.io/badge/Go-1.26-00ADD8?style=flat&logo=go)](go.mod)

# go-mlx

Native Apple Metal GPU inference via mlx-c CGO bindings, implementing the `inference.Backend` and `inference.TextModel` interfaces from go-inference for Apple Silicon (M1-M4). Supports Gemma 3, Gemma 4 (dense and MoE), Qwen 2/3, and Llama 3 architectures from HuggingFace safetensors directories and GGUF checkpoints, with fused Metal kernels for RMSNorm, RoPE, scaled dot-product attention, KV cache management, LoRA fine-tuning with AdamW, and batch inference. The root package also exposes an RFC-style direct model API (`mlx.LoadModel`, `model.Generate`, `model.GenerateStream`) and a non-LLM frame-compute API (`mlx.NewSession`, `Session.BeginFrame`, `Session.FinishFrame`, `PixelBuffer`, `KernelRGB565ToRGBA8`, `KernelNearestScale`, `KernelScanlineFilter`, `KernelCRTFilter`, `KernelSoftenFilter`, `KernelSharpenFilter`) for Apple GPU-accelerated image and emulator workloads. A Python subprocess backend (`mlxlm`) is provided as a CGO-free alternative. Platform-restricted: `darwin/arm64` only; a no-op stub compiles on all other platforms.

**Module**: `dappco.re/go/mlx`
**Licence**: EUPL-1.2
**Language**: Go 1.25

## Quick Start

```go
import (
    "context"
    "fmt"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"  // registers "metal" backend via init()
)

model, err := inference.LoadModel("/Volumes/Data/lem/safetensors/gemma-3-1b/")
defer model.Close()

for tok := range model.Generate(context.Background(), "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

## Root API

```go
import (
    "fmt"

    mlx "dappco.re/go/mlx"
)

model, err := mlx.LoadModel("/path/to/model",
    mlx.WithContextLength(8192),
    mlx.WithQuantization(4),
    mlx.WithDevice("gpu"),
)
if err != nil {
    panic(err)
}
defer model.Close()

reply, err := model.Generate("Explain Gemma 4 shared KV layers", mlx.WithMaxTokens(128))
if err != nil {
    panic(err)
}
fmt.Println(reply)
```

## Frame Compute

```go
import mlx "dappco.re/go/mlx"

session, err := mlx.NewSession(mlx.WithSessionLabel("frame-pipeline"))
if err != nil {
    panic(err)
}
defer session.Close()

src, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  320,
    Height: 224,
    Stride: 640,
    Format: mlx.PixelRGB565,
})
rgba, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  320,
    Height: 224,
    Stride: 1280,
    Format: mlx.PixelRGBA8,
})
scaled, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  960,
    Height: 672,
    Stride: 3840,
    Format: mlx.PixelRGBA8,
})

frameBytes := make([]byte, src.Descriptor().SizeBytes())
if err := src.Upload(frameBytes); err != nil {
    panic(err)
}
if err := session.BeginFrame(); err != nil {
    panic(err)
}
if err := session.Run(mlx.KernelRGB565ToRGBA8, mlx.KernelArgs{
    Inputs:  map[string]mlx.Buffer{"src": src},
    Outputs: map[string]mlx.Buffer{"dst": rgba},
}); err != nil {
    panic(err)
}
if err := session.Run(mlx.KernelNearestScale, mlx.KernelArgs{
    Inputs:  map[string]mlx.Buffer{"src": rgba},
    Outputs: map[string]mlx.Buffer{"dst": scaled},
}); err != nil {
    panic(err)
}
if err := session.Run(mlx.KernelScanlineFilter, mlx.KernelArgs{
    Inputs:  map[string]mlx.Buffer{"src": scaled},
    Outputs: map[string]mlx.Buffer{"dst": scaled},
    Scalars: map[string]float64{"strength": 0.3},
}); err != nil {
    panic(err)
}
frameMetrics, err := session.FinishFrame()
if err != nil {
    panic(err)
}

finalFrame, err := scaled.Read()
if err != nil {
    panic(err)
}
_ = finalFrame
_ = frameMetrics
```

## Documentation

- [Compute Guide](docs/compute.md) — frame-oriented Metal compute sessions, pixel buffers, kernels, metrics
- [Architecture](docs/architecture.md) — CGO binding, model architectures, weight loading, KV cache, attention, batch inference, LoRA training, mlxlm backend
- [Models](docs/models.md) — model loading, supported architectures, tokenisation, chat templates
- [Training](docs/training.md) — LoRA fine-tuning, AdamW, gradient computation, checkpoints
- [Development Guide](docs/development.md) — prerequisites (mlx-c CMake build), CGO flags, test patterns, benchmarks
- [Project History](docs/history.md) — completed phases, commit hashes, known limitations

## Build & Test

```bash
git submodule update --init --recursive
go generate ./...        # builds mlx-c C library (required first time)
go test ./...
go build ./...
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
