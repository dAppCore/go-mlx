# Quantising Safetensors → GGUF

`gguf.QuantizeModelPack` (package `dappco.re/go/mlx/gguf`) reads a HuggingFace safetensors model pack and writes a GGUF checkpoint with the requested quantisation format. Native Go — no `llama.cpp`, no Python, no external tools. The per-block quantisation maths (`Quantize`/`AppendQuantize`) live in the shared `dappco.re/go/inference/gguf` package so every engine (mlx, rocm, cpu) produces byte-identical GGUF blocks; this package owns the model-pack orchestration, streaming tensor I/O, and metadata copy around that shared codec.

## Q4_K_M (Recommended Default)

```go
package main

import (
    "context"
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
    "dappco.re/go/mlx/gguf"
)

func main() {
    ctx := context.Background()

    source, err := mlx.ValidateModelPack("/models/qwen3-8b")
    if err != nil {
        log.Fatal(err)
    }

    result, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
        SourcePack: source,
        OutputPath: "/models/qwen3-8b-q4km",
        Format:     gguf.QuantizeQ4_K_M,
        Labels:     map[string]string{"target": "phone-deploy", "source_revision": "v1.2"},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("%d/%d tensors quantised\n", result.QuantizedTensors, result.TensorCount)
    fmt.Printf("Output: %s\n", result.WeightPath)
    for _, note := range result.Notes {
        fmt.Println(" ·", note)
    }
}
```

`OutputPath` must be a model-pack directory (not a bare `.gguf` file) — `QuantizeModelPack` writes `model.gguf` plus copied metadata into it. A typical Qwen3-8B → Q4_K_M run takes roughly a minute on M3 Ultra.

## Comparing Formats

```go
for _, format := range []gguf.QuantizeFormat{
    gguf.QuantizeQ8_0,   // 8-bit, near-lossless
    gguf.QuantizeQ4_K_M, // ~4.5-bit K-quant, recommended
    gguf.QuantizeQ4_0,   // 4-bit, fastest
} {
    out := fmt.Sprintf("/models/qwen3-8b-%s", string(format))
    res, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
        SourcePack: source,
        OutputPath: out,
        Format:     format,
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("%s: %d quantised, %d total", format, res.QuantizedTensors, res.TensorCount)
}
```

Other supported formats: `QuantizeQ5_0`, `QuantizeQ4_K`, `QuantizeQ5_K`, `QuantizeQ6_K`, `QuantizeQ8_K`, `QuantizeQ3_K`, `QuantizeQ2_K`.

## What Gets Quantised

The orchestrator quantises matmul weight tensors (attention, MLP). It deliberately leaves these unchanged:

- **Embedding** and **`lm_head`** tables — typically copied as F16 because quantising them loses noticeably more quality than the gain in size
- **Norm scales** — already small, no benefit from quantisation
- **Per-layer biases** — small, kept at full precision

`result.Notes` records any per-tensor decisions (e.g. a fallback to F16 for a tensor too small for the requested K-quant block size).

## Loading the Output

The produced pack is a standard GGUF checkpoint and loads with no extra flags:

```go
model, err := mlx.LoadModel(result.OutputPath)
```

Architecture, tokenizer, and quant format are all read from the GGUF metadata.

## Inspecting Without Loading

`gguf.ReadInfo` reads just the metadata header — fast, no weight materialisation:

```go
info, err := gguf.ReadInfo(result.WeightPath)
fmt.Printf("arch=%s vocab=%d quant=%s tensors=%d\n",
    info.Architecture, info.VocabSize, info.QuantType, info.TensorCount)
```

## See Also

- [Inference: quantised models](../inference/quantization.md) — loading any quant format
- [Model merge](merge.md) — usually quantise *after* merging, not before
- [LoRA fuse](../training/lora-fuse.md) — fuse adapter into base before quantising for a single-file deploy
