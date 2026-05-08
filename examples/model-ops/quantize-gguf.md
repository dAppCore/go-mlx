# Quantising Safetensors → GGUF

`QuantizeModelPackToGGUF` reads a HuggingFace safetensors model pack and writes a GGUF checkpoint with the requested quantisation format. Native Go — no `llama.cpp`, no Python, no external tools.

## Q4_K_M (Recommended Default)

```go
package main

import (
    "context"
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    result, err := mlx.QuantizeModelPackToGGUF(context.Background(), mlx.QuantizeGGUFOptions{
        ModelPath:  "/models/qwen3-8b",
        OutputPath: "/models/qwen3-8b-q4km.gguf",
        Format:     mlx.GGUFQuantizeQ4_K_M,
        Labels:     map[string]string{"target": "phone-deploy", "source_revision": "v1.2"},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("%d/%d tensors quantised\n", result.QuantizedTensors, result.TensorCount)
    fmt.Printf("Output: %s (%.2f GB)\n", result.WeightPath, gibibytes(result.WeightPath))
    if len(result.Notes) > 0 {
        fmt.Println("Notes:")
        for _, note := range result.Notes {
            fmt.Println(" ·", note)
        }
    }
}
```

A typical Qwen3-8B → Q4_K_M run takes ~90 s on M3 Ultra and produces a ~4.6 GB file.

## Comparing Formats

```go
for _, format := range []mlx.GGUFQuantizeFormat{
    mlx.GGUFQuantizeQ8_0,    // 8-bit, ~8 GB, near-lossless
    mlx.GGUFQuantizeQ4_K_M,  // 4.5-bit, ~4.6 GB, recommended
    mlx.GGUFQuantizeQ4_0,    // 4-bit, ~4.3 GB, fastest
} {
    out := fmt.Sprintf("/models/qwen3-8b-%s.gguf", string(format))
    res, err := mlx.QuantizeModelPackToGGUF(ctx, mlx.QuantizeGGUFOptions{
        ModelPath:  "/models/qwen3-8b",
        OutputPath: out,
        Format:     format,
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("%s: %d quantised, %d total", format, res.QuantizedTensors, res.TensorCount)
}
```

## What Gets Quantised

The orchestrator quantises matmul weight tensors (attention, MLP). It deliberately leaves these unchanged:

- **Embedding** and **`lm_head`** tables — typically copied as F16 because quantising them loses noticeably more quality than the gain in size
- **Norm scales** — already small, no benefit from quantisation
- **Per-layer biases** — small, kept at full precision

`result.Notes` records any per-tensor decisions (e.g. "fell back to F16 for embed_tokens — output dim too small for Q4_K block").

## Loading the Output

The produced file is a standard GGUF checkpoint and loads with no extra flags:

```go
model, err := mlx.LoadModel("/models/qwen3-8b-q4km.gguf")
```

Architecture, tokenizer, and quant format are all read from the GGUF metadata.

## Inspecting Without Loading

`ReadGGUFInfo` reads just the metadata header — fast, no weight materialisation:

```go
info, _ := mlx.ReadGGUFInfo("/models/qwen3-8b-q4km.gguf")
fmt.Printf("arch=%s vocab=%d quant=%s tensors=%d\n",
    info.Architecture, info.VocabSize, info.QuantFormat, info.TensorCount)
```

## See Also

- [Inference: quantised models](../inference/quantization.md) — loading any quant format
- [Model merge](merge.md) — usually quantise *after* merging, not before
- [LoRA fuse](../training/lora-fuse.md) — fuse adapter into base before quantising for a single-file deploy
