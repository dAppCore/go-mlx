# Merging Model Packs

Combine two or more finetuned model packs of the same architecture into a single pack. Useful when you have specialist models (a math finetune, a coding finetune, a creative-writing finetune) and want a single generalist that inherits something from each.

`merge.Packs` (package `dappco.re/go/mlx/merge`) does the work. Compatibility checks, the metadata-copy-on-merge step, and the merge-method vocabulary are shared with every engine via `dappco.re/go/inference/merge` — this package owns the local orchestration (safetensors indexing, chunked tensor writes) on top of that shared contract.

## Linear — Baseline Default

Weighted average across sources. Simplest, fastest, and the only method that scales cleanly past two sources today:

```go
package main

import (
    "context"
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
    "dappco.re/go/mlx/merge"
)

func main() {
    ctx := context.Background()

    math, err := mlx.ValidateModelPack("/models/qwen3-8b-math")
    if err != nil {
        log.Fatal(err)
    }
    code, err := mlx.ValidateModelPack("/models/qwen3-8b-code")
    if err != nil {
        log.Fatal(err)
    }
    prose, err := mlx.ValidateModelPack("/models/qwen3-8b-prose")
    if err != nil {
        log.Fatal(err)
    }

    result, err := merge.Packs(ctx, merge.Options{
        Sources: []merge.Source{
            {Pack: math, Weight: 0.5},
            {Pack: code, Weight: 0.3},
            {Pack: prose, Weight: 0.2},
        },
        OutputPath: "/models/qwen3-8b-merged-linear",
        Method:     merge.MethodLinear,
        Labels: map[string]string{
            "experiment": "math-code-prose-blend",
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("merged %d tensors, copied %d, skipped %d\n",
        result.MergedTensors, result.CopiedTensors, len(result.SkippedTensors))
    fmt.Printf("output: %s\n", result.WeightPath)
    fmt.Printf("provenance: %s\n", result.ProvenancePath)
}
```

## Method Comparison

| Method | Status | When to use |
|--------|--------|-------------|
| `merge.MethodLinear` | Implemented | Weighted average — simplest, fastest, works with any number of sources. |
| `merge.MethodSLERP` | Implemented | Spherical interpolation. Better when two sources have learnt rotations of similar features. Requires exactly two sources. |
| `merge.MethodTIES` | Reserved, not implemented | Trim-Elect-Sign — `Packs` returns an error today. The constant exists so a future sparse-merge hook can land without a method-name break. |
| `merge.MethodDARE` | Reserved, not implemented | Drop-And-REscale — same reserved status as TIES. |

## Two-Source SLERP

```go
result, err := merge.Packs(ctx, merge.Options{
    Sources: []merge.Source{
        {Pack: baseA, Weight: 1.0},
        {Pack: baseB, Weight: 1.0},
    },
    OutputPath: "/models/qwen3-8b-slerp",
    Method:     merge.MethodSLERP,
    T:          0.5, // interpolation factor (0 = source A, 1 = source B)
})
```

## Compatibility Checks

By default, the merger refuses if sources disagree on architecture, tokenizer, or per-tensor shape. For experiments that deliberately cross those boundaries, relax explicitly:

```go
opts := merge.Options{
    Sources:                   []merge.Source{ /* ... */ },
    OutputPath:                "/models/cross-arch-experiment",
    Method:                    merge.MethodLinear,
    AllowArchitectureMismatch: true, // accept mismatched config.json model_type
    AllowTokenizerMismatch:    true, // accept mismatched tokenizer.json
    AllowTensorMismatch:       true, // skip tensors that don't share shape
}
```

`SkippedTensors` in the result lists every tensor that was copied verbatim or skipped because of incompatible shapes — useful for understanding what actually got merged when allow-flags were on.

## Provenance

Every merge writes a provenance file (`merge.ProvenanceFile`, shared with `dappco.re/go/inference/merge` so every engine agrees on the filename) alongside the output pack, recording the method, `T`, each source pack plus its weight, and the merged/copied/skipped tensor counts:

```go
type Provenance struct {
    Version        int
    Method         Method
    T              float64
    Sources        []Source       // Pack + Weight per source
    SourcePacks    []mp.ModelPack
    OutputWeight   string
    MergedTensors  int
    CopiedTensors  int
    SkippedTensors []string
    Labels         map[string]string
}
```

Reproducible to the byte: same inputs + method + T = same output.

## After Merging

The output pack is a standard safetensors model pack and loads as usual:

```go
m, err := mlx.LoadModel(result.OutputPath)
```

Common next steps:
- [Eval](../eval/perplexity.md) on a held-out set to confirm the merge didn't regress baseline quality
- [Quantise to GGUF](quantize-gguf.md) for deploy
- [Further LoRA fine-tune](../training/lora-finetune.md) to tune the merge
- [Weight comparison](../../docs/model-operations.md#weight-comparison) (`merge.ComparePacks`) to inspect what a merge actually changed

## See Also

- [Model operations docs](../../docs/model-operations.md#model-merge) — full reference
- [LoRA fuse](../training/lora-fuse.md) — bake an adapter in before merging
