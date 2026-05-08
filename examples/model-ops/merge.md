# Merging Model Packs

Combine two or more finetuned model packs of the same architecture into a single pack. Useful when you have specialist models (a math finetune, a coding finetune, a creative-writing finetune) and want a single generalist that inherits something from each.

## TIES — Best Default

TIES (Trim, Elect, Sign) keeps only the top-magnitude fraction of parameter changes per tensor and resolves sign conflicts between sources. It produces noticeably less interference than a plain weighted average.

```go
package main

import (
    "context"
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    result, err := mlx.MergeModelPacks(context.Background(), mlx.ModelMergeOptions{
        Sources: []mlx.ModelMergeSource{
            {Path: "/models/qwen3-8b-math",   Weight: 0.5},
            {Path: "/models/qwen3-8b-code",   Weight: 0.3},
            {Path: "/models/qwen3-8b-prose",  Weight: 0.2},
        },
        OutputPath: "/models/qwen3-8b-merged-ties",
        Method:     mlx.ModelMergeTIES,
        T:          0.7, // keep top 70% magnitude per tensor
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

| Method | When to use |
|--------|-------------|
| `ModelMergeLinear` | Baseline. Simple weighted average — works, but tensor changes can interfere destructively. |
| `ModelMergeSLERP` | Spherical interpolation. Better when the two sources have learnt rotations of similar features. Currently supports two-source merges most cleanly. |
| `ModelMergeTIES` | Top-magnitude trim with sign resolution. Best general-purpose default. Use `T` ∈ (0, 1] to control keep-fraction (0.7 is a sensible start). |
| `ModelMergeDARE` | Drop-And-REscale. Randomly zero parameters then scale. Sometimes pairs well with TIES. |

## Two-Source SLERP

```go
result, err := mlx.MergeModelPacks(ctx, mlx.ModelMergeOptions{
    Sources: []mlx.ModelMergeSource{
        {Path: "/models/qwen3-8b-base-A", Weight: 1.0},
        {Path: "/models/qwen3-8b-base-B", Weight: 1.0},
    },
    OutputPath: "/models/qwen3-8b-slerp",
    Method:     mlx.ModelMergeSLERP,
    T:          0.5, // interpolation factor (0 = source A, 1 = source B)
})
```

## Compatibility Checks

By default, the merger refuses if sources disagree on architecture, tokenizer, or per-tensor shape. For experiments that deliberately cross those boundaries, relax explicitly:

```go
opts := mlx.ModelMergeOptions{
    Sources:                   []mlx.ModelMergeSource{ /* ... */ },
    OutputPath:                "/models/cross-arch-experiment",
    Method:                    mlx.ModelMergeLinear,
    AllowArchitectureMismatch: true, // accept mismatched config.json model_type
    AllowTokenizerMismatch:    true, // accept mismatched tokenizer.json
    AllowTensorMismatch:       true, // skip tensors that don't share shape
}
```

`SkippedTensors` in the result lists every tensor that was copied verbatim or skipped because of incompatible shapes — useful for understanding what actually got merged when allow-flags were on.

## Provenance

Every merge writes `model_merge_provenance.json`:

```json
{
  "version": 1,
  "method": "ties",
  "t": 0.7,
  "sources": [
    {"path": "/models/qwen3-8b-math",  "weight": 0.5},
    {"path": "/models/qwen3-8b-code",  "weight": 0.3},
    {"path": "/models/qwen3-8b-prose", "weight": 0.2}
  ],
  "merged_tensors": 387,
  "copied_tensors": 12,
  "skipped_tensors": [],
  "labels": {"experiment": "math-code-prose-blend"}
}
```

Reproducible to the byte: same inputs + method + T = same output.

## After Merging

The output pack is a standard safetensors model pack and loads as usual:

```go
m, err := mlx.LoadModel("/models/qwen3-8b-merged-ties")
```

Common next steps:
- [Eval](../eval/perplexity.md) on a held-out set to confirm the merge didn't regress baseline quality
- [Quantise to GGUF](quantize-gguf.md) for deploy
- [Further LoRA fine-tune](../training/lora-finetune.md) to tune the merge

## See Also

- [Model operations docs](../../docs/model-operations.md#model-merge) — full reference
- [LoRA fuse](../training/lora-fuse.md) — bake an adapter in before merging
