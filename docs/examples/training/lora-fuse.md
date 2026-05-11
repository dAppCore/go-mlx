# Fusing a LoRA Adapter Into the Base Model

Once a LoRA adapter is trained, you can bake it into the base model's weights as a fresh, standalone safetensors pack. The fused pack runs without any adapter machinery at inference time — useful for deploy targets that need a single self-contained model, or to eliminate the per-step LoRA matmul cost when you've finalised an adapter you'll use forever.

The trade-off: you lose the ability to swap the adapter on/off or stack additional adapters on the same base. Fuse only when the adapter is "done."

## Basic Fusion

```go
package main

import (
    "context"
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    result, err := mlx.FuseLoRAIntoModelPack(context.Background(), mlx.FuseLoRAOptions{
        ModelPath:   "/models/qwen3-8b",            // safetensors model pack (input)
        AdapterPath: "/runs/qwen3-8b-domain-a/final", // adapter directory
        OutputPath:  "/models/qwen3-8b-domain-a",   // must be a directory, not a file
        Labels: map[string]string{
            "experiment": "domain-classifier-v3",
            "trained_steps": "10000",
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Fused %d weights into %s\n", result.FusedWeights, result.WeightPath)
    fmt.Printf("Provenance: %s\n", result.ProvenancePath)
}
```

## What Fusion Does

For every base weight `W` that has a matching `lora_a`/`lora_b` pair in the adapter:

```
W_fused = W + scale * Bᵀ @ Aᵀ
```

Where `scale = alpha / rank` (read from the adapter's `adapter_config.json`).

The output directory will contain:

| File | Origin |
|------|--------|
| `model.safetensors` (or shards) | Base weights with LoRA fused in |
| `config.json`, `tokenizer.json`, etc. | Copied verbatim from the source pack |
| `adapter_provenance.json` | Records the source model, adapter identity, and fused weight keys |

## Provenance

The provenance file makes the fusion reproducible and auditable:

```json
{
  "version": 1,
  "source_model": {
    "root": "/models/qwen3-8b",
    "format": "safetensors",
    "model_type": "qwen3"
  },
  "adapter": {
    "rank": 8,
    "alpha": 16,
    "scale": 2.0,
    "target_keys": ["q_proj", "v_proj"]
  },
  "output_weight": "model.safetensors",
  "fused_weight_keys": [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    ...
  ],
  "labels": {"experiment": "domain-classifier-v3"}
}
```

## Constraints

- Source pack must be **safetensors** — GGUF source fusion is not yet supported
- Output path must be a **directory**, not a `.safetensors` or `.gguf` file
- Output directory must be empty of `*.safetensors` and `*.gguf` (it can contain other metadata files; those are skipped)
- Output path must differ from the source path (no in-place fusion)
- The adapter's `rank` and `scale` must be present — reads from `adapter_config.json` if not on disk-detectable

## Verifying the Fusion

After fusion, the new pack loads exactly like any other model:

```go
fused, err := mlx.LoadModel("/models/qwen3-8b-domain-a")
// No adapter_path needed — the adapter is baked in.
```

To sanity-check, compare a generation from the fused pack against the same prompt run on the original `base + adapter`. They should match within numerical tolerance (Float16 differences in the matmul order can produce small last-bit divergences; semantic output should be identical).

## See Also

- [LoRA fine-tuning](lora-finetune.md) — produces the adapter you fuse
- [Model merging](../model-ops/merge.md) — combines multiple already-fused packs
- [Training docs](../../docs/training.md#fusing-an-adapter-into-the-base-model) — reference for `FuseLoRAOptions` and `FuseLoRAResult`
