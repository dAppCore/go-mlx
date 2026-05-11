# LoRA Fine-Tuning

End-to-end LoRA fine-tuning loop using `go-mlx` primitives directly. For higher-level orchestration, downstream `go-ml` wraps this same loop with progress tracking and checkpoint management.

## Setup

```go
package main

import (
    "fmt"
    "log"

    "dappco.re/go/inference"
    mlx "dappco.re/go/mlx"
)

func main() {
    // Load the base model as a TrainableModel.
    tm, err := inference.LoadTrainable("/models/qwen3-8b/")
    if err != nil {
        log.Fatal(err)
    }
    defer tm.Close()

    // Apply LoRA adapter to attention projections.
    adapter := tm.ApplyLoRA(inference.LoRAConfig{
        Rank:       8,
        Alpha:      16,
        TargetKeys: []string{"q_proj", "v_proj"},
        BFloat16:   true, // halves adapter memory
    })

    // Get direct InternalModel access for forward passes.
    model := mlx.TrainingModel(tm)
    concrete := mlx.ConcreteAdapter(adapter)
    fmt.Printf("LoRA params: %d\n", concrete.TotalParams())
```

## Loss Function

```go
    lossFn := func(params []*mlx.Array) []*mlx.Array {
        concrete.SetAllParams(params)

        // Forward pass on the current batch.
        logits := model.Forward(inputTokens, caches) // [B, L, V]

        // Cross-entropy on the next-token targets.
        loss := mlx.MaskedCrossEntropyLoss(logits, targets, mask)
        return []*mlx.Array{loss}
    }
```

`MaskedCrossEntropyLoss` excludes padding and (commonly) prompt-side tokens — the mask is `[B, L]` with 1.0 for "compute loss here" and 0.0 for "ignore".

## Training Loop

```go
    grad := mlx.ValueAndGrad(lossFn, 0)
    opt := mlx.NewAdamW(&mlx.AdamWConfig{
        LearningRate: 1e-4,
        WeightDecay:  0.01,
    })

    params := concrete.AllTrainableParams()
    for step := 0; step < numSteps; step++ {
        values, grads, err := grad.Apply(params...)
        if err != nil {
            log.Fatal(err)
        }
        loss := values[0]
        mlx.Materialize(loss)
        if step%50 == 0 {
            fmt.Printf("step %d: loss=%.4f\n", step, loss.Float())
        }

        params = opt.Step(params, grads)
        concrete.SetAllParams(params)
    }
```

## Checkpointing

Save adapter weights periodically:

```go
    if step%500 == 0 {
        path := fmt.Sprintf("/runs/qwen3-8b-domain-a/step-%06d.safetensors", step)
        if err := concrete.Save(path); err != nil {
            log.Fatal(err)
        }
    }
```

The saved file contains only the A and B matrices, not the base weights. To resume training, reload via `inference.WithAdapterPath` (see [Training docs](../../docs/training.md#saving-and-loading-adapters)).

## Gradient Checkpointing

For memory-constrained training with large models, wrap the forward pass:

```go
    lossFn = mlx.Checkpoint(lossFn)
```

Activations are recomputed during backward instead of stored — trades compute for memory. Useful for 27B+ at full sequence length on a single 96 GB Mac.

## Mixed Precision

`BFloat16: true` on `LoRAConfig` puts A/B in bfloat16 while the base model stays at its loaded precision (typically Float16). MLX auto-promotes for cross-dtype matmuls, so no manual casting is needed. Memory drops by ~50% on the adapter parameters with no measurable accuracy loss in practice.

## Next Steps

- [Fuse the trained adapter into the base](lora-fuse.md) for runtime-cost-free inference
- [Distil from a larger teacher into this base](distill.md) before LoRA fine-tuning to give the student a head start
- [Run perplexity eval](../eval/perplexity.md) on a held-out set after training
