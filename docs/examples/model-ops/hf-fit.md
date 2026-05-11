# HuggingFace Fit Planner

`PlanHFModelFits` queries HuggingFace Hub metadata and reports which models will actually fit on the target device, given the user's intended context length, KV headroom, and LoRA rank. No model files are downloaded — purely a planning step before you commit to a multi-GB pull.

## Searching The Hub

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    mlx "dappco.re/go/mlx"
)

func main() {
    src := mlx.NewHuggingFaceModelSource(mlx.HuggingFaceModelSourceConfig{
        BaseURL:   "https://huggingface.co",
        Token:     os.Getenv("HF_TOKEN"),
        UserAgent: "go-mlx/research",
    })

    ctx := context.Background()
    report, err := mlx.PlanHFModelFits(ctx, mlx.HFModelFitConfig{
        Query:       "qwen 3 instruct",
        MaxResults:  10,
        Device:      mlx.GetDeviceInfo(), // current Mac's hardware info
        Source:      src,
        LoRARank:    8,
        KVBytes:     2 << 30, // 2 GB KV headroom
        ContextHint: 8192,
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("device: %s, %s class\n", report.Device.Name, report.DeviceClass)
    fmt.Printf("memory plan: weights=%s kv=%s activations=%s\n",
        humanise(report.MemoryPlan.Weights),
        humanise(report.MemoryPlan.KV),
        humanise(report.MemoryPlan.Activations))

    fmt.Println()
    fmt.Println("Models:")
    for _, plan := range report.Models {
        marker := "✗"
        if plan.Fits {
            marker = "✓"
        }
        fmt.Printf("  %s %-50s  weights=%s\n", marker, plan.ModelID, humanise(plan.WeightsBytes))
    }
}
```

## Specific Model Lookup

Skip the search and ask about exact model IDs:

```go
report, err := mlx.PlanHFModelFits(ctx, mlx.HFModelFitConfig{
    ModelIDs: []string{
        "Qwen/Qwen3-8B-Instruct",
        "Qwen/Qwen3-32B-Instruct",
        "google/gemma-3-1b-it",
        "google/gemma-3-27b-it",
    },
    Device:      mlx.GetDeviceInfo(),
    Source:      src,
    KVBytes:     2 << 30,
    ContextHint: 8192,
})
```

## Local Pack Inspection

`LocalPaths` lets you fold already-downloaded packs into the same comparison:

```go
report, err := mlx.PlanHFModelFits(ctx, mlx.HFModelFitConfig{
    ModelIDs:   []string{"Qwen/Qwen3-32B-Instruct"},
    LocalPaths: []string{"/models/qwen3-8b", "/models/qwen3-8b-q4"},
    Device:     mlx.GetDeviceInfo(),
    Source:     src,
    KVBytes:    2 << 30,
})
```

The local packs are inspected via the same path that `LoadModel` uses, so you get accurate per-pack numbers without trusting the Hub's metadata.

## Output

```go
type HFModelFitReport struct {
    Query       string
    Device      DeviceInfo
    DeviceClass MemoryClass            // tiered classification (e.g. "96GB ultra")
    MemoryPlan  MemoryPlan             // weights / KV / activations / LoRA breakdown
    Models      []HFModelFitPlan       // per-model verdicts
}
```

Each `HFModelFitPlan` includes the model ID, the projected weights size in the requested quantisation tier, the KV size given `ContextHint`, the activations envelope, and a `Fits bool`.

## Pure Planning, No Network for Local

The HF source uses `core.HTTPClient`. If `Source` is nil, `PlanHFModelFits` will only consider `LocalPaths` and `ModelIDs` against any cached metadata — never makes a network call. Use this in CI or air-gapped setups.

## See Also

- [Model operations docs](../../docs/model-operations.md#huggingface-fit-planner) — full reference
- [Inference: quantised models](../inference/quantization.md) — once you've chosen a model that fits, load it
