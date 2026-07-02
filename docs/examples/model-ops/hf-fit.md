# HuggingFace Fit Planner

`hf.PlanFits` (package `dappco.re/go/mlx/hf`) queries HuggingFace Hub metadata and reports which models will actually fit on the target device, given the user's intended context length, KV headroom, and LoRA rank. No model files are downloaded — purely a planning step before you commit to a multi-GB pull.

## Device Info

`hf.FitConfig.Device` takes a `dappco.re/go/mlx/memory.DeviceInfo` — a narrower, sibling type to the `mlx.GetDeviceInfo()` result (it drops the `Name` field). Copy the shared fields across:

```go
gpu := mlx.GetDeviceInfo()
device := memory.DeviceInfo{
    Architecture:                 gpu.Architecture,
    MaxBufferLength:              gpu.MaxBufferLength,
    MaxRecommendedWorkingSetSize: gpu.MaxRecommendedWorkingSetSize,
    MemorySize:                   gpu.MemorySize,
}
```

## Searching The Hub

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    mlx "dappco.re/go/mlx"
    "dappco.re/go/mlx/hf"
    "dappco.re/go/mlx/memory"
)

func main() {
    gpu := mlx.GetDeviceInfo()
    device := memory.DeviceInfo{
        Architecture:                 gpu.Architecture,
        MaxBufferLength:              gpu.MaxBufferLength,
        MaxRecommendedWorkingSetSize: gpu.MaxRecommendedWorkingSetSize,
        MemorySize:                   gpu.MemorySize,
    }

    src := hf.NewRemoteSource(hf.RemoteConfig{
        Token:     os.Getenv("HF_TOKEN"),
        UserAgent: "go-mlx/research",
    })

    ctx := context.Background()
    report, err := hf.PlanFits(ctx, hf.FitConfig{
        Query:       "qwen 3 instruct",
        MaxResults:  10,
        Device:      device,
        Source:      src,
        LoRARank:    8,
        KVBytes:     2 << 30, // 2 GB KV headroom
        ContextHint: 8192,
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("device class: %s\n", report.DeviceClass)

    fmt.Println("Models:")
    for _, plan := range report.Models {
        marker := "✗"
        if plan.InferenceFits {
            marker = "✓"
        }
        fmt.Printf("  %s %-50s  weights=%d bytes\n", marker, plan.ModelID, plan.WeightBytes)
    }
}
```

## Specific Model Lookup

Skip the search and ask about exact model IDs:

```go
report, err := hf.PlanFits(ctx, hf.FitConfig{
    ModelIDs: []string{
        "Qwen/Qwen3-8B-Instruct",
        "Qwen/Qwen3-32B-Instruct",
        "google/gemma-3-1b-it",
        "google/gemma-3-27b-it",
    },
    Device:      device,
    Source:      src,
    KVBytes:     2 << 30,
    ContextHint: 8192,
})
```

## Local Pack Inspection

`LocalPaths` lets you fold already-downloaded packs into the same comparison:

```go
report, err := hf.PlanFits(ctx, hf.FitConfig{
    ModelIDs:   []string{"Qwen/Qwen3-32B-Instruct"},
    LocalPaths: []string{"/models/qwen3-8b", "/models/qwen3-8b-q4"},
    Device:     device,
    Source:     src,
    KVBytes:    2 << 30,
})
```

The local packs are inspected via the same path that `mlx.LoadModel` uses, so you get accurate per-pack numbers without trusting the Hub's metadata.

## Output

```go
type FitReport struct {
    Query       string
    Device      memory.DeviceInfo
    DeviceClass memory.Class // tiered classification, e.g. "apple-silicon-96gb"
    MemoryPlan  memory.Plan  // cache policy / batch / context-length planning
    Models      []FitPlan    // per-model verdicts, sorted fits-first then by size
}
```

Each `FitPlan` includes the model ID, architecture, `WeightBytes`/`ExpectedKVBytes`/`ExpectedRuntimeBytes`/`ExpectedTotalBytes` projections in the requested quantisation tier, and the two verdict booleans `MemoryFits` (weights + KV fit device memory) and `InferenceFits`.

## Pure Planning, No Network For Local-Only Reports

The HF source uses `core.HTTPClient`. `Source` may be left nil only when both `Query` and `ModelIDs` are empty — a `LocalPaths`-only report that never makes a network call, useful in CI or air-gapped setups. Setting `Query` or `ModelIDs` with a nil `Source` returns an error rather than silently skipping the lookup.

## See Also

- [Model operations docs](../../docs/model-operations.md#huggingface-fit-planner) — full reference
- [Inference: quantised models](../inference/quantization.md) — once you've chosen a model that fits, load it
