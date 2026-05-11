# Dataset-Native Perplexity

Run perplexity over a JSONL dataset directly against a loaded model. Useful for tracking quality during finetuning, gating CI, or comparing checkpoints.

## Minimal Run

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    core "dappco.re/go"
    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
    mlx "dappco.re/go/mlx"
)

func main() {
    model, err := inference.LoadModel("/models/qwen3-8b/")
    if err != nil { log.Fatal(err) }
    defer model.Close()

    f, err := os.Open("/data/heldout.jsonl")
    if err != nil { log.Fatal(err) }
    defer f.Close()

    dataset, err := mlx.LoadJSONLDataset(f, mlx.DatasetConfig{})
    if err != nil { log.Fatal(err) }

    report, err := mlx.RunModelEval(context.Background(), model.(*mlx.Model), dataset, mlx.EvalConfig{
        Batch:      mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
        MaxSamples: 1000,
    })
    if err != nil { log.Fatal(err) }

    fmt.Printf("loss=%.4f perplexity=%.2f tokens=%d samples=%d duration=%v\n",
        report.Metrics.Loss,
        report.Metrics.Perplexity,
        report.Metrics.Tokens,
        report.Metrics.Samples,
        report.Duration)

    data := core.JSONMarshal(report)
    core.WriteFile("/runs/eval-report.json", data.Value.([]byte), 0o644)
}
```

## With An Adapter

To evaluate a LoRA checkpoint without baking it into the base, set `AdapterPath`:

```go
report, err := mlx.RunModelEval(ctx, model, dataset, mlx.EvalConfig{
    Batch:       mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
    AdapterPath: "/runs/qwen3-8b-domain-a/step-005000",
})
fmt.Printf("with adapter %s: ppl=%.2f\n", report.Adapter.Path, report.Metrics.Perplexity)
```

The result's `Adapter` field records the adapter identity (rank, alpha, target keys) used for the eval.

## Quality Probes

Pluggable per-sample checks aggregate alongside loss. For example, an exact-match probe for a Q&A dataset:

```go
cfg := mlx.EvalConfig{
    Batch: mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
    QualityProbes: []mlx.EvalQualityProbe{
        {
            Name: "exact-match",
            Check: func(qctx mlx.EvalQualityContext) mlx.EvalQualityCheck {
                if core.Trim(qctx.Generated) == core.Trim(qctx.Expected) {
                    return mlx.EvalQualityCheck{Passed: true}
                }
                return mlx.EvalQualityCheck{
                    Passed: false,
                    Detail: fmt.Sprintf("got %q, want %q", qctx.Generated, qctx.Expected),
                }
            },
        },
    },
}

report, _ := mlx.RunModelEval(ctx, model, dataset, cfg)
for _, q := range report.Quality.Probes {
    fmt.Printf("probe %s: %d/%d passed\n", q.Name, q.Passed, q.Total)
}
```

## CI Gate

Eval reports are JSON-serialisable end-to-end. A CI step can fail when perplexity regresses against a baseline:

```bash
# baseline-ppl.txt holds the last known good perplexity
./bin/run-eval --output report.json
go run ./tools/check-perplexity --report report.json --baseline baseline-ppl.txt --tolerance 0.02
```

## Dataset Shape

`LoadJSONLDataset` accepts any of:

```jsonl
{"text": "raw completion text"}
{"prompt": "Q: ...", "response": "A: ..."}
{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

For chat-shaped datasets, configure the chat template:

```go
dataset, _ := mlx.LoadJSONLDataset(f, mlx.DatasetConfig{
    ChatTemplate: mlx.ChatTemplateConfig{
        Architecture: "qwen3",
        // Template/NoGenerationPrompt overrides if needed
    },
})
```

## See Also

- [Eval docs](../../docs/eval.md) — full reference
- [Attention probe](attention-probe.md) — extract per-head K vectors during eval
- [Distillation](../training/distill.md) — `EvalEvery` cadence shares this harness
- [GRPO](../training/grpo.md) — same shared harness
