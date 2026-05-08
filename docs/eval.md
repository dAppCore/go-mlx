---
title: Eval
description: Dataset-native perplexity, quality probes, and structured eval reports.
---

# Eval

`go-mlx` ships a Go-native evaluation harness for measuring how a base model or a LoRA-adapted model performs on a JSONL dataset. It computes loss and perplexity over a stream of samples, optionally runs pluggable quality probes (e.g. exact-match, contains-answer), and returns a structured `EvalReport` you can serialise to JSON.

## Two Entry Points

```go
import (
    "context"

    mlx "dappco.re/go/mlx"
)
```

**High-level — bind directly to a loaded model:**

```go
report, err := mlx.RunModelEval(ctx, model, dataset, mlx.EvalConfig{
    Batch: mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
})
```

`RunModelEval` is the convenience path: it wires the model's tokenizer, info, and forward-pass batch evaluator into the runner machinery automatically.

**Low-level — runner injection:**

```go
report, err := mlx.RunDatasetEval(ctx, mlx.EvalRunner{
    Info:          func(ctx context.Context) mlx.ModelInfo { return modelInfo },
    Tokenizer:     func(ctx context.Context) *mlx.Tokenizer { return tok },
    LoadAdapter:   func(ctx context.Context, path string) (mlx.LoRAAdapterInfo, error) { ... },
    BuildBatches:  func(ctx context.Context, ds mlx.SFTDataset, b mlx.DatasetBatchConfig) ([]mlx.SFTBatch, error) { ... },
    EvaluateBatch: func(ctx context.Context, batch mlx.SFTBatch) (mlx.EvalBatchMetrics, error) { ... },
}, dataset, cfg)
```

Use `RunDatasetEval` when you want to evaluate against a remote runner, a non-MLX backend, a mocked teacher, or any setup where the model isn't a local `*mlx.Model`.

## Config

```go
type EvalConfig struct {
    Batch         DatasetBatchConfig  // BatchSize, MaxSeqLen, SequencePacking, NoEOS
    AdapterPath   string              // optional LoRA adapter to load before eval
    MaxSamples    int                 // 0 = entire dataset
    QualityProbes []EvalQualityProbe  // optional pluggable checks
}
```

The dataset is any `mlx.SFTDataset` — typically a `JSONLDataset` loaded with `mlx.LoadJSONLDataset(reader, mlx.DatasetConfig{...})`. Samples can be plain `text`, `prompt/response`, ShareGPT `conversations`, or chat `messages` arrays — the loader normalises them all.

## Quality Probes

Probes are user-supplied functions that inspect each sample and return a check result:

```go
cfg.QualityProbes = []mlx.EvalQualityProbe{
    {
        Name: "contains-expected-answer",
        Check: func(qctx mlx.EvalQualityContext) mlx.EvalQualityCheck {
            generated := qctx.Generated
            expected  := qctx.Expected
            return mlx.EvalQualityCheck{
                Passed: core.Contains(generated, expected),
                Detail: "checked substring match",
            }
        },
    },
}
```

Probe results are aggregated into `EvalQualityReport` (per-probe pass/fail counts plus per-sample detail) inside the returned `EvalReport`.

## Report Shape

```go
type EvalReport struct {
    Version   int               // EvalReportVersion (currently 1)
    ModelInfo ModelInfo
    Adapter   LoRAAdapterInfo   // populated when AdapterPath was set
    Config    EvalConfig
    Metrics   EvalMetrics       // Loss, Perplexity, Tokens, Samples, Batches
    Quality   EvalQualityReport
    Duration  time.Duration
}
```

`EvalReport` is JSON-serialisable end-to-end, so checkpoint comparisons or CI harnesses can persist and diff runs:

```go
data := core.JSONMarshal(report)
core.WriteFile("eval-report.json", data.Value.([]byte), 0o644)
```

## Dataset Format

`LoadJSONLDataset` accepts JSONL with any of these per-line shapes:

```jsonl
{"text": "raw completion text"}
{"prompt": "Q: ...", "response": "A: ..."}
{"instruction": "...", "input": "...", "output": "..."}
{"problem": "...", "thinking": "...", "answer": "..."}
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."}]}
{"conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

A chat template (`ChatTemplateConfig`) on the `DatasetConfig` controls how `messages` and `conversations` arrays are rendered into a single text stream before tokenisation.

## See Also

- [`examples/eval/perplexity.md`](../examples/eval/perplexity.md) — end-to-end perplexity walkthrough
- [`examples/eval/attention-probe.md`](../examples/eval/attention-probe.md) — extracting per-layer post-RoPE K vectors
- [Distillation](distillation.md) — `EvalEvery` cadence integrates the same eval harness into KD training loops
- [GRPO](grpo.md) — same `EvalEvery` cadence for RL training
