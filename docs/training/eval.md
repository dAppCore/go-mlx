<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# eval.go — dataset-native evaluation

**Package**: `dappco.re/go/mlx`
**File**: `go/eval.go` (plus `eval_darwin.go` / `eval_stub.go`, `fast_eval.go`)

## What this is

The **evaluation runner** — score a model against a dataset, emit a structured report. Used as:

- Mid-training validation (called from SFT / GRPO / Distill at `CheckpointInterval`)
- Standalone "is this checkpoint better than the last one?" comparison
- Benchmark harness for the wider eval suite

`fast_eval.go` is the optimised path — batched, parallelised, prefill-only where possible.

## EvalConfig

```go
type EvalConfig struct {
    Dataset       DatasetStream
    Model         string             // model path
    Adapter       string             // optional adapter path
    Metrics       []EvalMetric       // ppl, accuracy, exact-match, judge, custom
    Judge         JudgeFunc          // for semantic eval
    MaxSamples    int                // 0 = all
    BatchSize     int
    ContextLength int
    ProbeSink     inference.ProbeSink
}
```

## Metrics

```
EvalMetricPerplexity   — token-level cross-entropy over the dataset
EvalMetricAccuracy     — exact-match accuracy on classification-style samples
EvalMetricExactMatch   — string equality on generated vs target
EvalMetricJudge        — LLM-judge semantic score (uses Judge callback)
EvalMetricCustom       — user-supplied scoring function via labels
```

Each metric is its own pass through the dataset (or sub-pass for batched runs).

## EvalReport

```go
type EvalReport struct {
    Version       int                          // EvalReportVersion = 1
    Model         inference.ModelIdentity
    Adapter       inference.AdapterIdentity
    Runtime       inference.RuntimeIdentity
    Dataset       string
    SampleCount   int

    Perplexity    *float64
    Accuracy      *float64
    ExactMatch    *float64
    JudgeScore    *float64
    CustomScores  map[string]float64

    DurationMs    int64
    Labels        map[string]string
}
```

Pointer fields so "metric not run" is distinguishable from "metric ran and produced 0".

## Fast path

`fast_eval.go` uses prefill-only inference where the metric allows — perplexity in particular only needs the full forward pass on prompts, not autoregressive decoding. This makes eval 10-50x faster than naïve generate-and-compare.

## Used by

- `sft.go` / `grpo.go` / `distill.go` — mid-training validation
- Vi training pipeline — sweep through reasoning + capability + safety evals
- LARQL eval harness — pre/post-SFT model comparison
- Lemma vertical stack — eval suite for distillation cascade

## Probes

`ProbeEventEntropy`, `ProbeEventLayerCoherence` emitted per sample so research-grade evaluation captures the cognitive shape, not just the score.

## Status

Production. Most metric types implemented; custom-metric DSL planned for power users who need per-domain scoring.

## Related

- [sft.md](sft.md) / [grpo.md](grpo.md) / [distill.md](distill.md) — training that calls eval at intervals
- `go/dataset_stream.go` — input shape
- `../../../go-inference/docs/inference/probe.md` — probe events emitted
- `../../../go-inference/docs/inference/capability.md` — `CapabilityEvaluation` flag
- `../../../go-ml/docs/scoring/` (planned) — go-ml's higher-level scoring engine builds on this
