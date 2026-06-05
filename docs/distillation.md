---
title: Knowledge Distillation
description: Train a smaller student model against a teacher's logits using KL or soft cross-entropy loss.
---

# Knowledge Distillation

go-mlx provides a Go-native knowledge distillation pipeline. A teacher model produces target logit distributions; a student model is trained to match them via KL divergence or soft cross-entropy. Checkpoints, eval cadence, and an in-memory teacher logit cache are first-class.

The pipeline mirrors the runner-injection pattern used by [`Eval`](eval.md) and [`GRPO`](grpo.md): you pass in functions that produce teacher logits, run student updates, and evaluate. The orchestrator handles batching, loss computation, checkpoint persistence, and resumption.

## Entry Point

```go
import (
    "context"

    mlx "dappco.re/go/mlx"
)

result, err := mlx.RunKnowledgeDistillation(ctx, mlx.DistillRunner{
    TeacherInfo:    func(ctx context.Context) mlx.ModelInfo { return teacherInfo },
    StudentInfo:    func(ctx context.Context) mlx.ModelInfo { return studentInfo },
    Tokenizer:      func(ctx context.Context) *mlx.Tokenizer { return tok },
    BuildBatches:   buildBatchesFn,
    TeacherLogits:  teacherLogitsFn,    // produces target distributions
    StudentLogits:  studentLogitsFn,    // student forward pass given teacher logits
    ApplyLoss:      applyLossFn,        // backward + optimiser step
    Evaluate:       evalFn,             // optional, runs on EvalEvery cadence
    SaveCheckpoint: saveFn,             // optional, runs on CheckpointEvery cadence
    TeacherCache:   mlx.NewMemoryDistillLogitCache(),
}, dataset, mlx.DistillConfig{
    Batch:           mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
    Epochs:          3,
    Temperature:     2.0,
    Loss:            mlx.DistillLossKL,
    LearningRate:    1e-4,
    CheckpointDir:   "/runs/distill-qwen3-to-qwen3-mini",
    CheckpointEvery: 500,
    EvalEvery:       1000,
})
```

`RunDistillation` is an alias for `RunKnowledgeDistillation` — same orchestrator, different name for narration in higher-level harnesses.

## Loss Kinds

```go
const (
    DistillLossKL                DistillLossKind = "kl"
    DistillLossSoftCrossEntropy  DistillLossKind = "soft_cross_entropy"
)
```

| Kind | Formula | When to use |
|------|---------|-------------|
| `DistillLossKL` | `KL(teacher_softmax(T) || student_softmax(T)) * T²` | Standard distillation; preserves teacher's full distribution shape |
| `DistillLossSoftCrossEntropy` | `-Σ teacher_softmax(T) * student_log_softmax(T)` | Equivalent gradient direction to KL when teacher is fixed; sometimes numerically nicer |

Both losses scale by `Temperature²` to keep gradients comparable across temperatures. `Temperature` is applied to both teacher and student logits before the softmax.

## DistillBatchLoss (Standalone)

If you want to compute a distillation loss outside the runner machinery (for unit tests, ad-hoc analysis, or a custom training loop), call:

```go
loss, err := mlx.DistillationBatchLoss(teacher, student, mask, cfg)
fmt.Printf("KL=%.4f, soft_xent=%.4f, teacher_entropy=%.4f, tokens=%d\n",
    loss.KL, loss.SoftCrossEntropy, loss.TeacherEntropy, loss.Tokens)
```

Each `DistillLoss` carries the chosen scalar (`Value`), both candidate scalars (`KL` and `SoftCrossEntropy`), the teacher's mean entropy (a useful signal for how confident the teacher is on this batch), the token count contributing to the average, and the temperature/kind used.

## Teacher Logit Cache

The teacher forward pass is the dominant cost when the teacher is much larger than the student. `DistillTeacherLogitCache` lets you cache teacher logits keyed by batch identity (`DistillBatchCacheKey(batch)`) so a multi-epoch run pays the teacher cost once.

```go
runner.TeacherCache = mlx.NewMemoryDistillLogitCache()
```

The default in-memory cache is fine for runs that fit in RAM. For larger corpora, implement the `DistillTeacherLogitCache` interface against on-disk storage.

## Checkpointing & Resume

When `CheckpointDir` and `CheckpointEvery` are set, the runner calls your `SaveCheckpoint` callback at the configured cadence and writes a `DistillCheckpointMetadata` JSON record alongside it:

```go
meta := mlx.NewDistillCheckpointMetadata(path, cfg, result, latestLoss, epoch)
if err := mlx.SaveDistillCheckpointMetadata(path, meta); err != nil { ... }
```

To resume, set `cfg.ResumePath` to the metadata file. `LoadDistillCheckpointMetadata` rehydrates the run, the orchestrator skips already-trained samples, and the result records `ResumedFrom`.

## Result

```go
type DistillResult struct {
    Teacher            ModelInfo
    Student            ModelInfo
    Config             DistillConfig
    Metrics            DistillMetrics              // tokens, samples, batches, mean loss
    Losses             []DistillLoss               // per-step loss history
    Checkpoints        []string                    // saved checkpoint paths
    CheckpointMetadata []DistillCheckpointMetadata
    Evaluations        []DistillEvalResult         // results from EvalEvery cadence
    ResumePath         string
    ResumedFrom        *DistillCheckpointMetadata
    Duration           time.Duration
}
```

The full result is JSON-serialisable so a downstream harness can persist and diff runs.

## Simple Self-Distillation

`RunSimpleSelfDistillation` implements the native SSD data-generation and SFT
core without Python. It samples raw responses from the frozen model with
`SampleMaxTokens`, non-unit `SampleTemperature`, `SampleTopP`, `SampleTopK`,
`SampleMinP`, and `RepetitionPenalty`, then trains those raw prompt/response rows
through the existing SFT path. `DecodeTemperature` is carried separately for the
post-SSD decode configuration.

When `SimpleSelfDistillationRunner.ModelInfo` is set, the generated SFT config
uses model-specific normalisation before training. `Model.RunSimpleSelfDistillation`
sets it automatically, so Gemma-4 SSD runs reuse the same LoRA target policy as
normal Gemma-4 SFT.

`DefaultSimpleSelfDistillationConfig()` mirrors the upstream ml-ssd
data-generation defaults: Qwen3-4B/rStar-Coder-style sampling at temperature
`1.5`, `top_k=20`, `top_p=0.8`, repetition penalty `1.0`, and `65536` sample
tokens.

The ml-ssd data-generation post-process is available through
`FilterShortestPercent`. A value of `10` drops the shortest generation decile
from the SFT dataset after raw sampling while preserving the full raw sample
record in the result for auditability.

`RunSimpleSelfDistillationCodeBenchmark` is the native code-eval seam for
LiveCodeBench-style checks. It samples `NRepeat` candidate solutions per task
with a caller-provided `GenerateConfig`, delegates code execution to the
runner's `RunTests` callback, extracts and post-processes fenced code blocks in
Go, aggregates candidate pass rate plus LiveCodeBench pass@k metrics (including
per-difficulty metrics when labels are present), and can write the JSON report to
`OutputPath`. The unavoidable language-specific execution boundary stays behind
the callback; the go-mlx harness itself does not import or shell out to Python.
When `Seeds` is set, each repeat receives `Seeds[0]+repeat` in the forwarded
`GenerateConfig`, matching the upstream eval loop while leaving ad hoc callers
free to provide their own sampler behaviour.

Use `LoadSimpleSelfDistillationLiveCodeBenchV6JSONL` or its file variant to
load LiveCodeBench-style JSONL and keep the v6 contest-date window natively in
Go. The broader `LoadSimpleSelfDistillationCodeBenchmarkJSONL` helper remains
available for other code benchmark datasets.

`DefaultSimpleSelfDistillationCodeBenchmarkConfig()` mirrors the upstream eval
shape: `LiveCodeBench-v6`, `n_repeat=20`, `max_tokens=32768`, temperature `0.6`,
`top_p=0.95`, `top_k=20`, `min_p=0.0`, and seeds `0,1234,1234,1234`.
`SimpleSelfDistillationRecipes()` describes the released SimpleSD-4B-instruct,
SimpleSD-4B-thinking, and SimpleSD-30b-a3b-instruct parity recipes for native
reproduction runs.

The `cmd/mlx` surface exposes two no-Python helpers for these artefacts:
`ssd-recipes -json` prints the native recipe defaults, and `ssd-eval -json
-samples livecodebench.jsonl -output results/lcb-report.json -n-repeat 10
-sampling-params "temperature=0.9,top_p=0.8,top_k=20,max_tokens=65536"`
loads LiveCodeBench-style JSONL, applies the v6 date filter, and emits the
normalised eval plan used by `RunSimpleSelfDistillationCodeBenchmark`.

## See Also

- [`examples/training/distill.md`](../examples/training/distill.md) — end-to-end walkthrough
- [Training](training.md) — supervised LoRA fine-tuning, the typical baseline before KD
- [Eval](eval.md) — the same `EvalEvery` cadence used here is the eval harness
- [GRPO](grpo.md) — sibling RL pipeline with the same runner shape
