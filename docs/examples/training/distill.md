# Knowledge Distillation

Train a student model to imitate a larger teacher's output distribution. The student sees the teacher's full softmax, not just the top-1 token, so it learns the teacher's *uncertainty* — which generalises better than hard-label fine-tuning on the same data.

## Conceptual Setup

```
teacher (frozen, e.g. Qwen3-32B)  --logits→  KL divergence  ←logits--  student (trainable, e.g. Qwen3-8B)
                                                  ↓
                                          backprop to student
```

You bring the teacher logits (computed once per batch, cached for multi-epoch runs) and the student forward+backward; the orchestrator handles batching, loss assembly, checkpoint cadence, and resumption.

## Minimal Run

```go
package main

import (
    "context"
    "log"
    "os"

    core "dappco.re/go"
    mlx "dappco.re/go/mlx"
)

func main() {
    ctx := context.Background()

    // Load the JSONL dataset.
    f, err := os.Open("/data/distill-corpus.jsonl")
    if err != nil { log.Fatal(err) }
    defer f.Close()
    dataset, err := mlx.LoadJSONLDataset(f, mlx.DatasetConfig{})
    if err != nil { log.Fatal(err) }

    runner := mlx.DistillRunner{
        TeacherInfo: func(ctx context.Context) mlx.ModelInfo { return teacher.Info() },
        StudentInfo: func(ctx context.Context) mlx.ModelInfo { return student.Info() },
        Tokenizer:   func(ctx context.Context) *mlx.Tokenizer { return student.Tokenizer() },
        BuildBatches: func(ctx context.Context, ds mlx.SFTDataset, bcfg mlx.DatasetBatchConfig) ([]mlx.SFTBatch, error) {
            return student.BuildBatches(ds, bcfg)
        },
        TeacherLogits: func(ctx context.Context, batch mlx.DistillBatch) (mlx.DistillLogits, error) {
            return teacher.LogitsForBatch(ctx, batch)
        },
        StudentLogits: func(ctx context.Context, batch mlx.DistillBatch, teacherL mlx.DistillLogits) (mlx.DistillLogits, error) {
            return student.LogitsForBatch(ctx, batch)
        },
        ApplyLoss: func(ctx context.Context, batch mlx.DistillBatch, loss mlx.DistillLoss) error {
            return student.BackwardAndStep(ctx, batch, loss)
        },
        Evaluate: func(ctx context.Context, ectx mlx.DistillEvalContext) (mlx.DistillEvalResult, error) {
            return student.EvaluateHeldOut(ctx, ectx)
        },
        SaveCheckpoint: func(ctx context.Context, cctx mlx.DistillCheckpointContext) error {
            return student.SaveCheckpoint(ctx, cctx)
        },
        TeacherCache: mlx.NewMemoryDistillLogitCache(),
    }

    cfg := mlx.DistillConfig{
        Batch:           mlx.DatasetBatchConfig{BatchSize: 4, MaxSeqLen: 2048},
        Epochs:          3,
        Temperature:     2.0,
        Loss:            mlx.DistillLossKL,
        LearningRate:    1e-4,
        CheckpointDir:   "/runs/qwen32b-to-qwen8b",
        CheckpointEvery: 500,
        EvalEvery:       1000,
    }

    result, err := mlx.RunKnowledgeDistillation(ctx, runner, dataset, cfg)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("done: %d tokens trained, mean loss %.4f, %d checkpoints",
        result.Metrics.Tokens, result.Metrics.MeanLoss, len(result.Checkpoints))

    // Persist the result for diff/replay.
    data := core.JSONMarshal(result)
    core.WriteFile("/runs/qwen32b-to-qwen8b/result.json", data.Value.([]byte), 0o644)
}
```

## Choosing Loss + Temperature

| Setting | When |
|---------|------|
| `DistillLossKL`, `Temperature: 2.0` | Default — preserves teacher distribution shape |
| `DistillLossKL`, `Temperature: 4.0+` | Teacher is much larger / much more confident; soften further |
| `DistillLossSoftCrossEntropy` | Equivalent gradient direction to KL when teacher is fixed; sometimes nicer numerically |

`Temperature` divides both teacher and student logits before softmax. The orchestrator scales the loss by `T²` so gradients stay comparable across temperatures.

## Multi-Epoch Caching

The teacher forward pass is the dominant cost for an asymmetric pair (e.g. 32B teacher → 8B student). The default `MemoryDistillLogitCache` keys batches via `DistillBatchCacheKey(batch)` and returns cached logits on subsequent epochs. For corpora that don't fit in RAM, implement `DistillTeacherLogitCache` against on-disk storage.

## Resumption

To resume from a checkpoint:

```go
cfg.ResumePath = "/runs/qwen32b-to-qwen8b/checkpoint-2500.json"
result, err := mlx.RunKnowledgeDistillation(ctx, runner, dataset, cfg)
// result.ResumedFrom is populated; the run continues from the recorded sample offset.
```

## See Also

- [Distillation docs](../../docs/distillation.md) — full reference for runner shape, config, and result
- [LoRA fine-tuning](lora-finetune.md) — supervised baseline
- [GRPO](grpo.md) — RL alternative once SFT/KD has converged
- [Eval](../eval/perplexity.md) — measure student quality on a held-out set
