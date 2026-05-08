---
title: GRPO
description: Group-relative policy optimisation for reasoning-style RL fine-tuning.
---

# GRPO — Group-Relative Policy Optimisation

GRPO is the RL training method behind reasoning models like DeepSeek-R1. For each prompt, the policy generates a *group* of candidate rollouts; rewards are scored against each, normalised within the group, and used to compute advantages without needing a value model. A KL term against a frozen reference policy keeps the training stable. The `mlx.RunGRPOReasoningTraining` orchestrator implements the loop; you supply the model-specific operations (rollout, reference log-prob, parameter update, eval) via a runner struct.

> The result type carries `Experimental: true`. The orchestrator is functional but the surface may evolve as we land more reward primitives and larger evals.

## Entry Point

```go
import (
    "context"

    mlx "dappco.re/go/mlx"
)

result, err := mlx.RunGRPOReasoningTraining(ctx, mlx.GRPORunner{
    PolicyInfo:       func(ctx context.Context) mlx.ModelInfo { return policyInfo },
    Tokenizer:        func(ctx context.Context) *mlx.Tokenizer { return tok },
    Rollout:          rolloutFn,
    ReferenceLogProb: referenceLogProbFn,
    ApplyUpdate:      applyUpdateFn,
    Evaluate:         evalFn,
    SaveCheckpoint:   saveFn,
}, dataset, mlx.GRPOConfig{
    GroupSize:        4,
    Epochs:           1,
    KLCoefficient:    0.04,
    AdvantageEpsilon: 1e-8,
    LearningRate:     1e-5,
    CheckpointDir:    "/runs/grpo-qwen3-math",
    CheckpointEvery:  100,
    EvalEvery:        500,
    RewardFuncs: []mlx.GRPORewardFunc{
        mlx.GRPORewardContainsAnswer(0.5),
        mlx.GRPORewardExactAnswer(1.0),
    },
})
```

## Runner Injection Points

| Field | Called per | Purpose |
|-------|------------|---------|
| `Rollout` | sample | Sample `GroupSize` candidate completions from the current policy |
| `ReferenceLogProb` | rollout | Score each rollout against the frozen reference for KL estimation |
| `ApplyUpdate` | step | Backpropagate the GRPO objective and step the optimiser |
| `Evaluate` | `EvalEvery` | Score policy quality on a held-out set |
| `SaveCheckpoint` | `CheckpointEvery` | Persist policy state |

The orchestrator handles dataset iteration, group bookkeeping, advantage normalisation, KL term assembly, and result aggregation.

## Reward Functions

A `GRPORewardFunc` takes a context (sample, rollout, expected answer) and returns a `GRPOReward { Value, Detail }`. Rewards from multiple funcs are summed.

Built-in helpers:

```go
mlx.GRPORewardContainsAnswer(weight) // weight applied if rollout contains expected answer substring
mlx.GRPORewardExactAnswer(weight)    // weight applied if rollout exactly equals expected answer
```

For reasoning data with fields like `{problem, thinking, answer}`, the helpers `GRPOSampleFromSFT(sample)` and `ExtractGRPOExpectedAnswer(sample)` adapt arbitrary SFT samples into GRPO's prompt/answer shape.

Custom rewards are just Go functions:

```go
parseAnswer := func(text string) (float64, bool) { /* ... */ }

mathReward := mlx.GRPORewardFunc(func(rctx mlx.GRPORewardContext) (mlx.GRPOReward, error) {
    expected, ok := parseAnswer(rctx.Expected)
    if !ok {
        return mlx.GRPOReward{}, nil
    }
    actual, ok := parseAnswer(rctx.Rollout.Text)
    if !ok {
        return mlx.GRPOReward{Value: 0, Detail: "unparseable rollout"}, nil
    }
    if math.Abs(expected - actual) < 1e-6 {
        return mlx.GRPOReward{Value: 1.0, Detail: "exact numeric match"}, nil
    }
    return mlx.GRPOReward{Value: 0.0}, nil
})
```

## Hyperparameters

| Field | Typical | Effect |
|-------|---------|--------|
| `GroupSize` | 4–16 | Number of rollouts per prompt; larger smooths advantage estimation |
| `KLCoefficient` | 0.01–0.1 | Higher keeps the policy closer to the reference (less drift, slower learning) |
| `AdvantageEpsilon` | 1e-8 | Numerical floor in advantage normalisation |
| `LearningRate` | 1e-6 – 1e-5 | RL is allergic to large LR; conservative defaults |
| `Epochs` | 1 | More epochs are unusual in RL; iterate by adding fresh prompts |

## Checkpointing & Resume

Same shape as [distillation](distillation.md):

```go
meta := mlx.NewGRPOCheckpointMetadata(path, cfg, result, latestUpdate)
if err := mlx.SaveGRPOCheckpointMetadata(path, meta); err != nil { ... }

// To resume:
cfg.ResumePath = "/runs/grpo-qwen3-math/checkpoint-500.json"
```

The result records `ResumedFrom` and continues from the persisted step count.

## Result

```go
type GRPOResult struct {
    Experimental       bool
    Policy             ModelInfo
    Config             GRPOConfig
    Metrics            GRPOMetrics             // mean reward, KL, advantage stats
    Updates            []GRPOUpdate            // per-step update history
    Checkpoints        []string
    CheckpointMetadata []GRPOCheckpointMetadata
    Evaluations        []GRPOEvalResult
    ResumePath         string
    ResumedFrom        *GRPOCheckpointMetadata
    Duration           time.Duration
}
```

JSON-serialisable end-to-end for diff and replay.

## See Also

- [`examples/training/grpo.md`](../examples/training/grpo.md) — end-to-end walkthrough
- [Distillation](distillation.md) — sibling pipeline with the same runner shape
- [Eval](eval.md) — `EvalEvery` cadence shares this harness
- [Training](training.md) — base LoRA fine-tuning is a typical pre-RL warm start
