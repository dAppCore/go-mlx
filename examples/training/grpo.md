# GRPO — Reasoning RL

Group-relative policy optimisation, the RL method used in DeepSeek-R1-style training. For each prompt, sample a group of rollouts, score them with reward functions, normalise advantages within the group (no value model needed), and update the policy under a KL constraint against a frozen reference.

## Conceptual Loop

```
prompt → policy → [4 rollouts] → reward funcs → group-normalised advantages
                       ↓
                 reference (frozen)         ←── KL divergence
                       ↓
                 ApplyUpdate(policy)
```

## Minimal Run

```go
package main

import (
    "context"
    "log"
    "os"

    mlx "dappco.re/go/mlx"
)

func main() {
    ctx := context.Background()

    f, err := os.Open("/data/grpo-math.jsonl")
    if err != nil { log.Fatal(err) }
    defer f.Close()
    dataset, err := mlx.LoadJSONLDataset(f, mlx.DatasetConfig{})
    if err != nil { log.Fatal(err) }

    runner := mlx.GRPORunner{
        PolicyInfo: func(ctx context.Context) mlx.ModelInfo { return policy.Info() },
        Tokenizer:  func(ctx context.Context) *mlx.Tokenizer { return policy.Tokenizer() },
        Rollout: func(ctx context.Context, req mlx.GRPORolloutRequest) ([]mlx.GRPORollout, error) {
            return policy.GenerateGroup(ctx, req)
        },
        ReferenceLogProb: func(ctx context.Context, req mlx.GRPORolloutRequest, r mlx.GRPORollout) (float64, error) {
            return reference.LogProb(ctx, req.Prompt, r.Tokens)
        },
        ApplyUpdate: func(ctx context.Context, update mlx.GRPOUpdate) error {
            return policy.ApplyGRPOUpdate(ctx, update)
        },
        Evaluate: func(ctx context.Context, ectx mlx.GRPOEvalContext) (mlx.GRPOEvalResult, error) {
            return policy.EvaluateRL(ctx, ectx)
        },
        SaveCheckpoint: func(ctx context.Context, cctx mlx.GRPOCheckpointContext) error {
            return policy.SaveCheckpoint(ctx, cctx)
        },
    }

    cfg := mlx.GRPOConfig{
        GroupSize:        4,
        Epochs:           1,
        KLCoefficient:    0.04,
        AdvantageEpsilon: 1e-8,
        LearningRate:     5e-6,
        CheckpointDir:    "/runs/qwen3-8b-grpo-math",
        CheckpointEvery:  100,
        EvalEvery:        500,
        RewardFuncs: []mlx.GRPORewardFunc{
            mlx.GRPORewardContainsAnswer(0.3), // partial credit for substring match
            mlx.GRPORewardExactAnswer(1.0),    // full credit for exact match
        },
    }

    result, err := mlx.RunGRPOReasoningTraining(ctx, runner, dataset, cfg)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("done: %d updates, mean reward %.3f, mean KL %.4f",
        len(result.Updates), result.Metrics.MeanReward, result.Metrics.MeanKL)
}
```

## Custom Reward Function

For domain-specific rewards (math correctness, code execution success, citation match), write your own:

```go
import "math"

parseAnswer := func(text string) (float64, bool) { /* extract number */ }

mathReward := mlx.GRPORewardFunc(func(rctx mlx.GRPORewardContext) (mlx.GRPOReward, error) {
    expected, ok := parseAnswer(rctx.Expected)
    if !ok {
        return mlx.GRPOReward{}, nil
    }
    actual, ok := parseAnswer(rctx.Rollout.Text)
    if !ok {
        return mlx.GRPOReward{Value: 0, Detail: "unparseable rollout"}, nil
    }
    if math.Abs(expected-actual) < 1e-6 {
        return mlx.GRPOReward{Value: 1.0, Detail: "exact match"}, nil
    }
    return mlx.GRPOReward{Value: 0.0}, nil
})

cfg.RewardFuncs = append(cfg.RewardFuncs, mathReward)
```

Rewards from multiple funcs are summed before group normalisation.

## Working With SFT Data

If your dataset is already SFT-shaped (`{problem, thinking, answer}` or `{instruction, output}`), the helpers convert per-sample:

```go
sample, _, _ := dataset.Next()
gsample := mlx.GRPOSampleFromSFT(sample)
expected := mlx.ExtractGRPOExpectedAnswer(sample)
```

## Tuning Knobs

| Knob | What it does | Typical range |
|------|--------------|---------------|
| `GroupSize` | Rollouts per prompt | 4–16 |
| `KLCoefficient` | Penalty for drifting from reference | 0.01 – 0.1 |
| `LearningRate` | Step size | 1e-6 – 1e-5 |
| `Temperature` (in policy.GenerateGroup) | Sampling spread per rollout | 0.7 – 1.0 |

Higher `KLCoefficient` keeps the policy stable but slows learning; lower lets it explore more aggressively but risks reward hacking. `LearningRate` should be ~10× smaller than supervised LoRA fine-tuning — RL is gradient-noise-allergic.

## Notes

- The result's `Experimental: true` flag reflects that the GRPO orchestrator is functional but the surface may evolve as more reward primitives and evals land
- The reference model is loaded by your runner; the orchestrator never touches its weights
- `MaxSamples` caps the per-epoch sample count if your dataset is much larger than you want to traverse in one run

## See Also

- [GRPO docs](../../docs/grpo.md) — runner shape, hyperparameters, result schema
- [Distillation](distill.md) — sibling pipeline with the same runner contract
- [LoRA fine-tuning](lora-finetune.md) — typical pre-RL warm start
- [Eval](../eval/perplexity.md) — measure pass-rate on held-out reasoning problems
