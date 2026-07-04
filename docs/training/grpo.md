<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# grpo.go — Group Relative Policy Optimisation (reasoning training)

**Package**: `dappco.re/go/mlx`
**File**: `go/grpo.go`
**Status**: experimental

## What this is

The **GRPO** training loop — group relative policy optimisation for reasoning models. The technique that DeepSeek-R1 popularised: sample multiple completions per prompt, score with a reward model (or programmatic checker), update the policy to favour higher-reward completions relative to the group mean.

Used by Lemma reasoning training and the Vi reasoning extension (per `project_lemma_vertical_stack.md`).

## GRPOConfig

```go
type GRPOConfig struct {
    Dataset            DatasetStream   // reasoning prompts
    BaseModel          string          // path
    Adapter            LoRAConfig      // adapter config to attach
    BatchSize          int             // prompts per step
    RolloutCount       int             // completions per prompt (group size, typical 8-16)
    MaxTokens          int             // per-rollout cap
    Temperature        float32         // rollout temp (typical 0.7-1.0)

    RewardFn           RewardFunction  // returns float64 reward per completion
    KLBeta             float64         // KL penalty against reference (typical 0.01-0.1)
    ClipEpsilon        float64         // PPO-style clipping (typical 0.2)

    LearningRate       float32
    WarmupSteps        int
    MaxSteps           int
    CheckpointDir      string
    CheckpointInterval int
    ProbeSink          inference.ProbeSink
}
```

## RewardFunction

```go
type RewardFunction func(
    ctx context.Context,
    prompt string,
    completion string,
    sample DatasetSample,
) (float64, error)
```

Programmatic (regex/AST checks for code/math) or model-based (LLM judge call). Reward in [0, 1] or wider — GRPO normalises within the group, so absolute scale doesn't matter as long as it's consistent.

## Algorithm sketch

```
for step in 1..MaxSteps:
    batch = dataset.Next() × BatchSize
    for prompt in batch:
        completions = [generate(prompt, T=Temperature) for _ in RolloutCount]
        rewards     = [RewardFn(prompt, c) for c in completions]
        advantages  = (rewards - mean(rewards)) / std(rewards)
        for i in 1..RolloutCount:
            loss = -advantage[i] * logprob(completions[i] | prompt)
                   + KLBeta * KL(policy, ref)
            loss = clip(loss, ClipEpsilon)
            backprop(loss)
    Adam step
```

Reasoning-specific tweaks: longer rollouts (1024-4096 tokens), lower temperatures than RLHF (0.7 vs 1.0), reward functions that check intermediate reasoning AND final answer.

## Checkpointing

`GRPOCheckpointMetadataVersion = 1`. Checkpoints record: current step, base model hash, adapter state, optimiser moments, recent rollout statistics (avg reward, KL divergence, completion length distribution).

## Status

Implementation complete; production use pending the reward-function library landing (`go-ml/judge.go` provides the LLM-judge primitive; programmatic checkers per task domain TBD).

## Used by

- Lemma reasoning training (production pipeline)
- Vi reasoning extension (planned)
- Distillation cascade — GRPO on the student post-distillation

## Related

- [sft.md](sft.md) — SFT often precedes GRPO (warm-start the adapter)
- [distill.md](distill.md) — distillation often precedes GRPO (compress then reason)
- [eval.md](eval.md) — reasoning-quality eval suite for checkpoint validation
- `../../../go-inference/docs/inference/capability.md` — `CapabilityGRPO` flag
- `project_lemma_vertical_stack.md` — Lemma training architecture
