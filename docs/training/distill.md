<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# distill.go — knowledge distillation

**Package**: `dappco.re/go/mlx`
**File**: `go/distill.go`

## What this is

The **knowledge distillation** loop — train a small "student" model to match the logits of a large "teacher" model. Output: a LoRA adapter (on the student) that captures the teacher's behaviour while running 5-10x faster.

This is the Vi training thesis: distil a 26B Gemma 4 into a 2B base + adapter so the production model is small enough for a phone but inherits the 26B's behavior.

Without-training-data variant: distillation can run on **GPT-OSS-style** open teacher endpoints — feed prompts, capture teacher logits, train student against captured logits. No labelled dataset needed; the teacher IS the supervision. See `design_models_as_queryable_databases.md`.

## DistillConfig

```go
type DistillConfig struct {
    Dataset       DatasetStream      // prompts (responses optional — teacher fills in)
    StudentModel  string             // base student path
    StudentAdapter LoRAConfig        // adapter config to attach to student
    TeacherModel  string             // teacher path OR endpoint URL
    TeacherIsLocal bool              // local load vs remote OpenAI-compat

    Temperature       float32        // distillation softness (1.0-3.0 typical)
    LossType          string         // "kl" | "mse" | "ce_soft"
    AlphaHard         float32        // mix in hard-label CE loss (0 = pure distillation)

    BatchSize         int
    MicroBatchSize    int
    LearningRate      float32
    MaxSteps          int
    CheckpointInterval int
    CheckpointDir     string
    ProbeSink         inference.ProbeSink

    SyncTeacher       sync.Locker    // when teacher is shared across processes
}
```

## DistillCheckpointMetadataVersion

`= 1`. Checkpoint metadata includes teacher identity (so resume after teacher version change fails fast) + student identity + step + loss.

## Loss

```
soft_loss = KL(softmax(student / T)  ‖  softmax(teacher / T)) × T²
hard_loss = CE(student_pred, true_label)   if sample has true response
loss      = (1 - AlphaHard) * soft_loss + AlphaHard * hard_loss
```

Pure distillation: `AlphaHard = 0`. Mixed: `AlphaHard = 0.5` — half "match teacher logits", half "match true labels when available".

## Teacher integration

- **Local teacher** — `TeacherIsLocal: true` + local model path → loaded into Metal alongside the student. Teacher forward pass runs synchronously per batch.
- **Remote teacher** — `TeacherIsLocal: false` + endpoint URL → student worker batches prompts and calls the teacher's `/v1/chat/completions` with logit-return. Cached locally to amortise cost.

Remote teacher path lets you distill from a teacher you can't run (e.g., GPT-4-class API) into a model you can run on your laptop. The cost is one teacher API call per training step × prompt-count — manageable for ~10k-step training runs.

## Sync.Locker on teacher

When multiple distillation workers share one local teacher (multi-student distillation, where different students learn different aspects), the teacher load needs synchronisation. The Locker is the consumer-supplied sync primitive.

## Status

Production for dense models. Sample workflows in `examples/`. Vi training is the primary live consumer.

## Used by

- Vi training pipeline — distill 26B Gemma 4 → Vi base
- Lemma model family — distill from larger Lemma into the LEK-fine-tuned compact

## Related

- [sft.md](sft.md) — supervised fine-tuning (alternative path when labelled data exists)
- [grpo.md](grpo.md) — reasoning training (often runs post-distillation)
- [lora_adapter.md](lora_adapter.md) — adapter shape produced
- [model_merge.md](model_merge.md) — alternative compression via interpolation
- `project_vi_training_plan.md` — Vi training architecture
- `design_models_as_queryable_databases.md` — distillation-without-training-data thesis
- `../../../go-inference/docs/inference/capability.md` — `CapabilityDistillation` flag
