<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# sft.go — supervised fine-tuning

**Package**: `dappco.re/go/mlx`
**File**: `go/sft.go` (plus `sft_darwin.go` / `sft_stub.go`)

## What this is

The **supervised fine-tuning loop** — labelled prompt/response pairs in, fine-tuned LoRA adapter out. Native AdamW optimiser, Metal-side gradient computation, optional gradient accumulation, checkpoint save/load.

This is the loop that fine-tunes Vi from Mattermost conversations (per `project_vi_training_plan.md`). It also serves as the base for distillation + GRPO — those files reuse the same training scaffolding with different loss functions.

## SFTSample

```go
type SFTSample struct {
    Prompt   string             // user prompt
    Response string             // assistant target response
    Text     string             // alternative — raw text (continuation pretraining)
    Meta     map[string]string  // routing / filtering
}
```

A sample is either `Prompt+Response` (instruct SFT) or `Text` (continuation SFT), not both. The loss masks differ — instruct SFT masks the prompt tokens; continuation SFT trains on all tokens.

## SFTDataset

```go
type SFTDataset interface {
    Next() (SFTSample, bool, error)
}
```

Same pull shape as `inference.DatasetStream`. The two interfaces coexist because go-mlx defines its own typed sample shapes locally; a wrapper would also satisfy `inference.DatasetStream`.

## SFTConfig

Controls: dataset, base model, LoRA config (Rank/Alpha/TargetKeys), batch size, micro-batch size, gradient accumulation, learning rate (typically 1e-4 to 2e-4 for adapter SFT), warmup steps, max steps, eval interval, eval dataset, checkpoint interval, checkpoint dir, KV encoding for any KV snapshots written during training.

## Loss

Standard next-token cross-entropy with optional prompt masking. Operates on tokenised batches; the tokenizer lives in the loaded model.

## Optimiser

AdamW (`go/internal/metal/optim.go`). Decoupled weight decay; default `weight_decay = 0.01`; betas `(0.9, 0.999)`.

## Checkpointing

Each checkpoint emits:

- LoRA adapter package (`adapter_config.json` plus `adapter.safetensors`) -- the
  actual fine-tune weights
- Optimiser state (m, v moments per parameter) -- for resume-from-checkpoint
- Step metadata (current step, loss, learning rate, elapsed)
- Eval report (if interval hit)

`SFTCheckpointMetadataVersion` constant tracks the on-disk schema; old checkpoints fail-fast on load.

## Native vs stub

`sft_darwin.go` holds the Metal-side gradient computation + Adam steps. `sft_stub.go` returns a fixed error on non-darwin builds (training is darwin-only — the Linux/ROCm path is `go-rocm` planned).

## Status

Production for dense models (Gemma 3/4, Qwen 3, Llama 3). MoE training (MiniMax M2) pending Phase 1 forward path. The 8B-class supports SFT comfortably on 96GB; 27B-class requires aggressive gradient checkpointing.

## Used by

- Vi training pipeline (per `project_vi_training_plan.md`)
- LARQL `vindex inspect` (compares pre/post-SFT models — see `project_larql_vindex_inspection.md`)
- `cmd/violet` exposes SFT runs over Unix socket for IDE-driven training

## Related

- [lora_adapter.md](lora_adapter.md) — the adapter shape produced
- [LoRA fuse](../examples/training/lora-fuse.md) — fuse SFT adapter into base for distribution
- [distill.md](distill.md) — distillation reuses SFT scaffolding
- [grpo.md](grpo.md) — reasoning training reuses SFT scaffolding
- `go/dataset_stream.go` — alternate dataset shape
- [HF model-fit example](../examples/model-ops/hf-fit.md) — Hub metadata and fit planning
- [eval.md](eval.md) — eval reports emitted at checkpoint intervals
- `../../../go-inference/docs/inference/training.md` — `TrainableModel` contract
- `../../../go-inference/docs/inference/capability.md` — `CapabilityLoRATraining` flag
