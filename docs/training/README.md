<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# training/ — fine-tuning + eval

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **research-grade training pipeline** that distinguishes go-mlx from a mere inference runtime. Native AdamW, native gradient computation through Metal, native LoRA, native distillation, native GRPO — no Python required, no subprocess hop, full primitives consumable from Go programs.

This is the substrate that fine-tunes Vi, distills Lemma, and generates the LARQL vindex inspection signals.

## File map

| File | Doc | Role |
|------|-----|------|
| `sft.go` | [sft.md](sft.md) | Supervised fine-tuning loop |
| `lora_adapter.go` | [lora_adapter.md](lora_adapter.md) | LoRA adapter identity + save/load |
| `lora_fuse.go` | (planned) | Fuse adapter into base for distribution |
| `grpo.go` | [grpo.md](grpo.md) | Group Relative Policy Optimisation (reasoning) |
| `distill.go` | [distill.md](distill.md) | Knowledge distillation (teacher→student) |
| `eval.go` | [eval.md](eval.md) | Dataset-native evaluation runner |
| `fast_eval.go` | (planned) | Optimised prefill-only eval |
| `dataset_stream.go` | (planned) | go-mlx native dataset iterator |
| `hf_fit.go` | (planned) | HuggingFace Hub source for training data |
| `model_merge.go` | (planned) | Tensor-level model interpolation/merge |
| `training.go` / `training_stub.go` | (planned) | Training entry points |

## Pipeline shape

```
       ┌──────────────────┐
       │   Base model     │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐       ┌──────────────────┐
       │ Distill          │       │ SFT              │
       │ from larger      │  AND/OR │ on labelled set │
       └────────┬─────────┘       └────────┬─────────┘
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────┐
                │ GRPO             │  ← reasoning post-train
                │ for reasoning    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Eval suite       │  ← capability + safety
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Fuse + Quantise  │  ← ship-ready
                │ (lora_fuse +     │
                │  gguf_quantize)  │
                └──────────────────┘
```

## Why training natively in Go

Three reasons the Python path didn't suffice:

1. **No Python on the hot path.** CoreAgent needs to train without spawning a Python subprocess from a Go binary.
2. **Same primitives as inference.** A training adapter loads into the same `metal.Model` that serves inference. No model-format conversion between train and serve.
3. **Compose with the rest of the stack.** `cmd/violet` can expose training over Unix socket; `core/ide` can launch a training run from its UI without bridging Python.

Status: dense-model training (Gemma 3/4 dense, Qwen 3, Llama 3) is production. MoE training (MiniMax M2) pending Phase 1 forward landing. Vi training uses this pipeline live.

## Used by

- Vi training (`project_vi_training_plan.md`)
- Lemma vertical stack (`project_lemma_vertical_stack.md`)
- LARQL vindex inspection (pre/post-SFT model diff)
- LEK ethics training (`project_lemer_lek_shipped.md`)

## Related

- `../../../go-inference/docs/inference/training.md` — TrainableModel contract
- `../../../go-inference/docs/inference/capability.md` — training capability flags
- `../memory/agent_memory.md` — Wake/Sleep on training checkpoints (resume mid-run)
- `examples/` — per-feature usage walkthroughs (training, distill, GRPO, eval)
