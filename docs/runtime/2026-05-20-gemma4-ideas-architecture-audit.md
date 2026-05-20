<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 IDEAS.md Architecture Audit

Date: 2026-05-20

This note turns the updated `IDEAS.md` guidance into code-grounded status. The
goal is to keep the optimisation backlog honest: confirmed paths should not stay
as vague research items, and missing paths should be named as concrete work.

## Current Findings

| Item | Status | Evidence | Next action |
| --- | --- | --- | --- |
| C++23 native bridge | Shipped for the repo-local native layer | `CMakeLists.txt:5-8` sets macOS 26.0 and C++23; `go/internal/metal/mlx_build_config.h:12-16` hard-fails older C++ | Keep as baseline; do not reopen as a speculative speed item |
| Pinned raw byte arrays | Shipped for snapshot byte slabs | `go/internal/metal/pinned_array.go:49-67` pins Go byte storage with `runtime.Pinner`; `go/internal/metal/pinned_array_bridge.cpp:137-225` passes it to `mlx_array_new_data_managed_payload` | Extend to direct mapped `.mp4` state only if the state file path can hand out stable aligned slabs |
| `std::mdspan` strided validation | Shipped for 4D pinned views | `go/internal/metal/pinned_array_bridge.cpp:81-109` wraps the raw pointer as a 4D `std::mdspan` and validates the strided view | Reuse this bridge for any future state-file slab view rather than adding a second layout checker |
| Proportional RoPE | Covered | Go precomputes Gemma 4 p-RoPE frequencies in `go/internal/metal/gemma4.go:1198-1224`; MLX selects `rope_*freqs*` kernels when a frequency array is supplied in `lib/mlx/mlx/backend/metal/rope.cpp:98-105`; Metal consumes per-dimension frequencies in `lib/mlx/mlx/backend/metal/kernels/rope.metal:69-81`; `TestGemma4_ProportionalRoPEFreqsMatchesHFDefinition_Good` protects the HF formula | No patch now |
| RMSNorm scale convention | Audited, leave direct-scale unless model weights prove otherwise | The MLX kernel multiplies the supplied scale exactly in `lib/mlx/mlx/backend/metal/kernels/rms_norm.metal:67-72`; Go passes the precomputed weight directly via `go/internal/metal/fast.go:25-31`; Gemma 4 currently copies norm weights in `go/internal/metal/gemma4.go:1390-1433`; `TestGemma4_PrecomputeNormWeightsUsesDirectScale_Good` asserts direct scale | Do not blindly add `(1 + weight)`; validate MLX-community Gemma 4 weight convention first |
| Cross-layer KV sharing | Shipped | `go/internal/metal/gemma4.go:1130-1160` builds shared owners by attention type; `TestGemma4_E4BSharedCacheLayoutUsesLayerTypes_Good` verifies shared layers allocate no fresh cache | Keep |
| Unified K=V storage | Rejected for final cache tensors | `go/internal/metal/gemma4.go:2527-2550` shares the projection source with a ref-counted MLX handle, then K takes KNorm+RoPE while V takes value RMSNorm; `TestGemma4_AttentionKEqVDoesNotAliasFinalCache_Good` guards that the final cache tensors diverge | Do not pack final K/V into one state slab. A future raw-projection timeline would need to store pre-transform projection plus metadata and recompute K/V on restore, which is not the zero-copy inference state path |
| LoRA PLE gradient isolation | Covered by default targets, needs policy guard if broadened | `DefaultLoRAConfig` targets `q_proj` and `v_proj` in `go/internal/metal/lora.go:146-155`; Gemma 4 LoRA only wraps named projection modules in `go/internal/metal/gemma4.go:3125-3181`; PLE embeddings are not trainable by default | Add a guard/test before enabling broad "all linear" LoRA on Gemma 4 |
| AdamW state layout | Shipped for homogeneous matrix moments | `go/internal/metal/optim.go` enables `PackedState` by default, keeps AdamW `m`/`v` in contiguous MLX slabs when parameter shapes and dtypes permit, and exposes an explicit fallback knob; `go/internal/metal/optim_test.go` covers packed, disabled, and mixed-dtype fallback paths; `go/sft.go` preserves the setting through SFT metadata/config replay | Keep the mdspan-backed parameter/file slab as part of the future LoRA delta `.mp4` timeline rather than claiming it from optimiser state alone |
| LoRA delta `.mp4` timeline | Not shipped | Existing KV state bridge handles inference snapshots, not training delta tracks | Design after the runner can complete a real LoRA step |
| MTP drafter co-training | Research only | Native MTP inference exists, but current GOAL rows reject it as production decode until acceptance improves | Revisit after target-model SFT is stable |
| Public training surface | Mostly shipped, adapter still open | `go/training.go:11-72` exports arrays, LoRA, AdamW, cache, dtype, and `InternalModel`; `go/training.go:211-219` exposes `TrainingModel`; `go/backend.go:1268-1307` exposes `Model.Tokenizer` and `NewLoRA`; `go/sft.go:592-659` exposes `Model.TrainSFT` | Build the downstream `gomlxrunner` against this surface or add only the missing thin wrappers it proves necessary |

## Practical Read

The next useful engineering target is not another broad C++23 conversion. That
baseline is already present, and AdamW now packs compatible moment state by
default. The highest-signal remaining items from the updated `IDEAS.md` are:

1. A downstream `gomlxrunner` compile pass that proves the public training
   surface is sufficient for `lthn/desktop`.
2. The LoRA delta `.mp4` timeline, including mdspan-backed parameter/file slabs,
   after one real runner step works end-to-end.
