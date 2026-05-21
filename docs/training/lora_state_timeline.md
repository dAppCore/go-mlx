<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# LoRA State Timeline

This document defines the training-state layout for LoRA adapter updates in the
go-mlx State engine. It follows the native one-step proof added in
`TestSFTNativeSmoke_OneLoRAStep_Good`: a real
`mlx-community/gemma-4-e2b-it-4bit` model can execute one rank-2 LoRA SFT step
against `q_proj` and return a finite loss.

## Scope

The timeline stores trainable adapter state, not base model weights. For Gemma 4
E2B/E4B the PLE tables, router weights, and frozen projections remain static
unless a caller explicitly opts into broader targets. The default target set is
the safe attention path (`q_proj`, `v_proj`, `o_proj`), with the same PLE guard
used by native LoRA config normalisation.

## Tracks

Each training run writes one State manifest plus append-only binary tracks:

| Track | Contents | Rollback use |
| --- | --- | --- |
| `manifest` | model identity, tokenizer identity, adapter config, target tensor table, dtype, alignment, seed, sample cursor | validates that a wake uses the same base model and adapter shape |
| `lora.a` | post-step LoRA A matrices grouped by dtype and target projection | restores trainable A for a chosen step |
| `lora.b` | post-step LoRA B matrices grouped by dtype and target projection | restores trainable B for a chosen step |
| `adam.m` | AdamW first-moment slab for each trainable matrix | resumes optimiser state without cold-starting momentum |
| `adam.v` | AdamW second-moment slab for each trainable matrix | resumes optimiser state without losing variance history |
| `events` | loss, learning rate, epoch, sample IDs, probe refs, checkpoint labels | supports divergence audits and training dashboards |

The default frame mode is full post-step frames for `lora.a`, `lora.b`,
`adam.m`, and `adam.v`. LoRA matrices are small relative to the base model, so
full frames make rollback O(1): move the manifest's active step pointer and map
the four frame offsets. A future delta-compressed mode may store per-step deltas
with periodic full keyframes, but that is not the default because it makes
rollback depend on replaying a delta chain.

## Layout

Frames are grouped by dtype, then by target tensor. Every tensor entry records:

- stable tensor key, for example `layers.3.self_attn.q_proj`
- logical matrix kind: `A`, `B`, `adam.m`, or `adam.v`
- element dtype and byte width
- rows, columns, and stride
- byte offset from the start of the frame slab
- byte length and alignment padding

The native reader must be able to wrap each frame as a non-owning view. The C++
side should expose this as `std::mdspan` over the pinned State bytes, then pass
the view pointer into the MLX array bridge without copying. The Go side owns the
manifest and file lifecycle; the native side owns only the evaluated view for
the current step.

## Write Protocol

1. Initialise LoRA with the normal native config path. This keeps PLE static and
   creates the trainable tensor table from the actual adapter layers.
2. Before the first optimiser step, write step `0` as a full frame. This captures
   the random LoRA A initialisation and the zero LoRA B / AdamW moments.
3. After each successful AdamW step and `mlx_eval` boundary, materialise the
   updated LoRA A/B and packed AdamW moment slabs.
4. Append one full frame for the step and one `events` row carrying loss,
   optimiser step, epoch, sample IDs, and probe refs.
5. Commit the manifest step pointer last. Readers only see complete frames.

If step write fails before the manifest pointer advances, the previous step
remains the active state. If loss diverges, rollback changes the active pointer
to a prior step and remaps the four frame offsets.

## Verification

The minimum implementation gate is:

```sh
env GO_MLX_SFT_SMOKE_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd \
  MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  GOCACHE=/private/tmp/go-mlx-gocache \
  go test ./go -run TestSFTNativeSmoke_OneLoRAStep_Good -count=1 -v -timeout=10m
```

The first State timeline implementation must add a second gate that performs
one step, writes step `0` and step `1`, wakes from step `1`, and verifies that
the adapter tensor table, AdamW step, and latest loss metadata round-trip.
