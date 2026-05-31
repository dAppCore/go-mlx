<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Official Gemma 4 E2B Local Preflight

Date: 2026-05-31
Build: `255eaad`
Binary: `/private/tmp/go-mlx-goal/bin/lthn-mlx`

This records the local proof that the locked official Google Gemma 4 E2B
target and MTP assistant snapshots are present, metadata-loadable, pairable,
and compatible with the archived q4 MLX community control.

## Snapshots

| Role | Model | Revision |
| --- | --- | --- |
| target | `google/gemma-4-E2B-it` | `905e84b50c4d2a365ebde34e685027578e6728db` |
| assistant | `google/gemma-4-E2B-it-assistant` | `5810c41a67974da9c7bd6f3e6c69d5d13854d9f0` |
| q4 control | `mlx-community/gemma-4-e2b-it-4bit` | `99d9a53ff828d365a8ecae538e45f80a08d612cd` |

## Commands

```sh
/private/tmp/go-mlx-goal/bin/lthn-mlx official-gemma4-verify \
  -role target \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db

/private/tmp/go-mlx-goal/bin/lthn-mlx official-gemma4-verify \
  -role assistant \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it-assistant/snapshots/5810c41a67974da9c7bd6f3e6c69d5d13854d9f0

/private/tmp/go-mlx-goal/bin/lthn-mlx official-gemma4-pair-verify \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it-assistant/snapshots/5810c41a67974da9c7bd6f3e6c69d5d13854d9f0

/private/tmp/go-mlx-goal/bin/lthn-mlx official-gemma4-control-compare \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

All four commands exited successfully.

## Target Metadata

- pack architecture: `gemma4_text`
- model type: `gemma4`
- context length: `131072`
- layers: `35`
- hidden size: `1536`
- vocab size: `262144`
- local sliding window: `512`
- layer pattern: `28` sliding-attention layers, `7` full-attention layers
- shared KV layers: `20`
- full attention RoPE: proportional, theta `1000000`, partial rotary factor `0.25`
- sliding attention RoPE: default, theta `10000`
- quantisation: unquantised official source snapshot

## Assistant Metadata

- pack architecture: `gemma4_assistant`
- backbone hidden size: `1536`
- assistant hidden size: `256`
- assistant layers: `4`
- vocab size: `262144`
- context length: `131072`
- ordered embeddings: enabled
- centroids: `2048`
- centroid intermediate top-K: `32`

## Control Comparison

The official target matches the archived q4 control for architecture, context,
hybrid attention shape, p-RoPE metadata, shared-KV metadata, PLE metadata, and
chat-template semantics. Quantisation differs by design: the official target is
the source snapshot, while the archived control is the accepted q4 baseline.

This closes the local metadata preflight step only. It does not promote the
official lane to production; native-load, retained-State, q6/q8/q4 runtime
selection, target-only versus MTP, and TurboQuant gates remain separate.
