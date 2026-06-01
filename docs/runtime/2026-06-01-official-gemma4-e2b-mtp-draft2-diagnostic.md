<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Official Gemma 4 E2B MTP Draft-2 Diagnostic

Date: 2026-06-01
Binary: `/private/tmp/go-mlx-self/bin/lthn-mlx`

Purpose: compare the official Google E2B target against the same target with
the official E2B MTP assistant attached, using go-mlx only. This is a
diagnostic self-comparison for the production MTP gate. It is not a retained
10-turn workflow and does not promote MTP as a default. The initial row used
the production default `draft_tokens=2`; follow-up rows added the required
diagnostic `1,2,4` draft-token sweep.

## Models

| Role | Model | Snapshot |
| --- | --- | --- |
| target | `google/gemma-4-E2B-it` | `905e84b50c4d2a365ebde34e685027578e6728db` |
| assistant | `google/gemma-4-E2B-it-assistant` | `5810c41a67974da9c7bd6f3e6c69d5d13854d9f0` |

The pair verifier passed and reported the official assistant layout:
`gemma4_assistant`, ordered embeddings enabled, `2048` centroids,
centroid intermediate top-K `32`, four assistant layers, and I64
`masked_embedding.token_ordering`. The source safetensors report the official
flat token-ordering shape `[262144]`; the loaded runtime layout normalises the
same ordering into `[2048,128]` for centroid lookup.

## Command Shape

Target-only:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx driver-profile \
  -json \
  -report-file /private/tmp/go-mlx-self/mtp-compare/official-e2b-target-only-diagnostic.json \
  -include-output=false \
  -context 4096 \
  -prompt "Write a compact technical paragraph about retained state, speculative drafting, and why greedy parity must be measured before making MTP default." \
  -max-tokens 256 \
  -runs 1 \
  -estimate-power-watts 75 \
  -trace-token-phases=false \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db
```

MTP draft-2:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx driver-profile \
  -json \
  -report-file /private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-draft2-diagnostic.json \
  -include-output=false \
  -context 4096 \
  -prompt "Write a compact technical paragraph about retained state, speculative drafting, and why greedy parity must be measured before making MTP default." \
  -max-tokens 256 \
  -runs 1 \
  -estimate-power-watts 75 \
  -trace-token-phases=false \
  -speculative-draft-model /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it-assistant/snapshots/5810c41a67974da9c7bd6f3e6c69d5d13854d9f0 \
  -speculative-draft-tokens 2 \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db
```

Production compare:

```bash
/private/tmp/go-mlx-self/bin/lthn-mlx production-mtp-compare \
  -json \
  -turns 1 \
  -draft-token-sweeps 2 \
  -official-pair-report /private/tmp/go-mlx-self/mtp-compare/official-e2b-pair-report.json \
  /private/tmp/go-mlx-self/mtp-compare/official-e2b-target-only-diagnostic.json \
  /private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-draft2-diagnostic.json
```

The compare JSON was also written to:
`/private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-compare-diagnostic.json`.

The same command shape was rerun with `-speculative-draft-tokens 1` and
`-speculative-draft-tokens 4`, writing:

- `/private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-draft1-diagnostic.json`
- `/private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-draft4-diagnostic.json`
- `/private/tmp/go-mlx-self/mtp-compare/official-e2b-mtp-compare-sweep-diagnostic.json`

## Results

| Lane | Visible tokens | Generated tokens | Decode tok/s | Visible tok/s | Wall | Peak memory | Active+cache | Energy at 75 W | Output hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| target-only | 109 | 109 | 28.154236962710087 | 28.154236962710087 | 4.23959175s | 11897395262 B | 13208314718 B | 317.96938125 J | `8ec3ab0e6411075429169a4e26806f06909b0369608422eb0a07bffb7e638397` |
| MTP draft-1 | 110 | 110 | 27.61607294358874 | 25.472911845025585 | 4.3187645s | 12054962106 B | 13383296602 B | 323.9073375 J | `360563c3ee985ddb3e47555859582903dfb5128ea78fb4ff4fce0222345bfa64` |
| MTP draft-2 | 110 | 110 | 26.88335766075169 | 24.793832705056825 | 4.437141333s | 12054962106 B | 13389436506 B | 332.785599975 J | `360563c3ee985ddb3e47555859582903dfb5128ea78fb4ff4fce0222345bfa64` |
| MTP draft-4 | 110 | 110 | 25.89798617826879 | 24.020443796221084 | 4.579903041s | 12054962106 B | 13396221530 B | 343.492728075 J | `360563c3ee985ddb3e47555859582903dfb5128ea78fb4ff4fce0222345bfa64` |

MTP counters:

| Draft tokens | Proposed | Accepted | Rejected | Acceptance | Target verifies | Draft calls | Target verify tok/s | Warm decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 110 | 26 | 84 | 0.23636363636363636 | 110 | 110 | 28.532267752321324 | 27.61607294358874 |
| 2 | 180 | 27 | 153 | 0.15 | 90 | 90 | 28.190787594849148 | 26.88335766075169 |
| 4 | 340 | 30 | 310 | 0.08823529411764706 | 85 | 85 | 28.759906213278196 | 25.89798617826879 |

The combined production compare observed all required draft-token sweep values:
`[2,1,4]`. The draft-2 row remains the candidate row because it is the current
production default; draft-1 and draft-4 supply sweep evidence only.

The production compare decision rejected promotion:

```text
enable_by_default=false
reason="retained workflow turn count is below the MTP promotion minimum"
wall_speedup=0.9554781855761997
visible_speedup=0.8806430356431229
acceptance_rate=0.15
```

Quality flags:

- `greedy_output_hash_mismatch`
- `target_only_restore_duration_missing`
- `mtp_restore_duration_missing`

## Interpretation

This is useful negative evidence. The official assistant is attachable and the
runtime emits the required MTP counters, but all measured draft depths are
slower than target-only on this short official source-snapshot prompt. The
acceptance curve also bends the wrong way as draft depth grows: draft-1 accepts
`23.6%`, draft-2 accepts `15.0%`, and draft-4 accepts `8.8%`. The shared MTP
output hash differs from target-only, so greedy parity is not proven.

The next MTP benchmark must use the retained workflow shape from `GOAL.md`:
at least `10` turns and real restore durations. This diagnostic narrows the
failure to MTP quality/performance under the current prompt/runtime path rather
than missing assistant attachment or missing sweep accounting.
