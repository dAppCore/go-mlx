<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B 4bit 100k Retained-State Run

This note records the 2026-05-19 investigation into the 100k-token E2B 4bit
long-context lane. The important finding is that the fixed retained-cache path
was not merely inefficient: it could reserve hundreds of GiB of MLX active or
virtual memory for a roughly 5 GiB quantised model. The accepted 100k lane is
therefore paged retained cache with sliding-tail prompt-cache snapshots.

## Model And Shape

- Model: `mlx-community/gemma-4-e2b-it-4bit`
- Local snapshot:
  `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`
- Context length: `131072`
- Prompt shape: README repeated to `100912` prompt tokens
- Power estimate: normalised `100 W` wall-clock estimate, not measured power
- Current accepted long-context fast lane:
  paged rotating cache, `prefill_chunk_size=512`, retained prompt cache,
  fixed Gemma 4 cache gates disabled above the long-context threshold

## Evidence Table

| Run | Artifact | Result | Wall | Prefill | Decode | Memory |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Paged no-fixed 8k return | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-longctx-r46-ctx131072-g8000-r1-nofixed-cachemem-energy100w.json` | 1/1 success, `8000` generated tokens | `841.019s` | `641.93 tok/s` | `11.98 tok/s` | peak `7.25 GiB`, active `3.53 GiB`, cache `6.13 GiB` |
| Fixed retained cache | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-fast-gemma4-lane-r46-ctx131072-g128-r3-patched-procmem-energy100w.json` | 3/3 short success, but rejected | `194.088s` | warm cache hits | `18.08 tok/s` avg | active `197.17 GiB`, virtual `1232.02 GiB`, RSS `2.96 GiB` by run 3 |
| Paged retained before sliding snapshot fix | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-paged-retained-r46-ctx131072-g128-r3-procmem-energy100w.json` | 3/3 success, but prompt-cache missed each turn | `515.428s` | `647.14 tok/s` avg | `12.16 tok/s` avg | active `3.53 GiB`, virtual `1320.02 GiB`, RSS `4.99 GiB` |
| Paged retained after sliding snapshot fix | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-paged-retained-r46-ctx131072-g128-r3-sliding-snapshot-procmem-energy100w.json` | 3/3 success, turns 2-3 restore from cache | `203.073s` | warm equivalent `32.96M tok/s` | `12.20 tok/s` avg | active `3.58 GiB`, virtual `732.01 GiB`, RSS `5.05 GiB` |
| Final 10-turn fast lane | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-fast-gemma4-lane-paged-retained-r46-ctx131072-g128-r10-procmem-energy100w.json` | 10/10 success, turns 2-10 restore from cache | `275.717s` | warm equivalent `45.19M tok/s` | `12.34 tok/s` avg | active `3.58 GiB`, virtual `734.41 GiB`, RSS `5.19 GiB` |

## Final 10-Turn Result

The final run processed `100912` prompt tokens on each of `10` turns and
generated `1280` visible tokens total. Treating the retained prefix as logical
work, that is `1010400` logical tokens over `275.717s`, or
`3664.63` effective logical tok/s.

The cache restore path removed almost all repeated prompt setup:

- Cold prompt prefill: `647.19 tok/s`
- Warm prompt restore average: `1.98 ms`
- Prompt setup saved versus replaying prefill every turn: `1403.301s`
- Wall-clock equivalent if replaying prefill: `1679.018s`
- Total wall-clock speedup versus replay: `6.09x`
- Estimated total energy at `100 W`: `27571.70 J`
- Estimated prompt setup energy saved at `100 W`: `140330.10 J`

This does not make raw decode fast at 100k. The final paged-retained raw decode
rate is `12.34 tok/s`, and the single 8k return control is `11.98 tok/s`. The
win is retained-state wall time across agentic turns, not raw token generation.

## What Went Wrong

The fixed retained cache path was the obvious suspect because it improved the
short warm-cache timing while making memory accounting absurd. With process
memory instrumentation enabled, run 3 reported:

- MLX active memory: `197.17 GiB`
- Process virtual memory: `1232.02 GiB`
- Process resident memory: `2.96 GiB`

That means the earlier RSS-only view hid the bad allocation pattern. The
process was not physically holding 1.2 TiB, but the virtual reservation and MLX
active accounting are still invalid for a 5 GiB model and can lead to OOM
behaviour. The fixed cache path is therefore not an accepted 100k lane.

The paged path had a separate bug: sliding paged caches were being rejected by
the prompt-cache snapshot code because their absolute offset did not equal
their retained tail length. At 100k, Gemma 4 sliding layers can have
`Offset=100912` and `Len=512`. The old snapshot guard treated that as
uncacheable, so each warm turn replayed the whole prefix. The fix snapshots
paged caches before the generic offset check and stores the bounded sliding
tail at its absolute offset.

## Current Policy

For hyper-long contexts, `-fast-gemma4-lane` now uses the normal fast decode
gates but excludes the fixed Gemma 4 cache gates. The long-context accepted
policy is:

- keep direct greedy, generation stream, router, native MLP, expert-id, and
  sorted-prefill gates enabled
- use paged retained cache for `131072` context
- keep fixed Gemma 4 cache and fixed sliding-mask gates out of 100k runs
- keep process virtual, resident, and peak resident memory in the JSON metrics

## External Runner Status

This file should not be read as a fresh 100k llama.cpp, `mlx_lm`, or vLLM
parity claim. Earlier small-context and 29k runner calibration is preserved in
`docs/runtime/2026-05-19-runner-calibration.md`, but this 100k investigation
only proves the corrected go-mlx retained-state lane and the fixed-cache memory
failure. A fair external 100k comparison still needs a successful same-shape
run with comparable cache reuse semantics.
