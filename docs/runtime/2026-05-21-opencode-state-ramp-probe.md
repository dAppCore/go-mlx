<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Opencode-Sized State Ramp Probe

Date: 2026-05-21

This probe exercises the new `state-ramp-profile` command against the primary
GOAL.md interactive shape: an opencode-sized retained state, real appended turn
material, generated assistant output counted into live state, and estimated
energy reported separately from raw decode.

## Inputs

- Model: `mlx-community/gemma-4-e2b-it-4bit`
- Snapshot:
  `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`
- Seed source: `/private/tmp/go-mlx-goal/opencode-seed.txt`
  - `160546` bytes
  - `51197` model tokens
  - The run retains the first `30000` tokens as the warmed state.
- Append source: `/private/tmp/go-mlx-goal/opencode-turns-delimited.txt`
  - `94998` bytes
  - `26433` model tokens
  - `10` explicit user-turn sections split by `---TURN---`
- Accepted chat-shaped append source:
  - `27303` model tokens after Gemma 4 turn wrapping and whole-section
    preservation
- Runtime gates: fast Gemma 4 lane, paged K/V, fp16 K/V storage,
  `GO_MLX_PAGED_KV_PAGE_SIZE=1024`

## Completed Delimited Run

Artifact:
`docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-delimited-r10-g1024-energy100w.json`

Command:

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-goal/lthn-mlx state-ramp-profile \
  -report-file /Users/snider/Code/core/go-mlx/docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-delimited-r10-g1024-energy100w.json \
  -prompt-file /private/tmp/go-mlx-goal/opencode-seed.txt \
  -append-file /private/tmp/go-mlx-goal/opencode-turns-delimited.txt \
  -append-turn-delimiter '---TURN---' \
  -start-tokens 30000 \
  -target-tokens 70000 \
  -append-tokens 4096 \
  -turn-max-tokens 1024 \
  -turns 10 \
  -temperature 1.0 \
  -top-p 0.95 \
  -top-k 64 \
  -repeat-penalty 1.0 \
  -estimate-power-watts 100 \
  -max-active-memory-bytes 12884901888 \
  -max-process-resident-memory-bytes 25769803776 \
  -repeated-line-loop-limit 128 \
  -repeated-sentence-loop-limit 16 \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

Result:

| Metric | Value |
| --- | ---: |
| Successful turns | `10/10` |
| Initial retained state | `30000` tokens |
| Final live state | `59146` tokens |
| Appended tokens | `24953` |
| Generated tokens | `4187` |
| Initial prefill | `2755.434 tok/s` |
| Append average | `1800.615 tok/s` |
| Raw decode average | `77.533 tok/s` |
| Effective turn throughput | `61.689 tok/s` |
| Total wall time | `78.761s` |
| Peak MLX memory | `3.596 GiB` |
| Active MLX memory | `3.114 GiB` |
| Process RSS | `3.246 GiB` |
| Estimated energy at 100 W | `7876.058 J` |

Verdict: useful retained-state scaling evidence, but **not accepted as the
primary interactive gate**. It completed with bounded memory, whole appended
turns, and realistic sampling defaults, but several generated turns naturally
ended after `1` to `8` visible tokens. A long output budget is not enough by
itself; the acceptance row needs a per-turn minimum or a stronger chat-shaped
prompt path that does not trigger degeneration.

## Strict Floor Diagnostic

Artifact:
`docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-delimited-r10-g1024-min512-suppress-eos-energy100w.json`

This rerun added `-turn-min-tokens 512` and `-suppress-eos` to prevent tiny
natural stops. It failed on turn 1 after generating `653` visible tokens because
the output repeated the line `// Implementation_` for `128` consecutive lines.

Verdict: suppressing EOS is **not an accepted solution** for this workflow. It
can force token volume, but it can also turn a model stop into a repeated-code
loop. The next accepted path should use chat-template turn shaping and retained
assistant-turn closure rather than suppressing EOS globally.

## Accepted Chat-Shaped Whole-Turn Run

Artifact:
`docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-chatwholelen-r10-g1024-min256-output-energy100w.json`

Command:

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-goal/lthn-mlx state-ramp-profile \
  -report-file /Users/snider/Code/core/go-mlx/docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-chatwholelen-r10-g1024-min256-output-energy100w.json \
  -prompt-file /private/tmp/go-mlx-goal/opencode-seed.txt \
  -append-file /private/tmp/go-mlx-goal/opencode-turns-delimited.txt \
  -append-turn-delimiter '---TURN---' \
  -chat-template gemma4 \
  -start-tokens 30000 \
  -target-tokens 70000 \
  -append-tokens 4096 \
  -turn-max-tokens 1024 \
  -turn-min-tokens 256 \
  -turns 10 \
  -temperature 1.0 \
  -top-p 0.95 \
  -top-k 64 \
  -repeat-penalty 1.0 \
  -include-output \
  -estimate-power-watts 100 \
  -max-active-memory-bytes 12884901888 \
  -max-process-resident-memory-bytes 25769803776 \
  -repeated-line-loop-limit 128 \
  -repeated-sentence-loop-limit 16 \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

Fixes made before this accepted row:

- Gemma 4 chat wrapping is now available in `state-ramp-profile`.
- Generated assistant turns are closed before the next retained user turn.
- Gemma 4 stop/suppress token controls are reused from `chapter-profile`.
- Delimited append mode preserves whole user-turn sections instead of clipping
  them with `-append-tokens`.
- The wrapper closes reference material and repeats the output-length
  instruction immediately before generation, avoiding raw code continuation.

Result:

| Metric | Value |
| --- | ---: |
| Successful turns | `10/10` |
| Initial retained state | `30000` tokens |
| Final live state | `63584` tokens |
| Appended tokens | `27303` |
| Generated/visible tokens | `6253` |
| Initial prefill | `2754.147 tok/s` |
| Append average | `1766.433 tok/s` |
| Raw decode average | `76.847 tok/s` |
| Effective turn throughput | `64.565 tok/s` |
| Total wall time | `107.741s` |
| Peak MLX memory | `3.612 GiB` |
| Active MLX memory | `3.137 GiB` |
| Process RSS | `3.295 GiB` |
| Estimated energy at 100 W | `10774.150 J` |
| Estimated joules per visible token | `1.723 J` |

Verdict: accepted as the current go-mlx opencode-sized retained workflow row.
It does **not** close the overall production gate yet because same-shape
`mlx_lm`, llama.cpp, and vLLM anchors still need to be run for this accepted
shape, and the warm build-up from this state toward `100k` remains open.

## Next Action

Run same-shape external anchors for the accepted chat-shaped workload, then run
the warm build-up stress path from the accepted `30k`-to-`63.5k` workflow
toward `100k`. Keep raw decode, append wall time, restore/prefill, wall time,
memory, and estimated energy separate.

The runner must treat the `100k` stress ceiling as a context lifecycle boundary.
`state-ramp-profile` now stops fixed-turn ramps once the live state reaches the
target or configured compaction threshold, caps fixed-token appends at that
limit, and emits `context_exhausted`, `folded_state_required`,
`compaction_threshold_tokens`, and `compaction_tail_tokens` in the summary. That
boundary means the next production step is to checkpoint, summarise the exhausted
window, keep a recent tail, and prefill a folded state before accepting more
turns.

The package API for that handoff is `Model.FoldAgentMemory`, which sleeps the
exhausted checkpoint, prefills a fresh session from summary plus recent tail
text, sleeps the folded state with parent lineage, and records folded-state
metadata in the durable index. The benchmark harness can now execute the same
handoff with `-fold-on-exhaustion -fold-store <path>` plus optional
`-fold-summary-file` and `-fold-tail-file`: when the lifecycle boundary is hit,
the report records checkpoint/folded `SleepReport` data, folded prompt byte
counts, folded wake latency, and an optional folded wake/continue turn governed
by `-fold-continue-max-tokens`. If no semantic summary is provided, the harness
uses a metric-only lifecycle summary so the state transition is measurable; real
agent acceptance runs should pass a semantic summary from the compaction layer.

## Folded Lifecycle Probe

After the compact wake path was wired, a focused lifecycle rerun used the same
Gemma 4 E2B 4-bit model, `30000` initial tokens, whole-turn append material,
`1024` generation budget, and a `50000` compaction threshold. The turn floor was
kept at `256` visible tokens but marked rather than failed so short model stops
remain visible without blocking the compaction handoff.

Result:

| Metric | Value |
| --- | ---: |
| Successful turns before fold | `6/6` |
| Initial retained state | `30000` tokens |
| Exhausted checkpoint | `50714` tokens |
| Folded compact state | `221` tokens |
| Appended tokens | `16093` |
| Generated/visible tokens | `4605` / `4601` |
| Initial prefill | `2757.703 tok/s` |
| Append average | `1903.262 tok/s` |
| Raw decode average | `80.213 tok/s` |
| Effective turn throughput | `69.908 tok/s` |
| Total wall time before fold | `76.751s` |
| Fold checkpoint + compact prefill | `1.800s` |
| Folded wake latency | `86.637ms` |
| Folded wake strategy | `folded-prefill` |
| Folded continue | `15` tokens at `103.060 tok/s` |
| Peak MLX memory | `3.283 GiB` |
| Active MLX memory | `3.063 GiB` |
| Process RSS | `3.255 GiB` |
| Estimated energy at 100 W | `7675.102 J` |
| Estimated total including fold lifecycle | `7885.064 J` |

Verdict: the engine now recognises the live context boundary, writes an exact
exhausted checkpoint, folds semantic summary/tail into a compact state, wakes
that folded state without replaying the exhausted prefix, and continues without
the prior non-finite-logits failure. The folded state wakes via
`restore_strategy=folded-prefill` because the compact state is deliberately
small; large non-folded checkpoints remain on the raw K/V block restore path.
