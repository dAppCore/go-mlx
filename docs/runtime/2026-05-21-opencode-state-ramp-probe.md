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
It does **not** close the overall production gate yet. The same-shape `mlx_lm`
anchor and llama.cpp anchor are now recorded below. The warm build-up from this
state toward `100k` is now recorded in the 100k folded State token-wake rerun
below; vLLM remains documented as a same-shape load failure.

## mlx_lm Same-Shape Anchor

Artifacts:

- `docs/runtime/2026-05-21-mlx-lm-gemma4-e2b-4bit-opencode-state-ramp-30k-chatwholelen-r10-g1024-energy100w.json`
- `docs/runtime/2026-05-21-mlx-lm-gemma4-e2b-4bit-opencode-state-ramp-30k-chatwholelen-r10-g1024-min256-mark-energy100w.json`

The anchor uses the same seed file, append file, Gemma 4 turn wrapping, `30000`
seed tokens, `10` whole turns, `1024` token budget, and sampling values. It runs
in an isolated `/private/tmp` Python environment with `mlx==0.31.2` and
`mlx_lm==0.31.3`; the system Homebrew Python was not used because it had drifted
to an incompatible `mlx_lm 0.31.2` / `mlx 0.30.6` pairing.

Result:

| Metric | Strict floor | Marked full run |
| --- | ---: | ---: |
| Completed turns | `2 ok / 1 failed` | `3 ok / 7 below-floor` |
| Initial retained state | `30000` tokens | `30000` tokens |
| Final live state | `39239` tokens | `59579` tokens |
| Appended tokens | `7987` | `27303` |
| Generated/visible tokens | `1246` | `2256` |
| Initial prefill | `9673.402 tok/s` | `9752.856 tok/s` |
| Raw decode average | `126.998 tok/s` | `122.556 tok/s` |
| Effective turn throughput | `109.249 tok/s` | `93.415 tok/s` |
| Total wall time | stopped on turn 3 | `28.284s` including load and prefill |
| Peak MLX memory | `3.944 GB` | `4.405 GB` |
| Estimated energy at 100 W | partial run only | `2828.354 J` |

Verdict: `mlx_lm` is faster on raw decode and wall time, but it does not pass
the accepted real-workload output floor on this prompt shape. The completed
marked row is a useful runner anchor, not an accepted production replacement,
because `7/10` turns fall below `256` visible tokens. This is now treated as
content-shape evidence, not only timing evidence: early natural stops and short
answers mean the runner/model stack is drifting away from the accepted agentic
workload even when tok/s is higher.

## llama.cpp Same-Shape Anchor

Artifact:
`docs/runtime/2026-05-21-llamacpp-gemma4-e2b-q4-k-m-opencode-state-ramp-30k-chatwholelen-r10-g1024-nativebos-energy100w.json`

The anchor uses `unsloth/gemma-4-E2B-it-GGUF` `Q4_K_M`, llama.cpp server build
`b8990-660b1b4bd`, `context=131072`, prompt cache enabled, `flash_attn=on`,
`batch=2048`, `ubatch=512`, `32` context checkpoints, and native llama.cpp BOS
handling. The earlier diagnostic with an explicit prompt `<bos>` was discarded
instead of promoted because llama.cpp warned that it would create a double-BOS
prompt.

Result:

| Metric | Value |
| --- | ---: |
| Successful turns | `10/10` |
| Initial retained state | `30000` tokens |
| Final live state | `67299` tokens |
| Appended tokens | `27303` |
| Generated/visible tokens | `9976` / `9973` |
| Initial prefill | `2585.450 tok/s` |
| Raw decode average | `102.714 tok/s` |
| Wall visible throughput | `76.012 tok/s` |
| Prompt work from llama.cpp timings | `33.429s` |
| Decode time from llama.cpp timings | `97.124s` |
| Total wall time | `131.202s` |
| Peak RSS | `4.398 GiB` |
| Estimated energy at 100 W | `13120.245 J` |
| Visible Gemma channel markers | `10` |

Verdict: llama.cpp is a useful same-shape speed anchor and passes the strict
`256` visible-token floor, but it does not beat the accepted go-mlx row on
wall time or estimated energy for this opencode-shaped workflow. It does beat
go-mlx on raw decode (`102.714 tok/s` versus `76.847 tok/s`) and generates more
visible output (`9973` versus `6253` tokens). The content-shape caveat is
important: every captured turn includes one visible `<channel|>` marker, while
the go-mlx accepted row has none. Treat this as runner/template drift evidence,
not just a formatting nuisance.

## vLLM Metal Same-Shape Attempt

Artifact:
`docs/runtime/2026-05-21-vllm-metal-gemma4-e2b-4bit-opencode-load-failure.md`

The vLLM Metal attempt uses the same MLX 4-bit snapshot, `max_model_len=131072`,
`input_len=31034`, `output_len=1024`, batch size `1`, no warmup, and BF16. It
does not reach latency measurement. The Metal plugin activates, the model is
resolved as `Gemma4ForConditionalGeneration`, chunked prefill is enabled at
`16384`, and the worker reaches `MLX device set to: Device(gpu, 0)`.

Failure:

```text
ValueError: Received 80 parameters not in model:
language_model.model.layers.15.self_attn.k_proj.biases,
language_model.model.layers.15.self_attn.k_proj.scales,
language_model.model.layers.15.self_attn.v_proj.biases,
language_model.model.layers.15.self_attn.v_proj.scales,
...
language_model.model.layers.34.self_attn.v_proj.scales.
```

Verdict: vLLM Metal is documented as unable to run this same-shape E2B 4-bit
workflow today. The blocker is strict `mlx_lm` compatibility with Gemma 4
shared/global K/V tensors, not measured throughput.

## Hot-Path Benchmark Sweep

The first repository-wide benchmark command did not expose useful numbers
because the only existing benchmarks were Metal-only and `go test` could not
see a usable Metal device in this lane:

```sh
GOWORK=/Users/snider/Code/core/go-mlx/go.work \
GOCACHE=/private/tmp/go-mlx-goal/gocache \
go test -run '^$' -bench=. -benchmem ./go/...
```

That surfaced a benchmark coverage gap for non-Metal retained-turn glue, so the
state-ramp prompt/append/report path now has cheap `go test` benchmarks. The
first run found two local wins:

| Benchmark | Before | After | Notes |
| --- | ---: | ---: | --- |
| `BenchmarkStateRampProfileTurnPrompt_Gemma4WholeTurn` | `579.5 ns/op`, `4752 B/op`, `7 allocs/op` | `132.1 ns/op`, `1056 B/op`, `2 allocs/op` | removed the nested reference-wrapper string build and pre-sized the builder |
| `BenchmarkRepeatedStateRampTokens_Append4096Contiguous` | contiguous appends used the same copy path as wrapped diagnostic appends | `0.4620 ns/op`, `0 B/op`, `0 allocs/op` | accepted whole-turn append sections now reuse the source slice instead of copying `4096` tokens |
| `BenchmarkRepeatedStateRampTokens_Append4096Wrapped` | n/a | `3363 ns/op`, `16384 B/op`, `1 alloc/op` | wrapped/repeated diagnostic prompts still allocate because they must materialise a cyclic span |
| `BenchmarkSummariseStateRampProfileTurns_TenTurns` | n/a | `98.65 ns/op`, `0 B/op`, `0 allocs/op` | summary accounting is not the retained-turn bottleneck |

Verification command:

```sh
GOWORK=/Users/snider/Code/core/go-mlx/go.work \
GOCACHE=/private/tmp/go-mlx-goal/gocache \
go test -run '^$' -bench=. -benchmem ./go/cmd/mlx
```

## Next Action

Run the warm build-up stress path from the accepted `30k`-to-`63.5k` workflow
toward `100k`. Keep raw decode, append wall time, restore/prefill, wall time,
memory, output length, content-shape markers, and estimated energy separate.

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
text, sleeps the folded State with parent lineage, and records folded-state
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

Verdict: the engine now recognises the live context boundary, writes an
exhausted checkpoint, folds semantic summary/tail into a compact State, wakes
that folded State without replaying the exhausted prefix, and continues without
the prior non-finite-logits failure. The folded State wakes via
`restore_strategy=folded-prefill` because the compact State is deliberately
small; large non-folded checkpoints remain on the raw State K/V block restore
path.

## 100k Folded State Token-Wake Rerun

After the State token-only wake fix landed, the same semantic fold workflow was
rerun from the accepted `30000` token warmed opencode shape to the `100000`
compaction threshold.

Report:
`docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-opencode-state-ramp-30k-to-100k-fold-semantic-state-tokenwake-energy100w.json`

Result:

| Metric | Value |
| --- | ---: |
| Successful turns before fold | `17/23` |
| Below-floor marked turns | `6/23` |
| Initial retained State | `30000` tokens |
| Final live State before fold | `102704` tokens |
| Appended tokens | `62593` |
| Generated/visible tokens | `10057` / `10057` |
| Initial prefill | `2725.175 tok/s` |
| Append average | `1586.425 tok/s` |
| Raw decode average | `75.368 tok/s` |
| Effective turn throughput | `58.162 tok/s` |
| Total wall time before fold | `183.923s` |
| Fold checkpoint + compact prefill | `2.104s` |
| Folded compact State | `677` tokens across `3` blocks |
| Folded wake latency | `223.207ms` |
| Folded wake strategy | `folded-prefill` |
| Folded continue | `512` tokens at `101.979 tok/s` |
| Peak MLX memory | `3.661 GiB` |
| Active MLX memory | `3.157 GiB` |
| Process RSS | `3.426 GiB` |
| Estimated energy at 100 W | `18392.311 J` |

Verdict: the previous multi-block folded wake failure is fixed in the real
model path. The folded State has three blocks and wakes via token-only prefill
instead of K/V assembly, then completes the configured `512` token continuation.
This closes the warm build-up `100k` stress gate.

Two caveats remain open. First, long-context content degradation is visible:
turns `17`, `19`, `20`, `21`, `22`, and `23` fall below the `256` visible-token
floor. Second, the exhausted checkpoint still reports `65536` captured tokens
while the live State was `102704` tokens, so exact checkpoint fidelity past
`64k` is not yet proven even though the compact folded continuation works.

## AX Hot-Path Benchmark Pass

The State wake path now has a Go benchmark contract. The folded wake path uses
`kv.LoadPrefixTokensFromStateBlocksWithOptions`, which parses only token IDs
from the State block payload and avoids assembling K/V tensors for compact
folded prefill.

Command:

```sh
GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-goal/gocache go test -bench=. -benchmem ./go/...
```

Key rows:

| Benchmark | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| `BenchmarkLoadPrefixFromStateBlocks_MixedWindowThreeBlocks` | `18968` | `80258` | `49` |
| `BenchmarkLoadPrefixTokensFromStateBlocks_MixedWindowThreeBlocks` | `13891` | `36993` | `14` |
| `BenchmarkStateRampProfileTurnPrompt_Gemma4WholeTurn` | `229.4` | `1056` | `2` |
| `BenchmarkRepeatedStateRampTokens_Append4096Contiguous` | `0.4691` | `0` | `0` |

The State token loader also has a regression test that intentionally builds
multi-block State data whose full K/V assembly path fails on shape mismatch;
the folded token prefill path still loads `[1 2 3 4]` because K/V tensors are
not needed for compact wake.
