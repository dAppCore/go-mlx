<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Production Benchmark Index

This is the current replay map for the Gemma 4 E2B production lane. It names
the canonical artefacts first and leaves rejected or incomplete probes out of
the main path so a new worker does not need to infer which JSON files matter.

## Current Verdict

The default small-model continuation path is accepted on
`mlx-community/gemma-4-e2b-it-4bit`: the C006 10-chapter run completed, stayed
on prompt through the final chapter, and ended without visible planning or
postscript text. The benchmark artefact set is now indexed, strict-verified,
and cleaned. The overall production goal is still not complete because the
long-context performance gap remains open.

The current measured blockers are still `mlx_lm` and llama.cpp: after shared
full-K/V reuse for paged full-attention owners, `mlx_lm` is `1.928x` faster by
wall time and estimated energy than go-mlx on the 100k cached workflow, while
the cached llama.cpp server row is `1.079x` faster by wall time. That keeps
go-mlx's long-context decode path as the next optimisation boundary. A
follow-up `5120` token-budget diagnostic now shows the current go-mlx path
holds the same `~60 tok/s` decode band for `2489` token natural turns with
bounded memory, but that prompt shape does not force a full `5k` token output.
A materialised-owner K/V probe also stayed flat at `59.855 tok/s` while raising
active/cache memory, so it is recorded as a rejected diagnostic rather than a
new default.

## Accepted go-mlx Artefacts

| Purpose | Artefact | Shape | Result |
| --- | --- | --- | --- |
| 100k retained workflow | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g1024-r10-shared-fullkv-energy100w.json` | `101005` prompt tokens, `10x1024` generation, paged cache with `1024`-token pages, retained prefix, shared full-K/V reuse for full-attention layers | `231.109s`, `60.011 tok/s` decode, `1678.322 tok/s` cold prefill, `0.368ms` warm restore, `3.710 GiB` active MLX, `23110.937 J` at `100 W` |
| 100k sustained long-turn diagnostic | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g5120-budget-r10-shared-fullkv-energy100w.json` | `101005` prompt tokens, `10x5120` budget, natural stop at `2489` tokens per turn, same retained prefix and shared full-K/V reuse | `475.571s`, `59.947 tok/s` decode, `59.962 tok/s` warm decode, `0.362ms` warm restore, `3.726 GiB` active MLX, `47557.087 J` at `100 W` |
| 100k retained book | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-energy100w.json` | `10` chapters, `8192` token budget, `768` visible-token floor, thinking enabled | `482.081s`, `41.442 tok/s` decode, `11425` visible tokens, `4.261 GiB` active MLX |
| C006 accepted continuation | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-energy100w.json` | `10` chapters, `8192` token budget, `512` visible-token floor, thinking enabled | `105.947s`, `80.343 tok/s` decode, `8201` visible tokens, `3.396 GB` active MLX |
| C006 markdown | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-book.md` | Captured book output | Operator-reviewed as on-prompt through the final silence |

Companion notes:

- `docs/runtime/2026-05-20-gemma4-e2b-current-100k-realwork.md`
- `docs/runtime/2026-05-20-gemma4-e2b-c006-report-file-book.md`
- `docs/runtime/2026-05-20-long-context-gap-diagnosis.md`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-token-phase-trace-summary.md`

## Runner Anchors

| Runner | Artefact | Comparable shape | Wall | Decode / throughput | Prefill / restore | Memory | Energy | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| go-mlx | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g1024-r10-shared-fullkv-energy100w.json` | MLX 4bit, `101005` prompt tokens, `10x1024` retained turns, paged K/V `1024`, shared full-K/V reuse for full-attention layers | `231.109s` | `60.011 tok/s` decode | `1678.322 tok/s` cold prefill, `0.368ms` warm restore | `3.710 GiB` active MLX, `3.146 GiB` peak RSS | `23110.937 J` | Current go-mlx baseline; `1.170x` faster on decode and `1.125x` faster by wall/energy than the borrowed-page row |
| `mlx_lm` | `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-cached-workflow-r46-g1024-r10-energy100w.json` | Same MLX 4bit snapshot, `100935` cached prompt tokens, `10x1024` turns | `119.866s` including load+prefill | `103.971 tok/s` decode | `5465.549 tok/s` prefill | `5.473 GB` MLX peak, `3.820 GB` peak RSS | `11986.551 J` | Current configured winner; go-mlx is `1.928x` slower by wall/energy |
| llama.cpp server | `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-100k-cached-server-r10-g1024-energy100w.json` | GGUF `Q4_K_M`, `100926` prompt tokens, `10x1024` cached-prefix turns | `214.205s` | `82.680 tok/s` decode | `1132.450 tok/s` first prefill, `45.591ms` average warm prompt work with `100921` cached tokens | `4.435 GiB` peak RSS | `21420.531 J` | Same-shape cached runner anchor; beats go-mlx by `1.079x` wall/energy |
| llama.cpp cold | `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-pg101005-1024-bench.json` | GGUF `Q4_K_M`, cold `pp101005+tg1024`, one run | `94.904s` | `1075.081 tok/s` combined | Cold replay only | Not recorded in JSON | `9490.352 J` if normalised at `100 W` | Calibration only; superseded by server cached-prefix row for runner-gate evidence |
| vLLM Metal | `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stderr` | Same MLX 4bit snapshot, `100935` input, `1024` output | n/a | n/a | n/a | n/a | n/a | Metal path starts, then strict MLX-LM load rejects extra Gemma 4 shared-K/V tensors |

Cold llama.cpp replay over ten turns would be roughly `949.035s` at the
measured one-run wall time, so go-mlx still beats CLI-style repeated cold
replay. The server-side cached-prefix row is the fairer retained-workflow
anchor and still beats go-mlx on the same repeated shape, but the gap is now
down to `1.079x` wall/energy.

## Rejected Long-Context Diagnostics

These artefacts are indexed because they bound the active 100k blocker, but
they are not accepted production paths.

| Probe | Artefact | Comparable shape | Result | Verdict |
| --- | --- | --- | ---: | --- |
| No paged fast-concat | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-no-fastconcat-g1024-r1-energy100w.json` | MLX 4bit, `100937` prompt tokens, `1024` generated tokens, paged K/V `1024`, accepted fast gates except `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` | `106.324s`, `22.956 tok/s` decode, `1638.525 tok/s` prefill, `3.640 GiB` active MLX | Rejected; page-by-page attention graph is slower than the accepted paged fast-concat lane |
| Native C++ paged attention | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-native-paged-attention-g1024-r1-energy100w.json` | MLX 4bit, `100937` prompt tokens, `1024` generated tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION`, no fast concat | `104.572s`, `23.448 tok/s` decode, `1660.523 tok/s` prefill, `3.640 GiB` active MLX | Rejected; one C++ call trims little overhead and does not replace a fused paged-attention kernel |
| Native C++ paged attention, no single-KV-head repeat | `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-native-paged-no-singlekv-repeat-g1024-r1-energy100w.json` | MLX 4bit, `100912` prompt tokens, `1024` generated tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION`; C++ broadcasts one-head K/V pages | `103.696s`, `23.828 tok/s` decode, `1665.263 tok/s` prefill, `3.613 GiB` active MLX | Rejected; valid micro-optimisation but still far slower than the accepted fast-concat lane |
| Larger paged K/V blocks | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-page2048-g1024-r1-energy100w.json` | MLX 4bit, `101005` prompt tokens, `1024` generated tokens, paged K/V `2048`, accepted fast gates | `80.787s`, `49.984 tok/s` decode, `1678.261 tok/s` prefill, `3.710 GiB` active MLX | Rejected; bigger pages reduce page count but lose decode speed and increase cache memory versus `1024` pages |
| Preallocated paged K/V | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-paged-prealloc-g1024-r1-energy100w.json` | MLX 4bit, `101005` prompt tokens, `1024` generated tokens, paged K/V `1024`, `GO_MLX_ENABLE_PAGED_KV_PREALLOC=1`, accepted fast gates | `80.459s`, `50.743 tok/s` decode, `1679.677 tok/s` prefill, `3.747 GiB` active MLX | Rejected; in-place page updates do not improve the 100k decode path and slightly increase active memory |
| Materialised owner K/V | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-materialized-owner-g1024-r1-energy100w.json` | MLX 4bit, `100932` prompt tokens, `1024` generated tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE=1` | `77.200s`, `59.855 tok/s` decode, `1682.696 tok/s` prefill, `4.385 GiB` active MLX | Rejected; full backing tensors for owner layers do not improve decode and increase active/cache memory |
| Hyper-long fixed cache | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-fixed-sliding-g1024-r1-energy100w.json` | MLX 4bit, `100937` prompt tokens, fixed Gemma 4 cache, shared fixed mask, sliding cache bound, `12 GiB` active/RSS guards | Failed after `13` visible tokens when active memory hit `13748980782` bytes | Rejected; fixed full-capacity global K/V is over the production memory guard |
| Right-sized fixed cache | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-fixed-sliding-rightsized102400-g1024-r1-energy100w.json` | MLX 4bit, README repeat `46`, fixed Gemma 4 cache forced to `102400`, shared fixed mask, sliding cache bound, `12 GiB` active/RSS guards | Failed after `13` visible tokens when active memory hit `13682988726` bytes | Rejected; reducing fixed cache capacity below `131072` still exceeds the production memory guard |
| Borrowed fixed-cache native state | `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-fixed-borrowed-g1024-r1-energy100w.json` | MLX 4bit, README repeat `46`, fixed Gemma 4 cache, shared fixed mask, sliding cache bound, borrowed full-capacity native K/V handles, `12 GiB` active/RSS guards | Failed after `13` visible tokens when active memory hit `13660804802` bytes | Rejected; removing fixed-cache handle clones is correct but not enough to bring the full fixed-cache attention path under the production memory guard |

## Seven-Format E2B Matrix

Source note: `docs/runtime/2026-05-20-gemma4-e2b-quant-matrix.md`.

| Quant | go-mlx status | Decode tok/s | Cold prefill tok/s | Peak GiB | Anchor status |
| --- | --- | ---: | ---: | ---: | --- |
| `mxfp4` | ok after lazy-logit materialisation fix | `84.282` | `3094.590` | `4.794` | `mlx_lm` fails with `100` extra tensors; vLLM fails with `40`; no llama.cpp equivalent |
| `mxfp8` | ok | `74.631` | `2102.044` | `6.256` | `mlx_lm` fails with `100` extra tensors; vLLM fails with `40`; no llama.cpp equivalent |
| `4bit` | ok | `107.914` | `2600.048` | `7.660` | `mlx_lm` fails with `140` extra tensors; vLLM fails with `80`; llama.cpp `Q4_K_M` is `143.952 tok/s` decode |
| `5bit` | ok | `76.489` | `2412.525` | `4.719` | `mlx_lm` fails with `140` extra tensors; vLLM fails with `80`; no llama.cpp equivalent |
| `6bit` | ok | `73.411` | `2297.405` | `5.446` | `mlx_lm` fails with `140` extra tensors; vLLM fails with `80`; no llama.cpp equivalent |
| `8bit` | ok | `78.326` | `2082.905` | `6.338` | `mlx_lm` fails with `140` extra tensors; vLLM fails with `80`; llama.cpp `Q8_0` is `122.513 tok/s` decode |
| `bf16` | ok | `27.703` | `1366.643` | `16.179` | `mlx_lm` fails with `60` extra tensors; vLLM BF16 loads at `3.571706959s` latency for `2205+128`; no llama.cpp BF16 row |

This matrix is a loader and short-latency smoke, not production acceptance
evidence. The raw go-mlx rows and external per-quant rows are now replay-grade;
the production decision still comes from the accepted 100k retained workflow
rather than this short matrix.

## Replay Manifest

This file is `docs/runtime/2026-05-20-production-benchmark-index.md`.

The canonical artefact set is pinned in
`docs/runtime/2026-05-20-production-benchmark-manifest.json`. Verify it with:

```sh
scripts/verify_production_benchmark_manifest.sh
```

The verifier checks that every manifest path exists, is tracked, is non-empty,
that JSON artefacts parse, and that indexed paths remain referenced from this
file. It intentionally only warns about extra `docs/runtime` working-tree
fragments; deletion or quarantine of abandoned probes is a separate cleanup
step so the verifier cannot destroy evidence while an investigation is active.
After that pruning pass, run the stricter cleanup gate:

```sh
scripts/verify_production_benchmark_manifest.sh --strict-clean
```

`--strict-clean` keeps the same artefact checks but fails if `docs/runtime`
still has non-manifest working-tree changes.

Cleanup completed by pruning three obsolete tracked 2026-05-19 book fragments
and moving 137 noncanonical generated runtime fragments into the ignored
`docs/runtime/.quarantine/2026-05-20-noncanonical/` directory.

Manifest coverage details not already shown in the tables above:

- Accepted 100k retained-book markdown:
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-book.md`
- Strict `mlx_lm` load failure evidence:
  `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-strict-load-failure.stderr`
- llama.cpp cached-server note:
  `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-100k-cached-server.md`
- vLLM Metal stdout companion:
  `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stdout`
- External quant rows:
  `docs/runtime/2026-05-20-gemma4-e2b-external-quant-rows.md`
- Safety note:
  `docs/runtime/2026-05-20-chapter-profile-safety.md`
- Seven-format raw JSON rows:
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-mxfp4-current-quant-matrix-3run-readme-energy100w.json`,
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-mxfp8-current-quant-matrix-3run-readme-energy100w.json`,
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-quant-matrix-3run-readme-energy100w.json`,
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-5bit-current-quant-matrix-3run-readme-energy100w.json`,
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-6bit-current-quant-matrix-3run-readme-energy100w.json`,
  `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-8bit-current-quant-matrix-3run-readme-energy100w.json`,
  and `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-bf16-current-quant-matrix-3run-readme-energy100w.json`.

## Replay Environment

Use the workspace-aware setup; do not force standalone `GOWORK=off` for this
repo's normal lane:

```sh
GOWORK=/Users/snider/Code/core/go-mlx/go.work
GOCACHE=/private/tmp/codex-go-mlx-cache
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
```

Run long `chapter-profile` jobs with `-report-file` instead of shell
redirection. In this environment shell redirection repeatedly hid the Metal
device from the runner, while the same workload with `-report-file` completed.

## Next Work

1. Close the `mlx_lm` and llama.cpp cached-runner gap or isolate the specific
   native cause. Borrowing full paged-K/V page handles removed one source of
   per-token graph churn, and retaining the owner materialised full K/V for
   shared full-attention layers improved the accepted 100k workflow from
   `260.093s` / `51.293 tok/s` to `231.109s` / `60.011 tok/s`. The remaining
   live boundary is still evaluated graph/kernel work in the long-context
   attention path, not prompt-cache restore. The current token-phase trace shows
   shared full-K/V reuse moved layers `19`, `24`, `29`, and `34` down to about
   `1.03ms/token`, leaving the early full-attention owner layers `4`, `9`, and
   `14` as the next target. The materialised-owner diagnostic rejected a pure
   MLX `slice_update` backing tensor workaround, so the remaining path is a
   lower-level fused/zero-copy global-attention storage shape. The current
   diagnosis is recorded in
   `docs/runtime/2026-05-20-long-context-gap-diagnosis.md`.
2. Keep the strict manifest gate green whenever new canonical runtime evidence
   is added.
