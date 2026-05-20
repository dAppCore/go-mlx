<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Production Benchmark Index

This is the current replay map for the Gemma 4 E2B production lane. It names
the canonical artefacts first and leaves rejected or incomplete probes out of
the main path so a new worker does not need to infer which JSON files matter.

## Current Verdict

The default small-model continuation path is accepted on
`mlx-community/gemma-4-e2b-it-4bit`: the C006 10-chapter run completed, stayed
on prompt through the final chapter, and ended without visible planning or
postscript text. The overall production goal is still not complete because the
same-shape runner-anchor gate and long-context performance gap remain open.

The current measured blocker is `mlx_lm`: on the 100k cached workflow it is
`3.408x` faster by wall time and estimated energy than go-mlx. That makes
go-mlx's long-context prefill/decode path the next optimisation boundary.

## Accepted go-mlx Artefacts

| Purpose | Artefact | Shape | Result |
| --- | --- | --- | --- |
| 100k retained workflow | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-guarded-r46-ctx131072-g1024-r10-longturn-naturalstop-energy100w.json` | `101005` prompt tokens, `10x1024` generation, paged cache, retained prefix | `408.483s`, `43.617 tok/s` decode, `642.657 tok/s` cold prefill, `2.116ms` warm restore, `3.699 GiB` active MLX, `40848.257 J` at `100 W` |
| 100k retained book | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-energy100w.json` | `10` chapters, `8192` token budget, `768` visible-token floor, thinking enabled | `482.081s`, `41.442 tok/s` decode, `11425` visible tokens, `4.261 GiB` active MLX |
| C006 accepted continuation | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-energy100w.json` | `10` chapters, `8192` token budget, `512` visible-token floor, thinking enabled | `105.947s`, `80.343 tok/s` decode, `8201` visible tokens, `3.396 GB` active MLX |
| C006 markdown | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-book.md` | Captured book output | Operator-reviewed as on-prompt through the final silence |

Companion notes:

- `docs/runtime/2026-05-20-gemma4-e2b-current-100k-realwork.md`
- `docs/runtime/2026-05-20-gemma4-e2b-c006-report-file-book.md`

## Runner Anchors

| Runner | Artefact | Comparable shape | Wall | Decode / throughput | Prefill / restore | Memory | Energy | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| go-mlx | `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-guarded-r46-ctx131072-g1024-r10-longturn-naturalstop-energy100w.json` | MLX 4bit, `101005` prompt tokens, `10x1024` retained turns | `408.483s` | `43.617 tok/s` decode | `642.657 tok/s` cold prefill, `2.116ms` warm restore | `3.699 GiB` active MLX, `6.509 GiB` peak RSS | `40848.257 J` | Accepted go-mlx baseline |
| `mlx_lm` | `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-cached-workflow-r46-g1024-r10-energy100w.json` | Same MLX 4bit snapshot, `100935` cached prompt tokens, `10x1024` turns | `119.866s` including load+prefill | `103.971 tok/s` decode | `5465.549 tok/s` prefill | `5.473 GB` MLX peak, `3.820 GB` peak RSS | `11986.551 J` | Current configured winner; go-mlx is `3.408x` slower by wall/energy |
| llama.cpp | `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-pg101005-1024-bench.json` | GGUF `Q4_K_M`, cold `pp101005+tg1024`, one run | `94.904s` | `1075.081 tok/s` combined | Cold replay only | Not recorded in JSON | `9490.352 J` if normalised at `100 W` | Cold calibration only; cached-prefix workflow still missing |
| vLLM Metal | `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stderr` | Same MLX 4bit snapshot, `100935` input, `1024` output | n/a | n/a | n/a | n/a | n/a | Metal path starts, then strict MLX-LM load rejects extra Gemma 4 shared-K/V tensors |

Cold llama.cpp replay over ten turns would be roughly `949.035s` at the
measured one-run wall time, so go-mlx still beats CLI-style repeated cold
replay. That does not close the runner gate because `mlx_lm` already has a
faster cached-prefix row on the same workflow.

## Seven-Format E2B Matrix

Source note: `docs/runtime/2026-05-19-gemma4-e2b-quant-matrix.md`. This is a
summary-only matrix in the current tree: the raw JSON/stderr artefacts named by
that older note are not present, so the seven-format gate still needs a rerun
or recovery of those files before it can be treated as replay-grade evidence.

| Quant | go-mlx status | Decode tok/s | Cold prefill tok/s | Peak GiB | Anchor status |
| --- | --- | ---: | ---: | ---: | --- |
| `mxfp4` | ok after affine override fix | `109.197` | `3735.077` | `5.139` | no llama.cpp equivalent; external per-quant failure artefact still missing |
| `mxfp8` | ok | `102.757` | `3096.460` | `6.516` | no llama.cpp equivalent; external per-quant failure artefact still missing |
| `4bit` | ok | `123.346` | `3724.280` | `4.607` | llama.cpp `Q4_K_M` anchor exists; `mlx_lm`/vLLM load failures recorded |
| `5bit` | ok | `110.243` | `3711.742` | `5.047` | no llama.cpp equivalent; external per-quant failure artefact still missing |
| `6bit` | ok | `103.056` | `3683.675` | `5.586` | no llama.cpp equivalent; external per-quant failure artefact still missing |
| `8bit` | ok | `101.268` | `3728.024` | `6.665` | llama.cpp `Q8_0` anchor exists; `mlx_lm`/vLLM load failures recorded |
| `bf16` | ok | `28.854` | `3594.309` | `11.790` | external per-quant failure artefact still missing |

This matrix is a loader and short-latency smoke, not production acceptance
evidence. The seven-format gate remains open until the raw go-mlx rows are
recovered or rerun and the missing external per-quant rows are either measured
or recorded as explicit command/version/error failures.

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

1. Close the `mlx_lm` gap or isolate the specific native cause. The most likely
   live boundary is evaluated graph/kernel work in the long-context path, not
   prompt-cache restore.
2. Produce a fair cached-prefix llama.cpp row or document why llama.cpp cannot
   run that same retained workflow.
3. Recover or rerun the seven raw go-mlx quant JSON artefacts, then fill the
   missing external rows for `mxfp4`, `mxfp8`, `5bit`, `6bit`, and `bf16` with
   command, runner version, and exact load error.
4. Prune or quarantine abandoned runtime fragments after the canonical rows
   above are no longer needed for investigation.
