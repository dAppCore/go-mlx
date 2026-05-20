<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Long-Context Gap Diagnosis

This note records the current answer to why go-mlx is still slower than
configured external runners on the accepted 100k retained workflow.

## Short Continuation Check

A current-source C006 regression check was built to
`/private/tmp/go-mlx-c006-regression/lthn-mlx` and run from `/private/tmp`
with the same C006 premise, `context=131072`, paged cache,
`prefill_chunk_size=512`, thinking enabled, and the accepted `512` visible-token
floor, but with `chapters=9`.

The run completed:

| Metric | Value |
| --- | ---: |
| Successful turns | `9/9` |
| Generated / visible tokens | `6851` |
| Total wall | `94.359181752s` |
| Average decode | `75.44102448821488 tok/s` |
| Average prefill | `2212.4547571311377 tok/s` |
| Active MLX memory | `3373521322` bytes |
| Cache memory | `6679911976` bytes |
| Process RSS | `3550920704` bytes |
| Process virtual reservation | `587977261056` bytes |
| Estimated energy at `100 W` | `9435.9181752 J` |

This does not reproduce a massive C006-path rollback. The nearby canonical
`92.814218749s` artefact was a stricter `chapter_min_tokens=640` neighbour that
reported `7` successful turns and failed on turn `8` because the model naturally
stopped at `563` visible tokens. The accepted `chapter_min_tokens=512` C006 run
completed `10/10` turns in `105.946990083s`.

## Production Gap

The slower path is the accepted 100k retained workflow, not the shorter C006
continuation lane. The first corrective change is now in the default fast lane:
hyper-long paged K/V caches use `1024`-token pages instead of the old `512`
default, and the CLI records that choice as
`GO_MLX_PAGED_KV_PAGE_SIZE=1024`.

| Runner | Shape | Warm per-turn decode | First prefill | Restore |
| --- | --- | ---: | ---: | ---: |
| go-mlx current | `101005` prompt tokens, `10x1024` retained turns, paged K/V `1024` | about `20.25s` per warm `1024` tokens, `50.566 tok/s` | `60.193s`, `1678.094 tok/s` | `0.365ms` average |
| go-mlx previous | `101005` prompt tokens, `10x1024` retained turns | about `23.4s` per `1024` tokens, `43.617 tok/s` | `157.168s`, `642.657 tok/s` | `2.116ms` average |
| llama.cpp server | `100926` prompt tokens, `10x1024` cached-prefix turns | about `12.5s` per `1024` tokens, `82.680 tok/s` | `89.122s`, `1132.450 tok/s` | `45.591ms` warm prompt work |
| `mlx_lm` | `100935` cached prompt tokens, `10x1024` turns | about `10.0s` per `1024` tokens, `103.971 tok/s` | about `18.5s`, `5465.549 tok/s` | cached prefix in-process |

The retained-state restore is already cheap enough that it is not the active
loss. The page-size correction improves the 100k row from `408.483s` to
`262.995s`, a `1.553x` wall/energy improvement, but the active loss is still
the evaluated long-context graph and kernel path:

- go-mlx cold 100k prefill is now `1.48x` faster than llama.cpp but still
  `3.26x` slower than the configured `mlx_lm` harness.
- go-mlx warm 100k decode remains `1.64x` slower than llama.cpp and `2.06x`
  slower than `mlx_lm`.
- The one-run token-phase trace records around `22ms` per generated token. Most
  of that wait is attributed under `cache_probe_duration`, but the label is
  misleading for the direct-greedy/async path: it is where the lazy next-token
  graph synchronises in practice, not evidence that prompt-cache restore is
  slow.

## Working Explanation

go-mlx has the retained-prefix architecture working, and the old paged-cache
block geometry was a real part of the long-context loss. The remaining 100k
decode path still evaluates a heavier per-token MLX graph than llama.cpp or
`mlx_lm`. The likely live boundary is full-attention K/V access and mask/graph
materialisation over a very large retained context, combined with the
paged-cache view/concat attention path. The shorter C006 path stays near the
useful `75-80 tok/s` band because it does not carry a 100k prompt prefix through
every generated token.

The next optimisation should target the 100k first-prefill and warm-decode
kernel path directly. Re-running small-context or short-output smokes will not
measure this boundary.

## Replay Harness

Use `scripts/gemma4_context_ramp.sh` for the next context-scaling pass. The
tracked harness now defaults to the current E2B q4 production snapshot and uses
`driver-profile -report-file` so each row is emitted by the runner rather than
by shell stdout redirection. Override `GO_MLX_MODEL` and `GO_MLX_MODEL_LABEL`
when comparing E4B, 26B, or future model snapshots.

The next long-turn fairness pass should keep the accepted repeat/context ladder
but set `GO_MLX_RAMP_MAX_TOKENS=5120`. That measures the 100k warm-decode path
with a generation budget large enough to avoid another tiny-token smoke.
