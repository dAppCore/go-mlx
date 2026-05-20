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

## Token-Phase Trace

A same-shape one-run trace was recorded with `GO_MLX_TRACE_FORWARD_EVAL=1` and
`driver-profile -trace-token-phases` on the accepted README-repeat 100k shape.
The raw trace is intentionally not tracked because it is about `17 MB`, but the
compact derived note is tracked at
`docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-token-phase-trace-summary.md`.

The trace itself slows decode to `19.026 tok/s`, so it is diagnostic rather
than a replacement for the accepted untraced `51.293 tok/s` row. The bucket
split is still decisive: out of `53.817s` traced decode-loop time, `53.084s`
is forward materialisation. Native event totals rank attention first at
`22.745s`, then output at `10.643s`, FFN at `9.909s`, and attention residual at
`7.817s`.

The expensive attention layers are exactly the full-attention owners in the
Gemma 4 local/full pattern: layers `4`, `9`, `14`, `19`, `24`, `29`, and `34`
sit around `1.8-2.0ms` each per traced token, while local sliding-attention
layers sit near the `0.3-0.4ms` band. The next implementation target should
therefore stay focused on the full-attention paged/global K/V path.

## Rejected 100k Branches

Three same-shape `100k` / `1024` one-run probes now bound the obvious branches:

| Probe | Shape | Result | Verdict |
| --- | --- | ---: | --- |
| Paged K/V without fast concat | `100937` prompt tokens, paged K/V `1024`, accepted fast gates except `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` | `106.324s` wall, `22.956 tok/s` decode, `1638.525 tok/s` prefill, `3.640 GiB` active MLX | Rejected. Avoiding the concat makes the per-page Go/MLX attention graph much slower than the accepted borrowed-page fast-concat lane. |
| Native C++ paged attention reduction | `100937` prompt tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION`, no fast concat | `104.572s` wall, `23.448 tok/s` decode, `1660.523 tok/s` prefill, `3.640 GiB` active MLX | Rejected. Moving the same page-reduction graph behind one C++ call trims only a little overhead; the missing path is a fused/custom paged-attention kernel. |
| Fixed cache with sliding layers bounded | `100937` prompt tokens, fixed Gemma 4 cache, shared mask, sliding cache bound, `12 GiB` active/RSS guards | Failed after `13` visible tokens; stream active memory hit `13748980782` bytes over the `12884901888` byte guard | Rejected. Hyper-long fixed cache is not the default path until a narrower global-only/native attention storage plan exists. |

The current boundary is therefore narrower than "turn off concat" or "restore
fixed cache": go-mlx needs a fused native paged/global-attention path that
avoids both per-token full K/V concatenation and the active-memory footprint of
a full fixed cache. A C++ wrapper around the existing page-reduction graph is
not enough.

## Replay Harness

Use `scripts/gemma4_context_ramp.sh` for the next context-scaling pass. The
tracked harness now defaults to the current E2B q4 production snapshot and uses
`driver-profile -report-file` so each row is emitted by the runner rather than
by shell stdout redirection. Override `GO_MLX_MODEL` and `GO_MLX_MODEL_LABEL`
when comparing E4B, 26B, or future model snapshots.

The next long-turn fairness pass should keep the accepted repeat/context ladder
but set `GO_MLX_RAMP_MAX_TOKENS=5120`. That measures the 100k warm-decode path
with a generation budget large enough to avoid another tiny-token smoke.
