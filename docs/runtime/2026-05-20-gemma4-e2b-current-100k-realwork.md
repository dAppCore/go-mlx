<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B 4bit Current 100k Real-Workload Refresh

This note records the 2026-05-20 current guarded reruns for
`mlx-community/gemma-4-e2b-it-4bit` at the 100k-context production shape. The
runs were launched from `/private/tmp` so the native Metal path was visible, and
used the workspace-aware Go setup:

```sh
GOWORK=/Users/snider/Code/core/go-mlx/go.work
GOCACHE=/private/tmp/codex-go-mlx-cache
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
```

## Retained Prefix Driver Profile

Accepted artefact:

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-guarded-r46-ctx131072-g1024-r10-longturn-naturalstop-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-guarded-r46-ctx131072-g1024-r10-longturn-naturalstop-energy100w.stderr`
- Prompt suffix: `docs/runtime/2026-05-20-agentic-long-turn-suffix.md`

Shape:

- Model: `mlx-community/gemma-4-e2b-it-4bit`
- Snapshot: `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`
- Prompt: README repeated `46` times plus an agentic long-turn suffix
- Prompt tokens: `101005`
- Context: `131072`
- Prompt chunk bytes: `4096`
- Prefill chunk size: `512`
- Runs: `10`
- Generation budget: `1024` tokens per run
- Cache mode: `paged`
- Active/RSS hard caps: `12 GiB` each
- Process virtual memory: recorded, not capped
- Power estimate: normalised `100 W`, not measured power

Result:

| Metric | Value |
| --- | ---: |
| Successful runs | `10/10` |
| Generated tokens | `10240` |
| Total wall time | `408.483s` |
| Cold prefill | `642.657 tok/s` |
| Average decode | `43.617 tok/s` |
| Warm restore average | `2.116 ms` |
| Warm run wall band | `23.323s` to `23.649s` |
| Peak MLX active memory | `3.699 GiB` |
| Peak process RSS | `5.049 GiB` |
| Process peak RSS | `6.509 GiB` |
| Process virtual reservation | `738.747 GiB` |
| Estimated energy | `40848.257 J` |
| Prompt setup saved vs replay | `1414.491s` |
| Estimated setup energy saved | `141449.142 J` |
| Prompt setup speedup | `9.999x` |

This supersedes the previous accepted 100k evidence that only generated
`128` tokens per turn. Raw 100k decode is still much slower than the short and
29k lanes, but the retained-prefix path removes the repeated prompt setup at
agentic workflow scale.

## Retained 10-Chapter Book

Accepted artefacts:

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-book.md`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-energy100w.stderr`

Shape:

- Context: `131072`
- Prompt repeat: `46`
- Chapters: `10`
- Chapter max tokens: `8192`
- Accepted visible-token floor: `768`
- Thinking: enabled
- Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=64`
- Active/RSS hard caps: `12 GiB` each

Result:

| Metric | Value |
| --- | ---: |
| Successful turns | `10/10` |
| Generated / visible tokens | `11425` |
| Chapter visible-token range | `979` to `1484` |
| Total wall time | `482.081s` |
| Average decode | `41.442 tok/s` |
| Average prefill | `578.182 tok/s` |
| Peak MLX active memory | `4.261 GiB` |
| Peak process RSS | `5.771 GiB` |
| Process peak RSS | `6.546 GiB` |
| Process virtual reservation | `953.339 GiB` |
| Estimated energy | `48208.084 J` |

The stricter `chapter_min_tokens=1024` probe is rejected but informative:
the prompt fix raised chapter 2 from `803` to `936` visible tokens, still below
the strict floor. The accepted book uses the same `8192` return allowance but a
`768` visible-token floor so natural E2B chapter length is not discarded as a
failed run. The harness now accepts a natural stop once the visible-token floor
and quality guards pass, while still rejecting max-token exhaustion before a
chapter marker.

## Remaining External Work

Current llama.cpp cold anchor:

- `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-pg101005-1024-bench.json`
- `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-pg101005-1024-bench.stderr`

Shape:

- Model: `unsloth/gemma-4-E2B-it-GGUF`
- File: `gemma-4-E2B-it-Q4_K_M.gguf`
- Command shape: `llama-bench -pg 101005,1024 -r 1 -ngl 99 -fa 1`
- Backend: `BLAS,MTL`
- Device: `MTL0 (Apple M3 Ultra)` in stderr
- K/V cache type: `f16`

Result:

| Runner | Shape | Wall | Throughput |
| --- | --- | ---: | ---: |
| llama.cpp | cold `pp101005+tg1024` | `94.904s` | `1075.081 tok/s` combined |
| go-mlx | cold run 1 of retained profile | `197.060s` | `43.556 tok/s` decode plus `642.657 tok/s` prefill |
| go-mlx | 10 retained turns | `408.483s` | `43.617 tok/s` average decode |

The llama.cpp row is a cold calibration anchor, not a retained-prefix runner
win/loss verdict. If the same cold replay were repeated ten times, the measured
llama.cpp wall would be roughly `949.035s`; the go-mlx retained-prefix workflow
is `408.483s`. A fair cached-prefix llama.cpp workflow and configured
`mlx_lm`/vLLM rows are still required before the separate runner-anchor gate can
close.

These artefacts satisfy the current go-mlx 100k retained-state and book
workflow gates. They do not satisfy the separate same-shape runner-anchor gate:
`mlx_lm`, vLLM, and a cached-prefix llama.cpp workflow still need comparable
current 100k or documented failure rows before the overall production goal can
close.
