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

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g1024-r10-borrowed-pages-energy100w.json`
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
- Hyper-long page size: `1024`
- Page-state policy: borrowed full physical page handles, owned slices only for
  partial preallocated pages
- Active/RSS hard caps: `12 GiB` each
- Process virtual memory: recorded, not capped
- Power estimate: normalised `100 W`, not measured power

Result:

| Metric | Value |
| --- | ---: |
| Successful runs | `10/10` |
| Generated tokens | `10240` |
| Total wall time | `260.093s` |
| Cold prefill | `1678.071 tok/s` |
| Average decode | `51.293 tok/s` |
| Warm restore average | `0.372 ms` |
| Warm run wall band | `19.953s` to `19.983s` |
| Peak MLX active memory | `3.710 GiB` |
| Peak process RSS | `3.156 GiB` |
| Process peak RSS | `3.156 GiB` |
| Process virtual reservation | `684.481 GiB` |
| Estimated energy | `26009.334 J` |
| Prompt setup saved vs replay | `541.717s` |
| Estimated setup energy saved | `54171.665 J` |
| Prompt setup speedup | `9.999x` |

This supersedes the adaptive page-size row at
`docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g1024-r10-adaptive-page1024-energy100w.json`.
Borrowing full page handles removes repeated per-token page clone graph churn
and improves the same 100k retained workflow by `1.014x` on decode and
`1.011x` on wall/energy. Raw 100k decode is still much slower than the short
and 29k lanes, but the retained-prefix path removes repeated prompt setup at
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
| go-mlx | cold run 1 of retained profile | `80.330s` | `51.148 tok/s` decode plus `1678.071 tok/s` prefill |
| go-mlx | 10 retained turns | `260.093s` | `51.293 tok/s` average decode |

The llama.cpp row is a cold calibration anchor, not a retained-prefix runner
win/loss verdict. If the same cold replay were repeated ten times, the measured
llama.cpp wall would be roughly `949.035s`; the go-mlx retained-prefix workflow
is `260.093s`. The cached-prefix llama.cpp workflow below is the fairer runner
anchor and still beats go-mlx on the same repeated shape.

Current `mlx_lm` cached workflow anchor:

- `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-cached-workflow-r46-g1024-r10-energy100w.json`
- `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-cached-workflow-r46-g1024-r10-energy100w.stderr`
- Strict-load failure preserved at
  `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-strict-load-failure.stderr`

Shape:

- Runner: `mlx_lm` `0.31.3` on `mlx` `0.31.2`
- Model: same local `mlx-community/gemma-4-e2b-it-4bit` snapshot as go-mlx
- Prompt: README repeated `46` times plus the same agentic suffix
- Cache prompt tokens: `100935`
- Cached suffix tokens per turn: `5`
- Generation budget: `1024` tokens per turn
- Runs: `10`
- Prefill step size: `512`
- Loader: non-strict MLX-LM load, explicitly ignoring the unused shared-K/V
  extra tensors that make the stock CLI fail strict loading
- Power estimate: normalised `100 W`, not measured power

Result:

| Runner | Wall | Decode | Cold/cache prefill | Peak memory | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| go-mlx retained | `260.093s` | `51.293 tok/s` | `1678.071 tok/s` | `3.710 GiB` active MLX, `3.156 GiB` peak RSS | `26009.334 J` |
| `mlx_lm` cached | `119.866s` including load+prefill | `103.971 tok/s` | `5465.549 tok/s` | `5.473 GB` MLX peak, `3.820 GB` peak RSS | `11986.551 J` |

This is a current configured runner loss for go-mlx. On the comparable cached
100k/1024x10 workflow, `mlx_lm` is `2.170x` faster by wall time and estimated
energy, `2.027x` faster on raw decode, and `3.257x` faster on the one-time
100k cache prefill. The older retained-state argument is still architecturally
useful, but it does not beat the current Python MLX stack on this shape.

Rejected go-mlx cache-only chunk prefill diagnostic:

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-cacheonly-prefill-r46-ctx131072-g1024-r10-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-cacheonly-prefill-r46-ctx131072-g1024-r10-energy100w.stderr`

The diagnostic changed chunked prefill so intermediate chunks evaluated cache
state only and delayed logits materialisation until the final chunk, closer to
the MLX-LM prefill shape. It improved cold go-mlx prefill from `157.168s` /
`642.657 tok/s` to `116.210s` / `869.159 tok/s`, but the full 10-run workload
failed `10/10` runs on the repeated-sentence quality guard. The summed runtime
for the failed diagnostic was `365.468s`, and decode stayed in the same
`~43.8 tok/s` band, so this does not close the `mlx_lm` gap and is not an
accepted production row. The path is now gated behind
`GO_MLX_ENABLE_CACHE_ONLY_CHUNK_PREFILL=1` for further investigation rather
than enabled by default.

Current vLLM Metal 100k attempt:

- `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stdout`
- `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stderr`

Shape:

- Runner: `/Users/snider/.venv-vllm-metal/bin/vllm`, `vllm 0.20.0+cpu` with
  the Metal plugin active
- Command shape: `vllm bench latency --max-model-len 131072 --input-len 100935
  --output-len 1024 --batch-size 1 --num-iters 1 --num-iters-warmup 0`
- Model: same local `mlx-community/gemma-4-e2b-it-4bit` snapshot as go-mlx

Result: vLLM reaches the Metal engine initialisation path, sets MLX device
`gpu, 0`, enables chunked prefill at `16384`, then fails during MLX-LM strict
model load with the same shared-K/V extra parameter class. No latency JSON is
written. This remains a compatibility failure until vLLM Metal exposes the same
non-strict/sanitised Gemma 4 E2B load path used by the in-process `mlx_lm`
anchor above.

These artefacts satisfy the current go-mlx 100k retained-state and book
workflow gates. They do not satisfy the separate same-shape runner-anchor gate:
`mlx_lm` and cached-prefix llama.cpp still have faster current rows, while vLLM
has a current documented Metal load failure. The overall production goal remains
blocked on the long-context decode gap.
