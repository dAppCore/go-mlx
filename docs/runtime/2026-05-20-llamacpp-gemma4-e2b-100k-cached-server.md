<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 llama.cpp Gemma 4 E2B 100k Cached Server Anchor

This note records the current same-shape llama.cpp retained-prefix anchor for
the E2B production lane. It supersedes the cold-only llama.cpp row as the
runner-anchor evidence, while keeping the cold row as calibration context.

## Shape

- Runner: `llama-server`, build `b8990-660b1b4bd`
- Model: `unsloth/gemma-4-E2B-it-GGUF`, `Q4_K_M`
- Prompt: `README.md` repeated `46` times with `\n\n` separators, then
  `docs/runtime/2026-05-20-agentic-long-turn-suffix.md`
- Prompt bytes: `325754`
- Prompt tokens reported by llama.cpp: `100926`
- Context: `131072`
- Runs: `10`
- Generated tokens per run: `1024`
- Sampling: `temperature=0`, `top_k=1`, `top_p=1`, `min_p=0`,
  `repeat_penalty=1`, `ignore_eos=true`
- Power estimate: normalised `100 W`, not measured power

## Server Command

```sh
llama-server \
  -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q4_K_M.gguf \
  -c 131072 \
  -ngl 99 \
  -fa on \
  --host 127.0.0.1 \
  --port 18080 \
  --no-webui \
  --metrics \
  --slots \
  --cache-prompt \
  --cache-reuse 2048 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --ctx-checkpoints 32 \
  --checkpoint-every-n-tokens 8192 \
  --cache-ram -1 \
  --no-warmup \
  --timeout 1200
```

The server reported `cache_reuse is not supported by this context`, so that
knob was disabled. Prompt cache remained enabled with no RAM limit, and warm
turns restored the last checkpoint before evaluating the final `5` prompt
tokens.

## Result

| Metric | Value |
| --- | ---: |
| Successful runs | `10/10` |
| Generated tokens | `10240` |
| Total wall | `214.2053115828894s` |
| Decode | `82.6804811755317 tok/s` |
| First prefill | `100926` tokens in `89.121828s`, `1132.4498415808976 tok/s` |
| Warm prompt cache | `100921` cached tokens average, `45.59077777777778ms` prompt work average |
| Wall visible throughput | `47.80460355688941 tok/s` |
| Peak RSS | `4762075136` bytes |
| Peak VSZ | `458686627840` bytes |
| Energy at `100 W` | `21420.53115828894 J` |

Against the accepted go-mlx retained row (`408.482573s`, `43.617197954723096
tok/s` decode), the cached llama.cpp server is `1.906x` faster by wall time and
`1.895x` faster by decode. Against the configured `mlx_lm` cached row
(`119.86551008420065s`, `103.97136858101358 tok/s` decode), llama.cpp is
`1.787x` slower by wall time and `1.258x` slower by decode.

## Artefact

- `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-100k-cached-server-r10-g1024-energy100w.json`

## Gate Impact

This closes the same-shape llama.cpp runner-anchor gap for the accepted
100k retained workflow. It does not close production: both `mlx_lm` and
llama.cpp now beat go-mlx on the same retained workflow, so the long-context
decode/prefill path remains the active optimisation boundary.
