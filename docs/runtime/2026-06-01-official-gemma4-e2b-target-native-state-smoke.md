<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Official Gemma 4 E2B Target Native State Smoke

Date: 2026-06-01
Binary: `/private/tmp/go-mlx-self/bin/lthn-mlx`

Purpose: prove the locked official Google Gemma 4 E2B target loads through the
native go-mlx Metal path and exercises the retained-State, prompt-cache, K/V
restore, and state-bundle contracts. This is a go-mlx smoke/contract artefact,
not a production throughput benchmark and not an external-runner comparison.

Model:

`google/gemma-4-E2B-it`
snapshot `905e84b50c4d2a365ebde34e685027578e6728db`

## Command

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx bench -json \
  -prompt "Continue this sentence in plain English: Retained state matters because" \
  -cache-prompt "Retained state keeps prior project context available without replaying the whole source prompt." \
  -max-tokens 32 \
  -runs 1 \
  -context 4096 \
  -no-probes \
  -state-kv-warm \
  -state-kv-block-size 512 \
  -state-kv-store /private/tmp/go-mlx-self/official-e2b-state-2.kv \
  /Users/snider/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db
```

The command exited successfully.

## Result

| Field | Value |
| --- | ---: |
| architecture | `gemma4_text` |
| vocab size | `262144` |
| layers | `35` |
| hidden size | `1536` |
| context | `4096` |
| generated text | ` they affect how things are done.` |
| prompt tokens | `13` |
| generated tokens | `7` |
| first token | `371.619583ms` |
| prefill | `35.03194078970354 tok/s` |
| decode | `28.043153814643514 tok/s` |
| total generation wall | `620.705375ms` |
| peak memory | `11871132097 B` |
| active memory | `9296004678 B` |

Quality flags passed: non-empty output and generated-token count were both
true. A previous prompt shape in the same session generated `0` primary tokens
and was rejected as generation evidence; this artefact records the successful
native target smoke.

## Prompt Cache

| Field | Value |
| --- | ---: |
| attempted | `true` |
| cache misses | `1` |
| miss tokens | `17` |
| warm duration | `54.715125ms` |
| generated tokens | `32` |
| prefill | `311.724036358318 tok/s` |
| decode | `29.15960979430287 tok/s` |
| peak memory | `11878590521 B` |
| active memory | `9296774726 B` |

## State K/V Warm Path

| Field | Value |
| --- | ---: |
| attempted | `true` |
| source | `state/file-log` |
| block size | `512` |
| total blocks | `1` |
| store bytes | `2515324 B` |
| build duration | `57.120167ms` |
| build tokens | `17` |
| build prefill | `297.61817748186905 tok/s` |
| blocks read | `1` |
| chunks read | `1` |
| prefix tokens restored | `17` |
| baseline prefill duration | `371.089917ms` |
| restore duration | `1.758417ms` |
| generate duration | `1.153725125s` |
| prefill saved per question | `369.3315ms` |
| restore speedup | `211.03635656388673x` |
| memory peak | `11899320377 B` |
| generated tokens | `32` |
| decode | `29.130996201816963 tok/s` |

K/V restore also completed independently in `883.958us`. State bundle export
completed in `110.9205ms` and wrote `8961138` bytes.

## Scope

This closes the official target native-load plus retained-State contract smoke
for the locked source snapshot. It does not promote the official E2B lane to
production and does not replace the q6 go-mlx self-benchmark. The target-only
versus MTP retained-workflow benchmark, TurboQuant validation, and regenerated
canonical benchmark set remain separate gates.
