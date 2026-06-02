<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B q6 go-mlx Self-Benchmark

Date: 2026-06-02

Purpose: compare the current go-mlx q6 default gate set against the same binary
with candidate gates isolated. This is a go-mlx-vs-go-mlx regression guard, not
an external runner parity claim.

Model:

`mlx-community/gemma-4-e2b-it-6bit`
snapshot `40d43b05f94ee798c0e40fe19fcd9ef49928486b`

Command shape:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx driver-profile \
  -report-file /private/tmp/go-mlx-self/q6-default-current.json \
  -include-output=false \
  -context 4096 \
  -prompt "Write a concise technical note explaining why retained model state matters for repeated agent workflows." \
  -max-tokens 512 \
  -runs 3 \
  -estimate-power-watts 75 \
  -trace-token-phases=false \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b
```

The baseline adds `-fast-gemma4-lane=false`. Isolated gate rows also add one of
`-direct-greedy-token` or `-paged-decode-fast-concat` while keeping
`-fast-gemma4-lane=false`.

Raw JSON was written during the run to:

- `/private/tmp/go-mlx-self/q6-default-current.json`
- `/private/tmp/go-mlx-self/q6-baseline-fast-off-current.json`
- `/private/tmp/go-mlx-self/q6-direct-greedy-only-current.json`
- `/private/tmp/go-mlx-self/q6-paged-fast-concat-only-current.json`
- `/private/tmp/go-mlx-self/q6-default-patched-rerun.json`

## Results

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak RSS | Active+cache | Output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | none | 1164 | 90.05828489228817 | 13.135163001s | 985.137225075 J | 4202987520 B | 4414100477 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Direct greedy only | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | 90.00163414721517 | 13.139377125s | 985.453284375 J | 4221779968 B | 4414034316 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Paged fast concat only | `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | 89.1661883374603 | 13.267558418s | 995.06688135 J | 4203577344 B | 4413838333 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Previous default combination | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | 66.04699256254571 | 19.379504541s | 1453.462840575 J | 4175757312 B | 4414659581 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Patched default rerun | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | 89.05331759347916 | 13.275426084s | 995.6569563 J | 4215341056 B | 4413571468 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |

The current fresh binary rejects the combined short-context default. Each gate
is individually healthy on this q6 prompt shape, but the combination drops
decode to `0.7334x` of the baseline and raises wall time and estimated energy by
`47.54%`.

The promoted short-context default is therefore narrowed to
`GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`. The patched default rerun confirms the
CLI default itself now reports only that gate and returns to the baseline band.
`GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` remains available as an explicit
diagnostic flag and should only be promoted again after a retained-workflow
self-benchmark shows a net wall-time win.

All four rows produced the same generated token hash and `1164` total generated
tokens, so the gate decision is about runtime cost, not visible output drift.
