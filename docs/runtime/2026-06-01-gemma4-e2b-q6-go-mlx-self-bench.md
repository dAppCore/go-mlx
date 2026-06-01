<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B q6 go-mlx Self-Benchmark

Date: 2026-06-01

Purpose: compare current go-mlx default fast-lane behaviour against the same
binary with `-fast-gemma4-lane=false`. This is a go-mlx-vs-go-mlx regression
guard, not an external runner parity claim.

Model:

`mlx-community/gemma-4-e2b-it-6bit`
snapshot `40d43b05f94ee798c0e40fe19fcd9ef49928486b`

Command shape:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx driver-profile \
  -json \
  -include-output=false \
  -context 4096 \
  -prompt "Write a concise technical note explaining why retained model state matters for repeated agent workflows." \
  -max-tokens 512 \
  -runs 3 \
  -estimate-power-watts 75 \
  -trace-token-phases=false \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b
```

Baseline adds:

```bash
-fast-gemma4-lane=false
```

Raw JSON was written during the run to:

- `/private/tmp/go-mlx-self/q6-baseline-fast-off-notrace.json`
- `/private/tmp/go-mlx-self/q6-default-fast-on.json`

## Results

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak RSS | Active+cache | Output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | none | 1164 | 89.11955735839634 | 13.266706126s | 995.00295945 J | 4222173184 B | 4413903244 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Default | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | 89.49273128756029 | 13.213508334s | 991.01312505 J | 4165910528 B | 4413737981 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |

The default lane is a small positive self-benchmark on this q6 prompt shape:
`1.0042x` raw decode, `0.40%` lower wall time and estimated energy, and the
same generated token hash. The natural stop produced `388` generated tokens per
run, so this is a multi-hundred-token decode sample rather than a tiny smoke,
but it is still a short-context regression guard and not a retained-workflow
acceptance benchmark.

The cache profile stayed bounded in both runs: `local_window_tokens=512`,
`local_window_leaked=false`, `12` local caches, `3` global caches, and `20`
shared layers.
