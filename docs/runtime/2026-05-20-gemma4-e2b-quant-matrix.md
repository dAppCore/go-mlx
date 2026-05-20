<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Gemma 4 E2B go-mlx Quant Matrix

This note supersedes the replay state of
`docs/runtime/2026-05-19-gemma4-e2b-quant-matrix.md` for go-mlx raw artefacts.
It uses the rebuilt current `lthn-mlx` binary after adding `driver-profile
-report-file` and fixing lazy float32 host-logit materialisation.

## Shape

- Prompt: `README.md` through the Gemma 4 chat template
- Prompt tokens: `2205`
- Context: `32768`
- Cache mode: `paged`
- Prefill chunk size: `512`
- Runs: `3`
- Generated tokens per run: `128`
- Output capture: disabled
- Power estimate: normalised `100 W`, not measured power
- Working directory: `/private/tmp`
- Metal library: `MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib`

The command shape for each row was:

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile \
  -report-file docs/runtime/<row>.json \
  -prompt-file /Users/snider/Code/core/go-mlx/README.md \
  -max-tokens 128 \
  -runs 3 \
  -include-output=false \
  -estimate-power-watts 100 \
  -context 32768 \
  -prefill-chunk-size 512 \
  -cache-mode paged \
  <snapshot>
```

## Results

| Quant | Status | Decode tok/s | Cold prefill tok/s | Wall s | Peak GiB | Active GiB | RSS GiB | Energy J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4bit` | ok | `107.914` | `2600.048` | `4.422` | `7.660` | `7.593` | `3.147` | `442.202` |
| `5bit` | ok | `76.489` | `2412.525` | `5.946` | `4.719` | `4.108` | `3.723` | `594.579` |
| `6bit` | ok | `73.411` | `2297.405` | `6.203` | `5.446` | `4.841` | `4.269` | `620.310` |
| `8bit` | ok | `78.326` | `2082.905` | `5.976` | `6.338` | `5.557` | `5.367` | `597.557` |
| `bf16` | ok | `27.703` | `1366.643` | `15.503` | `16.179` | `13.797` | `9.361` | `1550.289` |
| `mxfp4` | ok after materialisation fix | `84.282` | `3094.590` | `5.283` | `4.794` | `4.651` | `3.854` | `528.336` |
| `mxfp8` | ok | `74.631` | `2102.044` | `6.208` | `6.256` | `5.362` | `5.219` | `620.774` |

## Artefacts

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-5bit-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-6bit-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-8bit-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-bf16-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-mxfp4-current-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-mxfp8-current-quant-matrix-3run-readme-energy100w.json`

## MXFP4 Crash Fix

The first MXFP4 rerun crashed in `mlx_array_data_float32` while the
suppressed-token guard fell back to a host-side greedy scan of lazy float32
logits. `Array.Floats()` now materialises the row-contiguous source before raw
`mlx_array_data_float32` access and returns an empty slice instead of walking a
nil data pointer. The same MXFP4 row then completed `3/3` runs.

## Open External Rows

This file closes the raw go-mlx side of the seven-format matrix. The matrix
production gate remains open until external runner rows are refreshed:

- llama.cpp comparable anchors for `4bit` and `8bit` remain the GGUF
  `Q4_K_M`/`Q8_0` rows in the older matrix note.
- `mlx_lm` and vLLM Metal need current per-quant command/version/error
  artefacts for each unsupported or incompatible MLX-community snapshot.
- There is no direct llama.cpp equivalent for MLX `mxfp4`, `mxfp8`, `5bit`,
  `6bit`, or `bf16`; those rows should be labelled as nearest-comparable or
  unsupported rather than silently omitted.
