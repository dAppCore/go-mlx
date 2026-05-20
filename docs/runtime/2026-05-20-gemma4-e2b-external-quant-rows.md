<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Gemma 4 E2B External Quant Rows

This note refreshes the external-runner side of the seven-format
`mlx-community` Gemma 4 E2B matrix. The go-mlx rows live in
`docs/runtime/2026-05-20-gemma4-e2b-quant-matrix.md`.

The matrix shape is the current short compatibility profile: README-sized
prompt, `2205` prompt tokens on the go-mlx chat-template path, `context=32768`,
and `128` generated tokens where the external runner can reach generation.
Strict loader failures use a one-token prompt/output because generation is
unreachable; the command and loader error are the measured result.

## Runner Versions

| Runner | Version evidence |
| --- | --- |
| `mlx_lm.generate` | `mlx 0.31.2`, `mlx_lm 0.31.3` from `/private/tmp/go-mlx-mlx-lm-venv` |
| vLLM Metal | `vllm 0.20.0+cpu`, `vllm_metal 0.2.0`, `mlx 0.31.2`, `mlx_lm 0.31.3` |
| llama.cpp | `llama-bench` build `660b1b4bd`, build number `8990`, backends `BLAS,MTL`, GPU `Apple M3 Ultra` |

All Metal commands were run from `/private/tmp` with direct Metal access. The
non-escalated sandbox path reports no Metal device for Python/Metal runners, so
those sandbox-only errors are not counted as runner compatibility evidence.

## Summary

| Quant | `mlx_lm.generate` | vLLM Metal | llama.cpp comparable row |
| --- | --- | --- | --- |
| `mxfp4` | fail: strict load rejects `100` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `40` extra shared-K/V scale tensors | no direct GGUF equivalent |
| `mxfp8` | fail: strict load rejects `100` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `40` extra shared-K/V scale tensors | no direct GGUF equivalent |
| `4bit` | fail: strict load rejects `140` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `80` extra shared-K/V quant tensors | `Q4_K_M`: `4294.342 tok/s` prefill, `143.952 tok/s` decode |
| `5bit` | fail: strict load rejects `140` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `80` extra shared-K/V quant tensors | no direct GGUF equivalent |
| `6bit` | fail: strict load rejects `140` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `80` extra shared-K/V quant tensors | no direct GGUF equivalent |
| `8bit` | fail: strict load rejects `140` extra shared-K/V tensors | fail: Metal engine reaches MLX device, then strict load rejects `80` extra shared-K/V quant tensors | `Q8_0`: `4460.410 tok/s` prefill, `122.513 tok/s` decode |
| `bf16` | fail: strict load rejects `60` extra shared-K/V tensors | ok: `3.571706959s` one-batch latency for `input_len=2205`, `output_len=128` | no direct BF16 GGUF row in the local cache |

`mlx_lm.generate` and vLLM Metal fail for related but not identical reasons.
The standalone MLX-LM model sees the full shared-K/V tensor set as extra
weights. The vLLM Metal adapter first forces the model into a text-only
backbone, so BF16 can load, while quantised variants still expose unsupported
K/V quant sidecars to the strict MLX-LM loader.

## Commands And Error Text

`mlx_lm.generate` command shape:

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-mlx-lm-venv/bin/mlx_lm.generate \
  --model <snapshot> \
  --prompt "Answer with one word: ready" \
  --max-tokens 1 \
  --verbose True
```

Measured `mlx_lm.generate` failures:

- `mxfp4` and `mxfp8`: exit `1`, `ValueError: Received 100 parameters not in model`, including `language_model.model.layers.15.self_attn.k_norm.weight`, `k_proj.scales`, `k_proj.weight`, `v_proj.scales`, and `v_proj.weight` through layer `34`.
- `4bit`, `5bit`, `6bit`, and `8bit`: exit `1`, `ValueError: Received 140 parameters not in model`, including `k_norm.weight`, `k_proj.biases`, `k_proj.scales`, `k_proj.weight`, `v_proj.biases`, `v_proj.scales`, and `v_proj.weight` through layer `34`.
- `bf16`: exit `1`, `ValueError: Received 60 parameters not in model`, including `k_norm.weight`, `k_proj.weight`, and `v_proj.weight` through layer `34`.

vLLM Metal command shape:

```sh
env VLLM_LOGGING_LEVEL=ERROR \
  /Users/snider/.venv-vllm-metal/bin/vllm bench latency \
  --model <snapshot> \
  --max-model-len 32768 \
  --input-len 2205 \
  --output-len 1 \
  --batch-size 1 \
  --num-iters 1 \
  --num-iters-warmup 0
```

Measured vLLM Metal failures:

- `mxfp4` and `mxfp8`: exit `1`, Metal engine starts and reports `MLX device set to: Device(gpu, 0)`, then `ValueError: Received 40 parameters not in model`, including `k_proj.scales` and `v_proj.scales` through layer `34`.
- `4bit`, `5bit`, `6bit`, and `8bit`: exit `1`, Metal engine starts and reports `MLX device set to: Device(gpu, 0)`, then `ValueError: Received 80 parameters not in model`, including `k_proj.biases`, `k_proj.scales`, `v_proj.biases`, and `v_proj.scales` through layer `34`.

vLLM BF16 command:

```sh
env VLLM_LOGGING_LEVEL=ERROR \
  /Users/snider/.venv-vllm-metal/bin/vllm bench latency \
  --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09 \
  --max-model-len 32768 \
  --input-len 2205 \
  --output-len 128 \
  --batch-size 1 \
  --num-iters 1 \
  --num-iters-warmup 0
```

BF16 result:

```text
Avg latency: 3.5717069590464234 seconds
10% percentile latency: 3.5717069590464234 seconds
25% percentile latency: 3.5717069590464234 seconds
50% percentile latency: 3.5717069590464234 seconds
75% percentile latency: 3.5717069590464234 seconds
90% percentile latency: 3.5717069590464234 seconds
99% percentile latency: 3.5717069590464234 seconds
```

llama.cpp Q4_K_M command:

```sh
llama-bench \
  -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q4_K_M.gguf \
  -p 2205 \
  -n 128 \
  -r 3 \
  -ngl 99 \
  -fa 1 \
  -o json
```

Q4_K_M result:

```text
pp2205: avg_ts=4294.341924 tok/s, samples=[4306.07, 4281.34, 4295.62]
tg128:  avg_ts=143.952145 tok/s, samples=[142.078, 143.695, 146.084]
```

llama.cpp Q8_0 command:

```sh
llama-bench \
  -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q8_0.gguf \
  -p 2205 \
  -n 128 \
  -r 3 \
  -ngl 99 \
  -fa 1 \
  -o json
```

Q8_0 result:

```text
pp2205: avg_ts=4460.410077 tok/s, samples=[4458.04, 4456.41, 4466.78]
tg128:  avg_ts=122.512802 tok/s, samples=[122.175, 122.152, 123.211]
```

## Gate Impact

This closes the seven-format external compatibility ledger for the short E2B
matrix. It does not close the production runner-anchor gate, because the
accepted workflow is the 100k retained repeated workload and `mlx_lm` still
wins that same-shape cached workflow.
