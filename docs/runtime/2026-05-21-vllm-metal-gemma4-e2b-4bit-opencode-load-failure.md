<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# vLLM Metal Opencode Load Failure

Date: 2026-05-21

This is the same-shape vLLM Metal attempt for the opencode-sized Gemma 4 E2B
4-bit runner gate. It uses the accepted interactive prompt shape length
(`31034` initial prompt tokens plus `1024` output tokens) against the same
`mlx-community/gemma-4-e2b-it-4bit` snapshot used by the accepted go-mlx row.

## Command

```sh
/Users/snider/.venv-vllm-metal/bin/vllm bench latency \
  --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd \
  --max-model-len 131072 \
  --input-len 31034 \
  --output-len 1024 \
  --batch-size 1 \
  --num-iters 1 \
  --num-iters-warmup 0 \
  --dtype bfloat16
```

## Result

The command exits with status `1` before latency measurement.

Observed setup:

- vLLM reports `v0.20.0`.
- The Metal platform plugin activates.
- The resolved architecture is `Gemma4ForConditionalGeneration`.
- Chunked prefill is enabled with `max_num_batched_tokens=16384`.
- The Metal worker reaches `MLX device set to: Device(gpu, 0)`.
- Available Metal memory is reported as `72.5GB`.

Failure:

```text
ValueError: Received 80 parameters not in model:
language_model.model.layers.15.self_attn.k_proj.biases,
language_model.model.layers.15.self_attn.k_proj.scales,
language_model.model.layers.15.self_attn.v_proj.biases,
language_model.model.layers.15.self_attn.v_proj.scales,
...
language_model.model.layers.34.self_attn.k_proj.biases,
language_model.model.layers.34.self_attn.k_proj.scales,
language_model.model.layers.34.self_attn.v_proj.biases,
language_model.model.layers.34.self_attn.v_proj.scales.
```

Verdict: vLLM Metal cannot currently run the same opencode-sized E2B 4-bit
workflow on this model snapshot. The failure is a strict `mlx_lm` model-load
compatibility issue for the Gemma 4 shared/global K/V tensors, not a runtime
throughput result.
