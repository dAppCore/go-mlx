# Quantised Models

go-mlx loads quantised safetensors and GGUF checkpoints transparently. The runtime detects per-tensor quantisation (4-bit AWQ, 8-bit symmetric, GGUF Q-quants) from the safetensors metadata or GGUF header, picks the right `QuantizedMatmul` kernel, and the rest of the model code is unchanged.

## Loading 4-bit Safetensors

Models exported by `mlx-lm` with `--quantize` carry `_scales` and `_biases` tensors alongside packed `weight` tensors. The loader detects these automatically:

```go
import (
    mlx "dappco.re/go/mlx"
)

model, err := mlx.LoadModel("/models/qwen3-8b-q4/",
    mlx.WithQuantization(4), // hint, also auto-detected
)
```

Per-layer quantisation is fine — non-quantised layers (typically `lm_head` and embeddings) are loaded as full precision and matmuls dispatch through the appropriate kernel per layer.

## Loading GGUF

A single GGUF file is a complete model pack — config, tokenizer, and weights all in one:

```go
model, err := mlx.LoadModel("/models/qwen3-8b-q4_k_m.gguf")
```

Architecture is read from the GGUF metadata (`general.architecture`); tokeniser is reconstructed from the embedded vocabulary, merge table, and special tokens.

Supported GGUF quant formats on read: `Q8_0`, `Q4_0`, `Q4_K_M` (and several others through the same dequant path).

## Inspecting GGUF Metadata Without Loading

```go
info, err := mlx.ReadGGUFInfo("/models/qwen3-8b-q4_k_m.gguf")
fmt.Printf("arch=%s vocab_size=%d quant=%s tensors=%d\n",
    info.Architecture, info.VocabSize, info.QuantFormat, info.TensorCount)
```

Useful for build pipelines that need to validate model packs before deploy.

## Producing GGUF From Safetensors

If you have a finetuned safetensors pack and want a GGUF checkpoint for cross-tool deployment, use `QuantizeModelPackToGGUF` — see [`../model-ops/quantize-gguf.md`](../model-ops/quantize-gguf.md).

## Memory Footprint Comparison (Qwen3-8B)

| Format | On-disk | RAM resident |
|--------|---------|--------------|
| BF16 safetensors | ~16 GB | ~16 GB |
| 8-bit safetensors | ~8 GB | ~8 GB |
| 4-bit safetensors | ~4.5 GB | ~4.5 GB |
| Q4_K_M GGUF | ~4.6 GB | ~4.6 GB |
| Q4_0 GGUF | ~4.3 GB | ~4.3 GB |

Quality is generally indistinguishable between 8-bit and BF16 for inference. For Gemma 4 small-model production lanes, q6 is the normal app default when memory planning says it fits, q8 is the quality/headroom tier, and q4 is reserved for memory-constrained devices, very long retained contexts, or benchmark control runs.

## Quantising During Inference Runs

You can hint the loader to quantise a non-quantised checkpoint at load time:

```go
model, err := mlx.LoadModel("/models/qwen3-8b-bf16/",
    mlx.WithQuantization(4),
)
```

This computes the per-tensor scales on the fly and converts during weight loading. Expect a one-time ~30 s overhead on first load for an 8B model.
