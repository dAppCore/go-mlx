---
title: Model Operations
description: Merge model packs, quantise to GGUF, snapshot KV state, and plan HuggingFace fits.
---

# Model Operations

The root `mlx` package owns four model-pack-level operations beyond inference and training. Each takes a model directory in, produces another directory out, and writes a JSON provenance record so the operation is auditable.

| Operation | Function | Output |
|-----------|----------|--------|
| Merge | `MergeModelPacks` | New safetensors pack (Linear / SLERP / TIES / DARE) |
| GGUF quantise | `QuantizeModelPackToGGUF` | GGUF checkpoint (Q8_0 / Q4_0 / Q4_K_M) |
| KV snapshot | `KVSnapshot.Save` / `LoadKVSnapshot` | Portable binary KV cache (Float32 or Q8 int8) |
| HF fit | `PlanHFModelFits` | Memory-fit plan against HuggingFace Hub metadata |

## Model Merge

Merge multiple finetuned model packs into a single output pack using a chosen tensor-blending algorithm:

```go
result, err := mlx.MergeModelPacks(ctx, mlx.ModelMergeOptions{
    Sources: []mlx.ModelMergeSource{
        {Path: "/models/qwen3-8b-domain-a", Weight: 0.6},
        {Path: "/models/qwen3-8b-domain-b", Weight: 0.4},
    },
    OutputPath: "/models/qwen3-8b-merged",
    Method:     mlx.ModelMergeTIES,
    T:          0.7,
    Labels:     map[string]string{"experiment": "domain-a-and-b"},
})
```

### Methods

| `ModelMergeMethod` | Algorithm |
|--------------------|-----------|
| `ModelMergeLinear` | Weighted average — simplest, fastest, baseline |
| `ModelMergeSLERP`  | Spherical linear interpolation — preserves vector magnitude better |
| `ModelMergeTIES`   | Trim-Elect-Sign — keeps the top fraction of magnitude per tensor, resolves sign conflicts (use `T` ∈ (0,1] as keep-fraction) |
| `ModelMergeDARE`   | Drop-And-REscale — randomly zeros parameters then rescales to preserve expectation |

Architecture, tokenizer, and tensor-shape compatibility are checked by default. Pass `AllowArchitectureMismatch`, `AllowTokenizerMismatch`, or `AllowTensorMismatch` to relax the checks for cross-architecture experiments. The result writes `model.safetensors`, copies metadata files from the first source, and emits `model_merge_provenance.json` listing all sources, the method, and per-tensor merge/copy/skip counts.

## GGUF Quantisation

Convert a safetensors model pack to a GGUF checkpoint without leaving Go:

```go
result, err := mlx.QuantizeModelPackToGGUF(ctx, mlx.QuantizeGGUFOptions{
    ModelPath:  "/models/qwen3-8b",
    OutputPath: "/models/qwen3-8b-q4km.gguf",
    Format:     mlx.GGUFQuantizeQ4_K_M,
    Labels:     map[string]string{"target": "phone-deploy"},
})
fmt.Printf("quantised %d/%d tensors\n", result.QuantizedTensors, result.TensorCount)
```

### Formats

| `GGUFQuantizeFormat` | Bits/weight | Notes |
|---------------------|-------------|-------|
| `GGUFQuantizeQ8_0`  | 8           | Symmetric int8 with per-block scale, near-lossless |
| `GGUFQuantizeQ4_0`  | 4           | Simple 4-bit, good speed, modest quality loss |
| `GGUFQuantizeQ4_K_M` | ~4.5        | K-quants medium — best quality/size at 4-bit, recommended default |

The result records the requested format, the actually-applied format (which may fall back per-tensor for embedding/output layers), GGUF metadata, and any notes about tensors that were copied through unquantised.

## KV Snapshot

Snapshot a model's K/V cache plus the last-step logits and token history into a single portable binary file. Useful for resuming long generations across sessions, debugging KV growth, or feeding the same prefix to multiple sampler experiments.

### Save

After a generation step, get a snapshot from the metalAdapter and save it:

```go
inspector, ok := model.(inference.AttentionInspector) // or KVStateProvider
snapshot := inspector.SnapshotKV()

// Default Float32 encoding:
if err := snapshot.Save("/tmp/run.kv"); err != nil { ... }

// Q8 symmetric int8 encoding (smaller file, lossy):
if err := snapshot.SaveWithOptions("/tmp/run.q8.kv", mlx.KVSnapshotSaveOptions{
    KVEncoding: mlx.KVSnapshotEncodingQ8,
}); err != nil { ... }
```

### Load

```go
snap, err := mlx.LoadKVSnapshot("/tmp/run.kv")
fmt.Printf("architecture=%s layers=%d heads=%d head_dim=%d seq_len=%d\n",
    snap.Architecture, snap.NumLayers, snap.NumHeads, snap.HeadDim, snap.SeqLen)
fmt.Printf("token offset=%d, %d generated tokens\n", snap.TokenOffset, len(snap.Generated))

if head, ok := snap.Head(/*layer*/12, /*head*/3); ok {
    // head.K and head.V are []float32
}
```

Per-head access via `Head(layer, head)` makes the snapshot directly usable for attention analysis (same data plane as the live `AttentionInspector`).

### Encoding Options

- `KVSnapshotEncodingFloat32` (default) — bit-exact preservation
- `KVSnapshotEncodingQ8` — symmetric int8 + per-tensor scale; ~4× smaller, suitable for archive but not bit-stable round-trip

The format version is `KVSnapshotVersion = 3` with magic header `MLXKV001`.

## HuggingFace Fit Planner

Given device hardware info and a query (or list of model IDs), `PlanHFModelFits` walks HuggingFace Hub metadata and reports which models fit on the target device, with optional context length and LoRA rank planning.

```go
src := mlx.NewHuggingFaceModelSource(mlx.HuggingFaceModelSourceConfig{
    Token:     os.Getenv("HF_TOKEN"),
    UserAgent: "go-mlx/research",
})

report, err := mlx.PlanHFModelFits(ctx, mlx.HFModelFitConfig{
    Query:       "qwen 3",
    MaxResults:  10,
    Device:      mlx.GetDeviceInfo(),
    Source:      src,
    LoRARank:    8,
    KVBytes:     2 << 30,    // 2 GB headroom for KV
    ContextHint: 8192,
})
for _, plan := range report.Models {
    fmt.Printf("%s: fits=%v\n", plan.ModelID, plan.Fits)
}
```

The report carries the device info, classified memory tier, a `MemoryPlan` (weights + KV + activations + LoRA breakdown), and a per-model `HFModelFitPlan` with fit status, projected memory, and Hub metadata. No model files are downloaded — this is purely a planning step.

## See Also

- [`examples/model-ops/quantize-gguf.md`](../examples/model-ops/quantize-gguf.md)
- [`examples/model-ops/merge.md`](../examples/model-ops/merge.md)
- [`examples/model-ops/kv-snapshot.md`](../examples/model-ops/kv-snapshot.md)
- [`examples/model-ops/hf-fit.md`](../examples/model-ops/hf-fit.md)
- [Training](training.md) — `FuseLoRAIntoModelPack` is the LoRA-side equivalent of these pack-level ops
