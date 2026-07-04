---
title: Model Operations
description: Merge model packs, quantise to GGUF, snapshot KV state, and plan HuggingFace fits.
---

# Model Operations

The `mlx` package and its operation subpackages own model-pack-level operations
beyond inference and training. Mutating operations write JSON provenance records
so the operation is auditable; inspection operations return serialisable reports
that higher-level research tooling can store beside eval results.

| Operation | Function | Output |
|-----------|----------|--------|
| Merge | `merge.Packs` | New safetensors pack (Linear / SLERP; TIES / DARE reserved, not implemented) |
| Compare | `merge.ComparePacks` | Base/fine-tuned tensor delta report |
| GGUF quantise | `gguf.QuantizeModelPack` | GGUF checkpoint (Q8_0 / Q4_0 / Q4_K_M / …) |
| KV snapshot | `kv.Snapshot.Save` / `kv.Load` | Portable binary KV cache (Float32 or Q8 int8) |
| HF fit | `hf.PlanFits` | Memory-fit plan against HuggingFace Hub metadata |

Merge, GGUF quantise, KV snapshot, and HF fit are library-only today — reached
by importing their subpackage directly (`dappco.re/go/mlx/merge`,
`dappco.re/go/mlx/gguf`, `dappco.re/go/mlx/kv`, `dappco.re/go/mlx/hf`), not
through a root `mlx.*` wrapper. `mlx.ValidateModelPack` is the common way to
get the `ModelPack` value these operations take as input.

## Model Merge

Merge multiple finetuned model packs into a single output pack using a chosen tensor-blending algorithm:

```go
domainA, err := mlx.ValidateModelPack("/models/qwen3-8b-domain-a")
domainB, err := mlx.ValidateModelPack("/models/qwen3-8b-domain-b")

result, err := merge.Packs(ctx, merge.Options{
    Sources: []merge.Source{
        {Pack: domainA, Weight: 0.6},
        {Pack: domainB, Weight: 0.4},
    },
    OutputPath: "/models/qwen3-8b-merged",
    Method:     merge.MethodLinear,
    Labels:     map[string]string{"experiment": "domain-a-and-b"},
})
```

### Methods

| `merge.Method` | Status | Algorithm |
|-----------------|--------|-----------|
| `MethodLinear` | Implemented | Weighted average — simplest, fastest, baseline |
| `MethodSLERP`  | Implemented | Spherical linear interpolation — preserves vector magnitude better; requires exactly two sources |
| `MethodTIES`   | Reserved, not implemented | Trim-Elect-Sign — `Packs` returns an error; the constant is reserved for a future sparse-merge hook |
| `MethodDARE`   | Reserved, not implemented | Drop-And-REscale — same reserved status as TIES |

Architecture, tokenizer, and tensor-shape compatibility are checked by default. Pass `AllowArchitectureMismatch`, `AllowTokenizerMismatch`, or `AllowTensorMismatch` to relax the checks for cross-architecture experiments. The result writes `model.safetensors`, copies metadata files from the first source, and emits a provenance file (`merge.ProvenanceFile`) listing all sources, the method, and per-tensor merge/copy/skip counts.

## Weight Comparison

Compare a base safetensors pack with a fine-tuned pack without loading either
model through Metal:

```go
report, err := merge.ComparePacks(ctx, merge.CompareOptions{
    Base:             basePack,
    FineTuned:        tunedPack,
    IncludeUnchanged: false,
    Labels:           map[string]string{"run": "domain-a-sft"},
})
fmt.Printf("%d changed tensors, mean abs delta %.6f\n",
    report.ChangedTensors, report.MeanAbsDelta)
```

The report carries aggregate counts, missing/extra/shape-mismatch diagnostics,
and per-tensor distance metrics (`mean_abs_delta`, `rms_delta`, `max_abs_delta`,
`l2_delta`, and `cosine`). This keeps the research query path explicit: training
deltas can be inspected from weight files directly instead of guessed from a
single eval score.

## GGUF Quantisation

Convert a safetensors model pack to a GGUF checkpoint without leaving Go. The per-block quantisation maths (`Quantize`/`AppendQuantize`) live in the shared `dappco.re/go/inference/gguf` package; `gguf.QuantizeModelPack` here owns the model-pack orchestration and streaming tensor I/O around it:

```go
source, err := mlx.ValidateModelPack("/models/qwen3-8b")

result, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
    SourcePack: source,
    OutputPath: "/models/qwen3-8b-q4km",
    Format:     gguf.QuantizeQ4_K_M,
    Labels:     map[string]string{"target": "phone-deploy"},
})
fmt.Printf("quantised %d/%d tensors\n", result.QuantizedTensors, result.TensorCount)
```

`OutputPath` is a model-pack directory — `QuantizeModelPack` writes `model.gguf` plus copied metadata into it, not a bare `.gguf` file.

### Formats

| `gguf.QuantizeFormat` | Bits/weight | Notes |
|------------------------|-------------|-------|
| `QuantizeQ8_0`  | 8           | Symmetric int8 with per-block scale, near-lossless |
| `QuantizeQ4_0`  | 4           | Simple 4-bit, good speed, modest quality loss |
| `QuantizeQ4_K_M` | ~4.5        | K-quants medium — best quality/size at 4-bit, recommended default |

`QuantizeQ5_0`, `QuantizeQ4_K`, `QuantizeQ5_K`, `QuantizeQ6_K`, `QuantizeQ8_K`, `QuantizeQ3_K`, and `QuantizeQ2_K` are also available. The result records the requested format, the actually-applied format (which may fall back per-tensor for embedding/output layers), GGUF metadata (`result.Info`), and any notes about tensors that were copied through unquantised.

## KV Snapshot

Snapshot a model's K/V cache plus the last-step logits and token history into a single portable binary file (package `dappco.re/go/mlx/kv`). Useful for resuming long generations across sessions, debugging KV growth, or feeding the same prefix to multiple sampler experiments.

### Capture and Save

`*mlx.Model` (the value `mlx.LoadModel` returns) captures a snapshot directly — no interface assertion needed:

```go
snapshot, err := model.CaptureKV(prompt) // runs one prefill pass

// Default Float32 encoding:
if err := snapshot.Save("/tmp/run.kv"); err != nil { ... }

// Q8 symmetric int8 encoding (smaller file, lossy):
if err := snapshot.SaveWithOptions("/tmp/run.q8.kv", kv.SaveOptions{
    KVEncoding: kv.EncodingQ8,
}); err != nil { ... }
```

`CaptureKVWithOptions(prompt, kv.CaptureOptions{RawKVOnly: true})` skips retaining the float32 side slices when the backend can hand back native-dtype K/V bytes directly. `CaptureKVChunks`/`CaptureKVChunksWithOptions` capture from a streamed sequence of prompt chunks instead of one large string.

### Load

```go
snap, err := kv.Load("/tmp/run.kv")
fmt.Printf("architecture=%s layers=%d heads=%d head_dim=%d seq_len=%d\n",
    snap.Architecture, snap.NumLayers, snap.NumHeads, snap.HeadDim, snap.SeqLen)
fmt.Printf("token offset=%d, %d generated tokens\n", snap.TokenOffset, len(snap.Generated))

if head, ok := snap.Head(/*layer*/12, /*head*/3); ok {
    // head.K and head.V are []float32
}
```

Per-head access via `Head(layer, head)` makes the snapshot directly usable for attention analysis.

### Encoding Options

- `kv.KVSnapshotEncodingFloat32` (default) — bit-exact preservation
- `kv.EncodingQ8` — symmetric int8 + per-tensor scale; ~4× smaller, suitable for archive but not bit-stable round-trip
- `kv.EncodingNative` — captured dtype bytes when the backend provides them, falling back to float32 otherwise

The on-disk format version is `kv.SnapshotVersion = 6` (v6 added per-layer source-cache `MaxSize` so a wake restore carries the slept window/rotation geometry instead of trusting wake-era model templates) with magic header `MLXKV001`.

## HuggingFace Fit Planner

Given device hardware info and a query (or list of model IDs), `hf.PlanFits` (package `dappco.re/go/mlx/hf`) walks HuggingFace Hub metadata and reports which models fit on the target device, with optional context length and LoRA rank planning.

`hf.FitConfig.Device` takes a `dappco.re/go/mlx/memory.DeviceInfo`, a different (narrower) type from the `mlx.GetDeviceInfo()` result — copy the four shared fields across:

```go
gpu := mlx.GetDeviceInfo()
device := memory.DeviceInfo{
    Architecture:                 gpu.Architecture,
    MaxBufferLength:              gpu.MaxBufferLength,
    MaxRecommendedWorkingSetSize: gpu.MaxRecommendedWorkingSetSize,
    MemorySize:                   gpu.MemorySize,
}

src := hf.NewRemoteSource(hf.RemoteConfig{
    Token:     os.Getenv("HF_TOKEN"),
    UserAgent: "go-mlx/research",
})

report, err := hf.PlanFits(ctx, hf.FitConfig{
    Query:       "qwen 3",
    MaxResults:  10,
    Device:      device,
    Source:      src,
    LoRARank:    8,
    KVBytes:     2 << 30, // 2 GB headroom for KV
    ContextHint: 8192,
})
for _, plan := range report.Models {
    fmt.Printf("%s: memory_fits=%v inference_fits=%v weight_bytes=%d\n",
        plan.ModelID, plan.MemoryFits, plan.InferenceFits, plan.WeightBytes)
}
```

`FitReport` carries the device info, classified memory tier (`DeviceClass`), a `MemoryPlan` (cache policy, batch size, and context-length planning — the same `memory.Plan` shape `mlx.PlanMemory` returns), and a per-model `FitPlan` with `MemoryFits`/`InferenceFits` booleans, projected `WeightBytes`/`ExpectedKVBytes`/`ExpectedTotalBytes`, and Hub metadata. `LocalPaths` folds already-downloaded packs into the same report via the same inspection path `mlx.LoadModel` uses. No model files are downloaded — this is purely a planning step. `Source` may be nil only when both `Query` and `ModelIDs` are empty (a `LocalPaths`-only report); setting either with a nil `Source` returns an error rather than silently skipping the network.

## See Also

- [`examples/model-ops/quantize-gguf.md`](../examples/model-ops/quantize-gguf.md)
- [`examples/model-ops/merge.md`](../examples/model-ops/merge.md)
- [`examples/model-ops/kv-snapshot.md`](../examples/model-ops/kv-snapshot.md)
- [`examples/model-ops/hf-fit.md`](../examples/model-ops/hf-fit.md)
- [Training](training.md) — `FuseLoRAIntoModelPack` is the LoRA-side equivalent of these pack-level ops
