# KV Snapshots — Save & Restore Generation State

A `KVSnapshot` is a portable, self-contained binary capture of a model's K/V cache, the most recent logits, the token offset, and the generated-token history. With a snapshot you can:

- Resume a long generation across processes (interrupt + restart later)
- Try multiple sampler settings against the same KV state without re-prefilling
- Inspect post-RoPE K vectors per layer per head for attention analysis
- Diff KV growth between two model variants

## Saving a Snapshot

The `metalAdapter` exposes attention/KV state via the `inference.AttentionInspector` interface (or a `KVStateProvider`-style snapshot). After running prefill or a few decode steps, capture and save:

```go
package main

import (
    "context"
    "log"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
    mlx "dappco.re/go/mlx"
)

func main() {
    model, err := inference.LoadModel("/models/qwen3-8b/")
    if err != nil { log.Fatal(err) }
    defer model.Close()

    ctx := context.Background()
    for tok := range model.Generate(ctx, "Once upon a time", inference.WithMaxTokens(64)) {
        _ = tok
    }
    if err := model.Err(); err != nil { log.Fatal(err) }

    inspector, ok := model.(inference.AttentionInspector)
    if !ok {
        log.Fatal("model does not expose AttentionInspector")
    }
    snap := inspector.SnapshotKV()

    // Default: lossless Float32 encoding
    if err := snap.Save("/runs/story.kv"); err != nil {
        log.Fatal(err)
    }
}
```

## Lossy Q8 Encoding

For long contexts the snapshot can be large. `KVSnapshotEncodingQ8` stores symmetric int8 + per-tensor scale, ~4× smaller. Generation is approximate after restore but usually indistinguishable in practice for short continuations:

```go
err := snap.SaveWithOptions("/runs/story.q8.kv", mlx.KVSnapshotSaveOptions{
    KVEncoding: mlx.KVSnapshotEncodingQ8,
})
```

## Loading & Inspecting

```go
snap, err := mlx.LoadKVSnapshot("/runs/story.kv")
if err != nil { log.Fatal(err) }

fmt.Printf("architecture: %s\n", snap.Architecture)
fmt.Printf("layers=%d heads=%d head_dim=%d\n", snap.NumLayers, snap.NumHeads, snap.HeadDim)
fmt.Printf("seq_len=%d offset=%d generated=%d\n", snap.SeqLen, snap.TokenOffset, len(snap.Generated))
```

Per-head access for analysis:

```go
for layer := 0; layer < snap.NumLayers; layer++ {
    for head := 0; head < snap.NumHeads; head++ {
        h, ok := snap.Head(layer, head)
        if !ok { continue }
        // h.K and h.V are []float32 of length HeadDim * SeqLen
        analyseAttentionHead(h.K, h.V)
    }
}
```

## Restoring Into a Fresh Model Run

The typical restore path runs through the same model that produced the snapshot. Apply the snapshot's `Tokens` and `Generated` history during model warmup, then continue decode from `TokenOffset`:

```go
restoredModel, err := inference.LoadModel(modelPath)
if err != nil { log.Fatal(err) }

// Re-prefill from the snapshot's tokens; the model rebuilds KV state matching the snapshot.
restoredModel.WarmFromTokens(append(snap.Tokens, snap.Generated...))

// Continue generation:
for tok := range restoredModel.Generate(ctx, "", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
```

Exact-bit KV restore is on the roadmap (`docs/model-state-roadmap.md`) — today's flow re-prefills from the captured token history, which is bit-identical for deterministic samplers.

## Format

| | |
|---|---|
| Magic | `MLXKV001` |
| Version | `KVSnapshotVersion = 3` |
| Encoding | `KVSnapshotEncodingFloat32` (default) or `KVSnapshotEncodingQ8` |
| File | Binary, big-endian length prefixes, `MarshalBinary`/`UnmarshalBinary` round-trip |

The snapshot is fully self-describing — the architecture name, head/layer counts, head dimension, and number of query heads are all stored, so loaders can validate against the model they're being applied to.

## See Also

- [Model operations docs](../../docs/model-operations.md#kv-snapshot) — full reference
- [Attention probe](../eval/attention-probe.md) — same K-vector access via the live `AttentionInspector`
- [Model state roadmap](../../docs/model-state-roadmap.md) — exact-bit restore, state bundles, longer story
