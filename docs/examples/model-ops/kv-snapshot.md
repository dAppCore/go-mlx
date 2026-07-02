# KV Snapshots — Save & Restore Generation State

A `kv.Snapshot` (package `dappco.re/go/mlx/kv`) is a portable, self-contained binary capture of a model's K/V cache, the most recent logits, the token offset, and the generated-token history. With a snapshot you can:

- Resume a long generation across processes (interrupt + restart later)
- Try multiple sampler settings against the same KV state without re-prefilling
- Inspect post-RoPE K vectors per layer per head for attention analysis
- Diff KV growth between two model variants

## Capturing And Saving

`*mlx.Model` (the value `mlx.LoadModel` returns) captures a snapshot directly — no interface assertion needed:

```go
package main

import (
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    model, err := mlx.LoadModel("/models/qwen3-8b/")
    if err != nil { log.Fatal(err) }
    defer model.Close()

    snap, err := model.CaptureKV("Once upon a time")
    if err != nil { log.Fatal(err) }

    // Default: lossless Float32 encoding
    if err := snap.Save("/runs/story.kv"); err != nil {
        log.Fatal(err)
    }
}
```

`CaptureKV` runs a single prefill pass over the prompt and returns the resulting cache. `CaptureKVChunks` does the same from a streamed sequence of prompt chunks instead of one large string.

## Lossy Q8 Encoding

For long contexts the snapshot can be large. `kv.EncodingQ8` stores symmetric int8 + per-tensor scale, ~4× smaller. Generation is approximate after restore but usually indistinguishable in practice for short continuations:

```go
import "dappco.re/go/mlx/kv"

err := snap.SaveWithOptions("/runs/story.q8.kv", kv.SaveOptions{
    KVEncoding: kv.EncodingQ8,
})
```

## Loading & Inspecting

```go
snap, err := kv.Load("/runs/story.kv")
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
        // h.Key and h.Value are []float32 of length HeadDim * SeqLen
        analyseAttentionHead(h.Key, h.Value)
    }
}
```

## Restoring Into A Fresh Model Run

`WarmPromptCacheFromKV` installs a captured K/V prefix directly as a model's prompt cache — an exact-bit restore, not a re-prefill:

```go
restoredModel, err := mlx.LoadModel(modelPath)
if err != nil { log.Fatal(err) }

if err := restoredModel.WarmPromptCacheFromKV(snap); err != nil {
    log.Fatal(err)
}

// Continue generation from the restored prefix:
reply, err := restoredModel.Generate("", mlx.WithMaxTokens(128))
if err != nil { log.Fatal(err) }
fmt.Println(reply)

// Or stream it instead:
for tok := range restoredModel.GenerateStream(ctx, "", mlx.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
```

The durable session Wake/Sleep path (`Session.WakeAgentMemory` / `SleepAgentMemory` — see [`docs/memory/agent_memory.md`](../../docs/memory/agent_memory.md)) builds on the same KV-block machinery for cross-process, named conversation state; reach for `WarmPromptCacheFromKV` directly when you already hold a `*kv.Snapshot` in-process.

## Format

| | |
|---|---|
| Magic | `MLXKV001` |
| Version | `kv.SnapshotVersion = 6` |
| Encoding | `kv.KVSnapshotEncodingFloat32` (default), `kv.EncodingQ8`, or `kv.EncodingNative` |
| File | Binary, big-endian length prefixes, `MarshalBinary`/`UnmarshalBinary` round-trip |

The snapshot is fully self-describing — the architecture name, head/layer counts, head dimension, and number of query heads are all stored, so loaders can validate against the model they're being applied to.

## See Also

- [Model operations docs](../../docs/model-operations.md#kv-snapshot) — full reference
- [Attention probe](../eval/attention-probe.md) — the lighter, in-memory-only K-vector capture (`InspectAttention`) for ad-hoc analysis instead of session persistence
- [Agent memory](../../docs/memory/agent_memory.md) — the durable, named-slot Wake/Sleep path built on the same KV-block format
