# Attention Probe — Per-Head Post-RoPE K Vectors

`*mlx.Model` (the value `mlx.LoadModel` returns) exposes `InspectAttention`, and the `go-inference`-facing `inference.AttentionInspector` interface exposes the same capability by type assertion. Both give the model's KV cache contents per layer per head after RoPE has been applied — the access path for ad-hoc research (feature analysis, head-importance studies, attention-pattern visualisation) without paying the cost of running unfused attention at every inference step.

> If you need attention *weights* (the softmax(QKᵀ) tensor), see the architectural note in [`docs/architecture.md`](../../docs/architecture.md#attention) — fused SDPA never materialises them, so you'd switch the attention block to an eager path. The probe described here gives you Q/K representations themselves, which is sufficient for many head-level analyses.

`InspectAttention` returns a lightweight `AttentionSnapshot` — Key (and, when available, Query) tensors only, held in memory for the duration of your analysis. This is a different, simpler capture from the full K+V `kv.Snapshot` that [KV snapshots](../model-ops/kv-snapshot.md) save/restore for session resume; reach for that one instead if you need to persist state or need the Value tensors too.

## Live Probe During Generation

```go
package main

import (
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    model, err := mlx.LoadModel("/models/qwen3-8b/")
    if err != nil { log.Fatal(err) }
    defer model.Close()

    reply, err := model.Generate("Once upon a time", mlx.WithMaxTokens(32))
    if err != nil { log.Fatal(err) }
    fmt.Println(reply)

    // Re-run the prompt as a probe and inspect the resulting K tensors.
    snap, err := model.InspectAttention("Once upon a time")
    if err != nil { log.Fatal(err) }

    // Walk every head and compute mean K-vector magnitude per head.
    fmt.Printf("layer | head | mean(||K||)\n")
    for layer := 0; layer < snap.NumLayers; layer++ {
        for head := 0; head < snap.NumHeads; head++ {
            k := snap.Keys[layer][head] // []float32, length = SeqLen * HeadDim
            magnitude := meanL2(k, snap.HeadDim, snap.SeqLen)
            if magnitude > 5.0 {
                fmt.Printf("%5d | %4d | %.3f  ← outlier\n", layer, head, magnitude)
            }
        }
    }
}

func meanL2(values []float32, headDim, seqLen int) float64 {
    if len(values) == 0 || headDim == 0 || seqLen == 0 {
        return 0
    }
    var total float64
    for t := 0; t < seqLen; t++ {
        var sumSq float64
        for d := 0; d < headDim; d++ {
            v := float64(values[t*headDim+d])
            sumSq += v * v
        }
        total += sumSq
    }
    return total / float64(seqLen)
}
```

`InspectAttention` runs its own prefill pass over the prompt, so probing is a separate call from `Generate` — it doesn't reach into an in-flight generation's cache.

## Via The go-inference Interface

Callers already working through the portable interfaces reach the same data by type assertion:

```go
import "dappco.re/go/inference"

inspector, ok := model.(inference.AttentionInspector)
if !ok {
    log.Fatal("model does not expose AttentionInspector")
}
snap, err := inspector.InspectAttention(ctx, "Once upon a time")
```

`inspector.InspectAttention` returns `*inference.AttentionSnapshot` — field-identical to the root `*mlx.AttentionSnapshot` used above (`NumLayers`, `NumHeads`, `SeqLen`, `HeadDim`, `NumQueryHeads`, `Keys`, `Queries`, `Architecture`).

## What Lives In An Attention Snapshot

```go
type AttentionSnapshot struct {
    NumLayers     int
    NumHeads      int      // num_kv_heads (may differ from query heads in GQA)
    SeqLen        int
    HeadDim       int
    NumQueryHeads int
    Keys          [][][]float32 // [layer][head] -> flat float32 of len seq_len*head_dim
    Queries       [][][]float32 // [layer][head] -> flat float32, nil if Q not captured
    Architecture  string
}
```

`Keys` has had RoPE applied — i.e. it's the same K representation the attention kernel actually consumes. For most analyses this is what you want; `HasQueries()` reports whether `Queries` was populated for this capture.

## Per-Layer All-Heads Read

```go
for layer := 0; layer < snap.NumLayers; layer++ {
    for head := 0; head < snap.NumHeads; head++ {
        k := snap.Keys[layer][head]
        saveCSV(fmt.Sprintf("/probes/L%02d-H%02d.csv", layer, head), k, snap.HeadDim, snap.SeqLen)
    }
}
```

## Persisting To Disk For Offline Analysis

`AttentionSnapshot` is in-memory only — it has no Save/Load pair. To persist a K/V capture for offline post-processing in another tool, use the full [KV snapshot](../model-ops/kv-snapshot.md) mechanism instead, which also carries the Value tensors and round-trips through `kv.Load`:

```go
kvSnap, err := model.CaptureKV("Once upon a time")
if err != nil { log.Fatal(err) }
if err := kvSnap.Save("/probes/run-A.kv"); err != nil {
    log.Fatal(err)
}
```

A separate program then loads it via `kv.Load` and walks `kvSnap.Head(layer, head)` — see [`../model-ops/kv-snapshot.md`](../model-ops/kv-snapshot.md).

## Cost

Both `InspectAttention` and `CaptureKV` copy from Metal-resident memory to host memory and run their own prefill pass. For an 8B model with 32 layers × 32 heads × 128 head_dim × seq_len 8192 in Float32, the Key-only capture is ≈4.3 GB (double that, ≈8.6 GB, for the full K+V `kv.Snapshot`) — don't do it every step. Probe at session boundaries or interesting events.

## See Also

- [Attention architecture](../../docs/architecture.md#attention) — why attention weights aren't directly accessible (and how to get them via an eager path)
- [KV snapshot](../model-ops/kv-snapshot.md) — the persistent K+V capture, for session resume rather than ad-hoc probing
- [Perplexity](perplexity.md) — quantitative eval to pair with qualitative head probing
