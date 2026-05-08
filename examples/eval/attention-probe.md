# Attention Probe — Per-Head Post-RoPE K Vectors

The `metalAdapter` implements `inference.AttentionInspector`, exposing the model's KV cache contents per layer per head after RoPE has been applied. This is the access path for ad-hoc research — feature analysis, head-importance studies, attention-pattern visualisation — without paying the cost of running unfused attention at every inference step.

> If you need attention *weights* (the softmax(QKᵀ) tensor), see the architectural note in [`docs/architecture.md`](../../docs/architecture.md#attention) — fused SDPA never materialises them, so you'd switch the attention block to an eager path. The probe described here gives you Q/K/V representations themselves, which is sufficient for many head-level analyses.

## Live Probe During Generation

```go
package main

import (
    "context"
    "fmt"
    "log"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
)

func main() {
    model, err := inference.LoadModel("/models/qwen3-8b/")
    if err != nil { log.Fatal(err) }
    defer model.Close()

    inspector, ok := model.(inference.AttentionInspector)
    if !ok {
        log.Fatal("model does not expose AttentionInspector")
    }

    ctx := context.Background()
    for tok := range model.Generate(ctx, "Once upon a time", inference.WithMaxTokens(32)) {
        fmt.Print(tok.Text)
    }
    fmt.Println()
    if err := model.Err(); err != nil { log.Fatal(err) }

    // Snapshot KV state after generation finishes.
    snap := inspector.SnapshotKV()

    // Walk every head and compute mean K-vector magnitude per head.
    fmt.Printf("layer | head | mean(||K||)\n")
    for layer := 0; layer < snap.NumLayers; layer++ {
        for head := 0; head < snap.NumHeads; head++ {
            h, ok := snap.Head(layer, head)
            if !ok {
                continue
            }
            // h.K is []float32 of length HeadDim * SeqLen
            magnitude := meanL2(h.K, snap.HeadDim, snap.SeqLen)
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

## What Lives In A Head Snapshot

```go
type KVHeadSnapshot struct {
    K []float32 // post-RoPE keys, length = HeadDim * SeqLen
    V []float32 // values, length = HeadDim * SeqLen
}
```

`K` has had RoPE applied — i.e. it's the same K representation the attention kernel actually consumes. For most analyses this is what you want; the pre-RoPE Q/K representations are a different research question and would need an eager-attention probe (see the architecture note linked above).

## Per-Layer All-Heads Read

```go
for layer := 0; layer < snap.NumLayers; layer++ {
    for head := 0; head < snap.NumHeads; head++ {
        if h, ok := snap.Head(layer, head); ok {
            saveCSV(fmt.Sprintf("/probes/L%02d-H%02d.csv", layer, head), h.K, snap.HeadDim, snap.SeqLen)
        }
    }
}
```

## Persisting To Disk For Offline Analysis

The same data can be saved as a `KVSnapshot` binary for offline post-processing in another tool — see [`../model-ops/kv-snapshot.md`](../model-ops/kv-snapshot.md).

```go
if err := snap.Save("/probes/run-A.kv"); err != nil {
    log.Fatal(err)
}
```

Then a separate program (or notebook) loads the snapshot via `mlx.LoadKVSnapshot` and runs whatever analysis is convenient.

## Cost

`SnapshotKV()` is a copy from Metal-resident memory to host memory. For an 8B model with 32 layers × 32 heads × 128 head_dim × seq_len 8192 in Float32 ≈ 8.6 GB, so don't do it every step. Probe at session boundaries or interesting events.

## See Also

- [Attention architecture](../../docs/architecture.md#attention) — why attention weights aren't directly accessible (and how to get them via an eager path)
- [KV snapshot](../model-ops/kv-snapshot.md) — same data plane, persistent
- [Perplexity](perplexity.md) — quantitative eval to pair with qualitative head probing
