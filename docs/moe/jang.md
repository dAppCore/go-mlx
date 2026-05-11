<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# jang.go — JANG / JANGTQ quantisation metadata

**Package**: `dappco.re/go/mlx`
**File**: `go/jang.go` (plus `jang_native_darwin.go` / `_stub.go`, `jang_darwin_test.go`)
**Status**: experimental (vMLX parity Phase 1)

## What this is

The metadata-layer support for JANG and JANGTQ — the quantisation schemes MiniMax M2 (and several Qwen variants) use. Owns:

- `JANGQuantizationInfo` — the `jang_config.json` sidecar parser
- `JANGCapabilities` — runtime-facing affordances declared by the pack (which tool parser, which reasoning parser)
- `JANGPackedQuantizationProfile` — packed-format shape (group size, bit budgets per tensor class, codebook flags)
- Detection / validation

JANG is interesting because it's **per-tensor-class quantisation** — attention weights, shared experts, routed experts, embeddings, and LM head each get their own bit budget. JANGTQ adds packed tensor formats with group-shared scales.

## JANGQuantizationInfo

```go
type JANGQuantizationInfo struct {
    Version            int
    WeightFormat       string    // "jang" | "jangtq" | "jangtq_k"
    Profile            string    // "JANG_2M" | "JANG_3M" | "JANG_4M" | "JANG_6M" | …
    Method             string    // "symmetric" | "asymmetric"
    GroupSize          int       // 64 | 128 typical

    BitsDefault        int       // fallback when not overridden
    AttentionBits      int       // override for attention projections
    SharedExpertBits   int       // override for the shared FFN expert
    RoutedExpertBits   int       // override for routed experts
    EmbedTokensBits    int       // override for token embeddings
    LMHeadBits         int       // override for LM head

    SourceName         string    // upstream model id
    SourceOrg          string
    SourceArchitecture string

    Capabilities       JANGCapabilities
    Packed             *JANGPackedQuantizationProfile
}
```

Why per-class bits: attention is more sensitive than expert FFN; LM head needs higher precision than mid-layers; embeddings can usually go to 4-bit cheap. A single global bit-width either over-spends on tolerant tensors or under-spends on sensitive ones.

## JANGCapabilities

```go
type JANGCapabilities struct {
    ReasoningParser  string  // "qwen-think" | "gemma-think" | "deepseek-r1" | …
    ToolParser       string  // "qwen-tools" | "minimax-tools" | …
    ChatTemplate     string  // template hash or name
    // …
}
```

The pack declares which model-family-specific parsers it wants. The runtime uses these strings to pick handlers from `parser_registry.go`.

## JANGPackedQuantizationProfile

The packed-format extension. Describes:

- How tensor rows are packed into uint8 / uint16 streams
- Group-shared scale storage layout
- Whether codebook indices accompany packed weights

Detection is metadata-first — the runtime knows whether a `*.safetensors` shard carries packed JANGTQ tensors before opening any of the binary blobs.

## Detection

```go
ok := mlx.IsJANGModelPack(packDir)
info, err := mlx.LoadJANGQuantizationInfo(packDir)
```

`IsJANGModelPack` is the fast existence check (`jang_config.json` present + parses). `LoadJANGQuantizationInfo` parses + validates + returns the full descriptor.

## Profile names

```
JANG_2M — 2-bit mid-tier
JANG_3M — 3-bit mid-tier
JANG_4M — 4-bit (most common)
JANG_6M — 6-bit (highest quality JANG)
JANG_2L / JANG_3L / JANG_4L / JANG_6L — same bit budgets, looser groups (denoted L)
```

The 'M' / 'L' suffix maps to group size — M is the medium granularity (typically 128), L is the loose granularity (typically 256). Smaller groups → higher quality, more scale storage overhead.

## Status

Metadata recognition: done. Native packed tensor load: in progress (`jang_native_darwin.go`). MoE forward against JANGTQ weights: paired with MiniMax M2 forward work.

When complete, this gives go-mlx native loading of:

- MiniMax M2 / 2.7 (JANGTQ_K)
- JANG-quantised Qwen variants
- Future packs declaring `weight_format: "jang"` in their sidecar

## Related

- [minimax_m2.md](minimax_m2.md) — the model family that drove this work
- [codebook_vq.md](codebook_vq.md) — adjacent quant scheme (VQ codebooks)
- [expert_residency.md](expert_residency.md) — MoE expert VRAM management
- `../model/model_pack.md` (planned) — `IsJANGModelPack` is one branch in pack detection
- `../../../go-inference/docs/inference/capability.md` — `CapabilityJANGTQ` flag
- `docs/vmlx-feature-gap-report.md` — why this is here
