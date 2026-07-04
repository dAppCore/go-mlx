<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# minimax_m2.go — MiniMax M2-class MoE config

**Package**: `dappco.re/go/mlx`
**File**: `go/minimax_m2.go` (plus `minimax_m2_native_darwin.go` / `_stub.go`)
**Status**: experimental (vMLX parity Phase 1)

## What this is

The **config layer** for MiniMax M2-class Mixture-of-Experts architectures. MiniMax M2 (and 2.7) ship as JANGTQ-quantised MoE models with sparse expert routing — a class of architecture vMLX supports natively but vanilla MLX-LM ran via Python-only paths.

This file owns:

- `MiniMaxM2Config` — the config.json shape parser (routing, attention, MTP flags, tensor mapping)
- Validation that a model pack's tensors match the declared topology
- Detection helper (`IsMiniMaxM2Config`) — used by `model_pack.go` to route during load

The actual MoE forward pass and routing kernels live in `minimax_m2_native_darwin.go` (Metal-side); this file is the platform-agnostic config + planning surface.

## MiniMaxM2Config

```go
type MiniMaxM2Config struct {
    ModelType            string
    Architectures        []string
    VocabSize            int
    HiddenSize           int
    IntermediateSize     int
    NumHiddenLayers      int
    NumAttentionHeads    int
    NumKeyValueHeads     int
    HeadDim              int
    ContextLength        int       // max_position_embeddings
    NumLocalExperts      int       // total experts per layer
    NumExpertsPerToken   int       // top-k experts activated per token
    ScoringFunc          string    // "softmax" | "sigmoid" | …
    UseRoutingBias       bool      // bias-on-router term
    UseMTP               bool      // multi-token-prediction (Gemma-4-assistant style)
    NumMTPModules        int       // drafter module count when UseMTP
    // … RoPE scaling, attention type, expert grouping fields
}
```

The fields mirror the `config.json` MiniMax M2 ships. JSON-tagged so `core.JSONUnmarshalString(raw, &cfg)` works straight against the file.

## Detection

```go
ok := mlx.IsMiniMaxM2Config(cfg)
```

True when `ModelType` ∈ {"minimax_m2", "minimax_m2_7"} or `Architectures` contains a MiniMax-family arch. Used by `model_pack.go`'s arch router.

## Validation

Layer count vs tensor count, expert count vs tensor count, KV-head sanity — pre-load checks that fail fast with descriptive errors instead of late-load Metal crashes.

## Why MiniMax specifically

The 2026-05-09 vMLX gap report identified MiniMax M2/M2.7 as the **highest-value missing model class** — production tools depend on it, vMLX supports it, vanilla MLX-LM forces a Python detour. Native support unblocks CoreAgent for MiniMax-shaped workloads without spawning a Python subprocess.

## Status

Config + validation: present. Native MoE forward: in progress (`minimax_m2_native_darwin.go`). JANGTQ-K weight loading: in progress (paired with `jang_native_darwin.go`). Multi-token prediction modules: planned.

The `capability.go` enum lists `CapabilityMoERouting` and `CapabilityMoELazyExperts` (`experimental` status today; will graduate to `supported` when the forward pass lands).

## Related

- [jang.md](jang.md) — JANGTQ quantisation metadata MiniMax models use
- [expert_residency.md](expert_residency.md) — controls which experts stay resident in VRAM
- [codebook_vq.md](codebook_vq.md) — codebook-quantised tensors (separate but adjacent quant scheme)
- `../../../go-inference/docs/inference/capability.md` — `CapabilityMoERouting` flag
- `docs/vmlx-feature-gap-report.md` — why this is here
- `docs/superpowers/plans/2026-05-09-vmlx-feature-parity.md` — phase plan
