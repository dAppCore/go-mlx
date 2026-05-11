<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# expert_residency.go — MoE expert VRAM management

**Package**: `dappco.re/go/mlx`
**File**: `go/expert_residency.go`
**Status**: experimental (vMLX parity Phase 1)

## What this is

The strategy for **deciding which MoE experts live in VRAM at any moment**. A MiniMax M2-class model can have hundreds of experts per layer; loading them all into VRAM costs more than the device has. Expert residency makes the trade: keep hot experts pinned, swap cold experts in on demand, evict by LRU when VRAM pressure builds.

## Modes

```go
type ExpertResidencyMode string

ExpertResidencyModeOff    = ""        // load everything (small models only)
ExpertResidencyModePinned = "pinned"  // user-named experts always resident
ExpertResidencyModeLazy   = "lazy"    // load on first activation, evict by policy
```

`Off` is the default for non-MoE or small-MoE models. `Pinned` is for known-routing workloads (an instruct-fine-tuned model with a tight expert pattern). `Lazy` is the general production mode.

## Eviction

```go
type ExpertEvictionPolicy string
ExpertEvictionLRU = "lru"
```

LRU is the only policy today. Future: usage-weighted (combine recency with router-score frequency), workload-aware (don't evict experts the next prompt is likely to need).

## Probe events

```go
type ExpertResidencyAction string
// "load" | "evict" | "pin" | "unpin"
```

Each transition emits a probe event so the core/ide MoE panel can render expert residency live during a prompt. Useful for diagnosing slow first-token latency (cold experts → load → spend wall-clock).

## Capacity planning

This file pairs with `memory_plan.go` — the memory planner pre-computes how many experts can be resident given device class + context length + KV cache reservation. The planner publishes an `ExpertCapacity` figure; expert-residency obeys it.

For an M3 Ultra 96GB with a MiniMax M2 model:

- ~30GB for weights (when fully resident)
- ~15GB for KV cache at 32k context
- ~10GB Metal allocator overhead + working sets
- ~40GB for expert residency cache

The planner sizes the resident-set cap so the LRU evictor has headroom before VRAM hits the wall.

## API surface (planned)

```go
runtime.SetExpertResidency(mode ExpertResidencyMode, opts ExpertResidencyOptions) error
runtime.PinExpert(layer int, expertID int) error
runtime.UnpinExpert(layer int, expertID int) error
runtime.ExpertResidencyStats() ExpertResidencyStats
```

`Stats` reports hot-set size, eviction count, average load latency, current LRU depth — fed into the probe bus and the eval pipeline.

## Why this matters for CoreAgent

Without expert residency:

- Large MoE models simply don't fit; the runtime rejects loads
- Workloads that exceed VRAM crash mid-prompt

With expert residency:

- Models 2-3x larger than VRAM still run (cold experts load on demand)
- First-token latency rises (the cost of laziness), but the model loads at all
- Snapshots remain portable across machine classes — a bundle from an M3 Ultra wakes on an M1 Air, just slower

## Status

Mode + policy enums: present. Probe action enum: present. Native load/evict path: in progress (depends on JANGTQ + MoE forward landing first). Eval harness: planned.

## Related

- [minimax_m2.md](minimax_m2.md) — the model class that requires this
- [jang.md](jang.md) — JANGTQ tensor format that experts use
- [codebook_vq.md](codebook_vq.md) — VQ-quantised experts
- `../model/memory_plan.md` (planned) — capacity planning
- `../../../go-inference/docs/inference/capability.md` — `CapabilityMoELazyExperts`
- `../../../go-inference/docs/inference/probe.md` — `ProbeEventRouterDecision` + residency events
