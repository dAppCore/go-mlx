<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# memory_plan.go — device-aware memory planner

**Package**: `dappco.re/go/mlx`
**File**: `go/memory_plan.go`

## What this is

The **"sizes for the box you're running on"** planner. Given a `MemoryClass` (16GB Air through 96GB Ultra), returns a coherent set of runtime knobs:

- Context length
- Parallel slot count
- Batch size
- Prefill chunk size
- Prompt cache thresholds
- Cache / wired / memory limit bytes
- Preferred quantisation
- Quality/fallback quantisation options when the model family has a product
  policy
- Expert capacity (for MoE)

This is what makes `LoadModel(path)` Just Work without the caller specifying every knob. `register_metal.go` calls `PlanMemory()` first; the caller's `WithContextLen(N)` and friends override the plan.

## MemoryClass

```go
MemoryClassUnknown    = "unknown"
MemoryClassApple16GB  = "apple-silicon-16gb"
MemoryClassApple24GB  = "apple-silicon-24gb"
MemoryClassApple32GB  = "apple-silicon-32gb"
MemoryClassApple64GB  = "apple-silicon-64gb"
MemoryClassApple96GB  = "apple-silicon-96gb"
MemoryClassApple128GB = "apple-silicon-128gb"
MemoryClassApple192GB = "apple-silicon-192gb"
MemoryClassApple512GB = "apple-silicon-512gb"   // Mac Pro M-Ultra tiers
```

Detected from `metal.GetDeviceInfo().MemorySize` rounded to the nearest tier.

## MemoryPlan

The planner output:

```go
type MemoryPlan struct {
    ContextLength         int                  // tokens
    ParallelSlots         int                  // concurrent inference slots
    BatchSize             int                  // for batched ops
    PrefillChunkSize      int                  // for chunked prefill
    PromptCache           bool                 // enable prompt cache
    PromptCacheMinTokens  int                  // threshold for caching
    CachePolicy           CachePolicy          // eviction policy
    PreferredQuantization int                  // default quant for this box/model
    QualityQuantization   int                  // opt-in quality tier when it fits
    FallbackQuantization  int                  // constrained-memory tier
    QuantizationPolicy    string               // user-facing policy label
    MemoryLimitBytes      uint64               // Metal allocator hard cap
    CacheLimitBytes       uint64               // Metal allocator cache cap
    WiredLimitBytes       uint64               // Metal wired pages cap
    ExpertCapacity        int                  // resident MoE expert count
    // …
}
```

Per memory class, the planner returns conservative values that leave headroom. Examples:

- **16GB Air**: 4096 ctx / 1 slot / Q4 preferred / 12GB memory cap
- **96GB Ultra**: 32k ctx / 4 slots / Q8 preferred / 80GB cap / 200 experts resident
- **192GB Mac Pro**: 128k ctx / 8 slots / fp16 acceptable / 170GB cap

Gemma 4 small-model plans use a model-family policy rather than the generic
machine-class default: q6 is the normal app default when the memory planner says
it fits, q8 is exposed as the quality/headroom option, and q4 is kept as the
constrained-device fallback.

## MemoryPlanInput

```go
type MemoryPlanInput struct {
    Device          DeviceInfo            // from metal.GetDeviceInfo
    UserContextLen  int                   // override
    UserBatchSize   int                   // override
    Architecture    string                // "minimax_m2" needs different sizing
    ModelBytes      uint64                // measured / estimated
    AdapterBytes    uint64
    // …
}
```

User overrides win; the planner uses them as fixed constraints and adjusts the remaining knobs accordingly. So `WithContextLen(32768)` on a 16GB Air results in *very* tight cache budgets, but it goes through if the model fits at all.

## Why a planner not just per-knob defaults

Three knobs interact. Context-length + parallel-slots + batch-size all consume KV cache memory. Independent defaults would either:

- Set conservative individual values → overall too conservative
- Set generous individual values → OOM at first request

The planner solves them as a single optimisation: max total throughput subject to "stay under the device's safe budget".

## ExpertCapacity for MoE

When `Architecture: "minimax_m2"`, the planner reserves space for resident experts:

```
expert_cap = (MemoryLimitBytes
              - ModelBytes_base
              - KVCacheBytes(ContextLength, ParallelSlots)
              - OverheadBytes) / per_expert_bytes
```

Feeds straight into `expert_residency.go`. A 96GB Ultra running MiniMax M2 7B-active / 56B-total: capacity ~200 experts resident, lazy-loading the rest.

## Status

Apple tier detection: production. Per-architecture sizing: production for dense models, in progress for MoE.

## Used by

- `register_metal.go` LoadModel — pre-load planning
- `cmd/violet` — sidecar prints plan summary at startup
- `core/ide` — surfaces planned values in the model loader UI
- Audit pipeline — sanity-check actual usage vs plan

## Related

- [model_pack.md](model_pack.md) — pack-side metadata feeds into the planner
- [../runtime/register_metal.md](../runtime/register_metal.md) — the LoadModel caller
- [../moe/expert_residency.md](../moe/expert_residency.md) — consumes ExpertCapacity
- `../../../go-inference/docs/inference/capability.md` — `CapabilityMemoryPlanning`
- `project_local_inference_topology.md` — measured numbers per device class
