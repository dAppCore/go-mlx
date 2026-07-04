<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# register_metal.go — Metal backend registration + adapter

**Package**: `dappco.re/go/mlx`
**File**: `go/register_metal.go`
**Build tags**: `darwin && arm64 && !nomlx`

## What this is

The **bridge between the inference contract and Apple's Metal GPU**. Three things happen here:

1. `init()` registers a `metalbackend` instance with the `inference.Register` global registry under the name `"metal"`.
2. `metalbackend.LoadModel(path)` returns a `metaladapter` that wraps the internal `metal.Model` (CGO-backed by mlx-c).
3. `metaladapter` implements the full `inference.TextModel` interface — Generate, Chat, Classify, BatchGenerate, ModelType, Info, Metrics, Err, Close, plus optional `AttentionInspector`.

This file is the entry point for the entire native Metal inference stack.

## Auto-registration

```go
func init() { inference.Register(&metalbackend{}) }
```

A consumer writes:

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"   // blank import triggers the init()
)

r := inference.LoadModel(path)
```

— and Metal becomes available without naming it. `inference.Default()` picks Metal first because `preferredBackendOrder` is `metal → rocm → llama_cpp`.

## metalbackend

```go
type metalbackend struct{}

func (b *metalbackend) Name() string                                        { return "metal" }
func (b *metalbackend) Available() bool                                     { return MetalAvailable() }
func (b *metalbackend) LoadModel(path, opts...) (inference.TextModel, error)
```

`Available()` returns false on non-Apple hardware or when MLX library isn't loadable — the build tag prevents this file from compiling on Linux at all, but `Available()` guards against runtime issues like a Metal-less VM.

## LoadModel

Translates `inference.LoadOption` into `metal.LoadConfig` and calls into the internal Metal layer. Key translations:

- `GPULayers != -1` → emits a warning (Metal doesn't do partial offload) and uses full GPU
- `ContextLen == 0` → memory planner picks based on device class
- `ParallelSlots == 0` → memory planner picks based on device class
- `AdapterPath != ""` → loads LoRA on top of base model
- `MemoryPlanInput{Device: memoryPlannerDeviceInfo()}` → resolves to a `MemoryPlan` with batch size, prefill chunk size, prompt cache thresholds, cache/wired/memory limits

The memory planner is what makes loading Just Work across M1 Air (16GB) and M3 Ultra (96GB) — it sizes the context window, cache policy, and KV chunk strategy to what the box actually has.

## metaladapter

Wraps `*metal.Model` and translates between `inference.*` and `metal.*` types. Each method is a near-1:1 transform:

| inference method | metal call | transform |
|------------------|------------|-----------|
| `Generate(ctx, prompt, opts)` | `model.Generate` | wrap iter.Seq, project Token shape |
| `Chat(ctx, msgs, opts)` | `model.Chat` | convert `[]inference.Message` → `[]metal.ChatMessage` |
| `Classify(ctx, prompts, opts)` | `model.Classify` | project `[]metal.ClassifyResult` → `[]inference.ClassifyResult` |
| `BatchGenerate(ctx, prompts, opts)` | `model.BatchGenerate` | project each `BatchResult.Tokens` |
| `Metrics()` | `model.LastMetrics()` | direct projection |
| `ModelType() / Info()` | `model.ModelType / Info` | direct projection |
| `InspectAttention(ctx, prompt)` | `model.InspectAttention` | project `AttentionSnapshot` |

`Err()` and `Close()` pass straight through.

## Memory planner exports

This file also re-exports the package-level Metal allocator controls:

```go
mlx.SetCacheLimit(uint64) uint64           // bytes for Metal cache
mlx.SetMemoryLimit(uint64) uint64          // bytes hard cap
mlx.SetWiredLimit(uint64) uint64           // bytes wired
mlx.GetActiveMemory() uint64               // current usage
mlx.GetPeakMemory() uint64                 // high-water mark
mlx.GetCacheMemory() uint64                // cache occupancy
mlx.ClearCache()                           // release cache between chat turns
mlx.ResetPeakMemory()                      // zero the high-water mark
mlx.GetDeviceInfo() DeviceInfo             // architecture + memory size
```

These are exposed on the parent package because:

1. Callers want to tune limits *before* loading a model.
2. The `inference.RuntimeMemoryLimiter` interface in `go-inference` is the cross-backend surface — `metalbackend` implements it; these getters/setters back that implementation.

## Optional capability surfaces

`metaladapter` implements `inference.AttentionInspector` (always — Apple Metal supports K/Q export).

Other capability interfaces (Scheduler, Cache, CacheService, etc.) are added by **sibling files** that extend `metaladapter` with additional methods:

- `register_metal_cache.go` — wires `inference.CacheService` onto the adapter (block cache stats / warm / clear)
- `register_metal_parser.go` — wires `inference.ToolParser` + `inference.ReasoningParser` via `parser_registry.go`
- `register_metal_scheduler.go` — wires `inference.SchedulerModel` via `scheduler.go`

Each is a small file that adds methods to the existing `metaladapter`, preserving the cohesion of "one type, many opt-in interfaces".

## Stub fallback

`register_metal_stub.go` provides a no-op implementation for non-darwin builds. `MetalAvailable()` returns false there; the backend doesn't register; consumers fall back to whatever else is available (`llama_cpp` typically).

## Related

- [adapter.md](adapter.md) — `InferenceAdapter` — the inverse direction (TextModel → string-buffer API)
- [../inference/scheduler.md](../inference/scheduler.md) — Scheduler implementation
- [../inference/block_cache.md](../inference/block_cache.md) — Block-cache implementation
- [../memory/agent_memory.md](../memory/agent_memory.md) — Wake/Sleep/Fork on top of the adapter
- [../model/memory_plan.md](../model/memory_plan.md) — memory planner that sizes context/cache
- `../../../go-inference/docs/inference/inference.md` — `Backend` + `TextModel` contracts this file implements
