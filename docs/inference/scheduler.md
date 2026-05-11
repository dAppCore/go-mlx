<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# scheduler.go — request scheduler

**Package**: `dappco.re/go/mlx`
**File**: `go/scheduler.go`
**Implements**: `inference.SchedulerModel`

## What this is

The **queue-aware request scheduler** that turns a single `metal.Model` into a multi-request server. Handles:

- Concurrent request admission up to `MaxConcurrent`
- Queue overflow (reject vs block) at `MaxQueue`
- Cancellation by request id
- Per-request streaming with bounded buffers
- Fair scheduling (FIFO + priority labels)

Implements `inference.SchedulerModel.Schedule(req)` and `inference.CancellableModel.CancelRequest(id)`. Mounted onto `metaladapter` by `register_metal_scheduler.go`.

## SchedulerConfig

```go
type SchedulerConfig struct {
    MaxConcurrent  int      // simultaneous in-flight requests
    MaxQueue       int      // pending queue depth
    StreamBuffer   int      // token channel buffer per request
    PreemptTimeout time.Duration  // how long a request can hold a slot
}
```

`MaxConcurrent` defaults from `MemoryPlan.ParallelSlots`. Bigger isn't always better — KV cache memory scales with concurrent slots.

## Schedule

```go
handle, tokens, err := sched.Schedule(ctx, ScheduledRequest{
    ID:       "req-123",
    Model:    "gemma-4-e2b",
    Messages: messages,
    Sampler:  sampler,
})

for tok := range tokens {
    // each tok carries Request ID + Token + Metrics + Labels
}
```

`tokens` is a buffered channel of `inference.ScheduledToken`. The scheduler closes it on completion (natural EOS, cancel, error).

## Cancellation

```go
sched.CancelRequest(ctx, "req-123")
```

Cancels by request id. The in-flight goroutine notices via shared context.Done, stops decoding mid-stream, releases the slot.

## Fairness

FIFO with optional priority labels. A request with `Labels: {"priority": "high"}` jumps the queue (but doesn't preempt running requests). Used by:

- `core/api` to fast-path interactive chat over batch eval
- `cmd/violet` for "this is a user-typed prompt, ahead of background distillation"

## Why a separate scheduler vs running ad-hoc

Three reasons:

1. **VRAM budget.** Without scheduling, two concurrent prompts double the KV cache footprint mid-flight. The scheduler enforces the `MemoryPlan` budget.
2. **Cancellation.** A pure iter.Seq has no out-of-band cancel; the scheduler wraps with `context.WithCancel` + the cancel API.
3. **Observability.** All requests flow through one chokepoint → emits scheduler stats (queue depth, wait time, throughput) as probe events.

## Probe events

`ProbeEventCachePressure` + `ProbeEventMemoryPressure` per scheduling decision. Lets eval / monitoring track when the scheduler is the bottleneck vs the model.

## Status

Production. Tuning under MoE load pending Phase 1.

## Related

- [block_cache.md](block_cache.md) — KV block sharing across requests in the scheduler
- [decode_optimisation.md](decode_optimisation.md) — speculative + prompt-lookup decode hooks
- [../runtime/register_metal.md](../runtime/register_metal.md) — `register_metal_scheduler.go` mounts this
- `../../../go-inference/docs/inference/contracts.md` — `SchedulerModel` + `CancellableModel` interfaces
- `../../../go-inference/docs/inference/capability.md` — `CapabilityScheduler` + `CapabilityRequestCancel`
