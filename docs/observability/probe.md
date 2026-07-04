<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# probe.go — runtime telemetry emitter

**Package**: `dappco.re/go/mlx`
**File**: `go/probe.go`

## What this is

The **go-mlx side** of the probe bus. Implements emit hooks for the event kinds defined in `go-inference/probe.go`, plus go-mlx-specific event detail (Metal allocator state, expert routing per layer, cache pressure per-block).

`metaladapter.ProbeSink` is set by the consumer (via load option or scheduler attach); emit calls fan out to it. No-op when no sink attached.

## Event kinds emitted

From the inference probe set:

- `ProbeEventToken` — every generated token (id, text, sample temperature)
- `ProbeEventLogits` — raw logits (when `WithLogits()` set)
- `ProbeEventEntropy` — per-step sampling entropy
- `ProbeEventSelectedHeads` — attention head selection per layer
- `ProbeEventLayerCoherence` — per-layer activation alignment
- `ProbeEventRouterDecision` — MoE expert routing per token
- `ProbeEventResidual` — residual-stream magnitude per layer
- `ProbeEventCachePressure` — block cache fill / eviction
- `ProbeEventMemoryPressure` — Metal allocator state
- `ProbeEventTraining` — SFT / GRPO / Distill step events

## Emission points

```
Generate / Chat:
  prefill start                → cache_pressure (initial)
  per layer                    → layer_coherence + selected_heads
  per token                    → token + entropy
  router (MoE only)            → router_decision
  forward done                 → memory_pressure

Training:
  per step                     → training (loss, lr, grad-norm)
  per epoch                    → training (epoch boundary marker)

Memory:
  wake start / per block / done → cache_pressure (decode side)
  sleep start / per block / done → cache_pressure (encode side)
```

## Payload shape

Each event carries a small fixed payload + free-form labels. The runtime emits structured fields (per-layer floats, expert indices, byte counts); the sink decides what to do with them — log, accumulate into eval report, stream to SSE, drop.

## Subscribers

| Subscriber | Use |
|------------|-----|
| `core/api` SSE handler | live UI in core/ide reasoning + memory panels |
| `eval.go` | accumulate per-sample probes into eval reports |
| `go-ml/agent_eval.go` | scoring engine consumes router/coherence events |
| audit / dev log | dump JSON for offline analysis |

A consumer attaches a sink via `WithProbeSink(...)` option on `LoadModel`, or per-request via the scheduler.

## Why all these events

Each one answers a real question:

- **Token / entropy** → "is the model confident or hedging here?"
- **Selected heads** → "which heads carry meaning for this prompt?" (attention probe)
- **Layer coherence** → "is layer N adding signal or noise?" (used in pruning research)
- **Router decision** → "which experts fire? are some always-cold?" (MoE health)
- **Residual** → "is the residual stream stable or blowing up?" (training diagnostic)
- **Cache pressure** → "are we hitting the prompt cache?" (perf)
- **Memory pressure** → "are we close to allocator limit?" (capacity planning)
- **Training** → "loss curve, grad norm, lr — is this run healthy?"

Together these are the cognitive shape of inference + training, captured at runtime.

## Performance

Probe emission is allocation-light — events use stack-allocated structs where possible, copy maps only on emit-with-labels. A typical 1024-token generation emits ~5000 events; the sink's overhead dominates the cost, not the emission.

When no sink is attached, emit is a single nil check.

## Related

- `../../../go-inference/docs/inference/probe.md` — base contract this implements
- [../training/eval.md](../training/eval.md) — eval consumes probe events
- [../inference/scheduler.md](../inference/scheduler.md) — per-request probe sinks
- `../../../go-inference/docs/inference/capability.md` — `CapabilityProbeEvents` + `CapabilityAttentionProbe` + `CapabilityLogitProbe` flags
