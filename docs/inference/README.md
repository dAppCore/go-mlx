<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# inference/ — request scheduling, cache, decode, parsers

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **runtime hot path** beyond raw forward pass — everything that turns "I can run a forward pass" into "I can serve many concurrent requests efficiently with shared prefix cache, optional speculative decode, and model-family-specific output parsing".

These are the capability-interface implementations that `register_metal_*.go` files mount onto the metal adapter.

## File map

| File | Doc | Implements (inference contract) |
|------|-----|--------------------------------|
| `scheduler.go` | [scheduler.md](scheduler.md) | `SchedulerModel` + `CancellableModel` |
| `block_cache.go` | [block_cache.md](block_cache.md) | `CacheService` |
| `decode_optimisation.go` | [decode_optimisation.md](decode_optimisation.md) | speculative + prompt-lookup hooks |
| `parser_registry.go` | [parser_registry.md](parser_registry.md) | `ReasoningParser` + `ToolParser` routing |
| `thinking.go` | [thinking.md](thinking.md) | thinking-channel policy |

## How they mount onto the adapter

`register_metal.go` builds the base `metaladapter` implementing `inference.TextModel`. Three sibling files add capability interfaces:

```go
// register_metal_scheduler.go
func (a *metaladapter) Schedule(ctx, req) (...) { return a.scheduler.Schedule(...) }

// register_metal_cache.go
func (a *metaladapter) CacheStats(ctx) (...) { return a.blockCache.CacheStats(...) }

// register_metal_parser.go
func (a *metaladapter) ParseReasoning(...) { return a.reasoningParser.ParseReasoning(...) }
```

A consumer probes via type assertion:

```go
if sched, ok := model.(inference.SchedulerModel); ok { ... }
if cache, ok := model.(inference.CacheService);    ok { ... }
if parser, ok := model.(inference.ReasoningParser); ok { ... }
```

## Why each in its own file

Each capability is independently optional. A backend can implement Scheduler without Cache, Cache without Parsers, etc. Co-locating them would be smaller but bigger files; separating them lets each evolve at its own pace.

## Related

- [../runtime/register_metal.md](../runtime/register_metal.md) — base adapter + how these mount
- `../../../go-inference/docs/inference/contracts.md` — the contracts each implements
- `../../../go-inference/docs/inference/capability.md` — capability flags
- `../../../go-inference/docs/openai/services.md` — HTTP handlers that consume the cache + cancel surfaces
- [../memory/agent_memory.md](../memory/agent_memory.md) — Wake/Sleep coordinates with the scheduler for in-flight session preservation
