<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# runtime/ — boot + adapter + API entry

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **load-and-call surface** of the package. How Metal gets registered with go-inference, how a loaded model is wrapped into the runtime, what entry points callers use.

## File map

| File | Doc | Role |
|------|-----|------|
| `register_metal.go` | [register_metal.md](register_metal.md) | Backend registration + metaladapter + Metal allocator controls |
| `production_lane.go` | `GOAL.md` / `TODO.md` | Package-owned Gemma 4 production target and driver-profile shape |
| official Gemma 4 E2B source locks | [2026-05-31-official-gemma4-e2b-source-lock.json](2026-05-31-official-gemma4-e2b-source-lock.json) | Target, MTP assistant, and q8/q6/q4 target packs |
| official Gemma 4 12B Unified source lock | [2026-06-04-official-gemma4-12b-unified-source-lock.json](2026-06-04-official-gemma4-12b-unified-source-lock.json) | Goal 4 unified text/vision/audio config lock plus command-ready driver-profile bench shape |
| Gemma 4 12B 6-bit performance manifest | [2026-06-04-gemma4-12b-6bit-performance.json](2026-06-04-gemma4-12b-6bit-performance.json) | Downloaded MLX 12B 6-bit pack, baseline bench, promoted fast-lane gates, and zero-copy streaming follow-up |
| Gemma 4 6-bit chapter-profile baselines | [2026-06-05-gemma4-6bit-chapter-profile.md](2026-06-05-gemma4-6bit-chapter-profile.md) | Real book/chapter bench baselines for E2B, E4B, and 12B Unified plus failed 31B/MoE quality probes and the post-cleanup uncapped-request load failure |
| AutoRound profile manifest | [2026-06-04-auto-round-profiles.json](2026-06-04-auto-round-profiles.json) | Native no-Python AutoRound, AutoRound Best, AutoRound Light profile defaults, pack sidecar recognition, calibration plan, and RTN/SignRound primitive status |
| Simple Self-Distillation recipe manifest | [2026-06-04-simple-self-distillation-recipes.json](2026-06-04-simple-self-distillation-recipes.json) | Native no-Python data-generation and LiveCodeBench-v6 eval defaults for the three SimpleSD recipes |
| hierarchical memory-pretraining artifact manifest | [2026-06-04-memory-pretraining-artifacts.json](2026-06-04-memory-pretraining-artifacts.json) | Native no-Python router, FFN memory-bank, and JSONL cluster-ID artifact defaults for Goal 3 |
| official Gemma 4 E2B preflight | [2026-05-31-official-gemma4-e2b-local-preflight.md](2026-05-31-official-gemma4-e2b-local-preflight.md) | Local locked-source, MTP assistant, and q4 control compatibility proof |
| official Gemma 4 E2B target state smoke | [2026-06-01-official-gemma4-e2b-target-native-state-smoke.md](2026-06-01-official-gemma4-e2b-target-native-state-smoke.md) | Native target generation plus prompt-cache, K/V restore, state bundle, and State K/V block warm smoke |
| official Gemma 4 E2B MTP draft-2 diagnostic | [2026-06-01-official-gemma4-e2b-mtp-draft2-diagnostic.md](2026-06-01-official-gemma4-e2b-mtp-draft2-diagnostic.md) | go-mlx target-only versus official assistant draft-2 diagnostic; rejected for production promotion |
| `local_tuning.go` | [local_autotune.md](local_autotune.md) | Machine/model discovery + opt-in streamed autotune candidates |
| `turboquant` cache mode | [turboquant_kv.md](turboquant_kv.md) | Explicit research lane for compressed KV State pages; fail-closed until the versioned physical layout exists |
| runtime benchmark artefacts | `GOAL.md` / `/private/tmp/go-mlx-goal/reports` | Current measurements are summarised in the goal doc; fresh accepted artefacts should be regenerated after code stabilises |
| `register_metal_cache.go` | (planned) | Mount `CacheService` onto metaladapter |
| `register_metal_parser.go` | (planned) | Mount `ReasoningParser` + `ToolParser` onto metaladapter |
| `register_metal_scheduler.go` | (planned) | Mount `SchedulerModel` + `CancellableModel` |
| `register_metal_stub.go` | (planned) | No-op fallback for non-darwin |
| `adapter.go` | [adapter.md](adapter.md) | `InferenceAdapter` — buffered/string client API |
| `api_common.go` / `api_darwin.go` / `api_stub.go` | (planned) | Public root API (`LoadModel`, `WithContextLength`, …) |
| `api_shape_common.go` | (planned) | Shared API shapes |
| `api_tokenizer_*.go` | (planned) | Tokenizer subsurface |
| `backend_common.go` | (planned) | Shared backend helpers |
| `mlx.go` / `mlx_stub.go` | (planned) | Package init + version |
| `options_darwin.go` | (planned) | Darwin-specific load options |

## Two adapter directions

A confusing-but-deliberate naming pattern:

- **`metaladapter`** (in `register_metal.go`) wraps `*metal.Model` to implement `inference.TextModel`. **Server-side.**
- **`InferenceAdapter`** (in `adapter.go`) wraps `inference.TextModel` to expose buffered string API. **Client-side.**

They are not the same type, despite the name overlap. See [adapter.md](adapter.md) for the disambiguation.

## Boot flow

```
package init time:
  register_metal.go init() → inference.Register(&metalbackend{})

caller imports:
  import _ "dappco.re/go/mlx"

caller calls:
  inference.LoadModel("/models/gemma-4-e2b")
   → inference.Default() returns metalbackend
   → metalbackend.LoadModel(path)
     → memory_plan.PlanMemory() — sizes for this device
     → metal.LoadAndInit(path, planCfg) — CGO call into mlx-c
     → returns &metaladapter{model, scheduler, cache, parsers}
   → returns metaladapter (implements TextModel)

caller uses:
  for tok := range model.Generate(ctx, prompt) { … }
```

## Related

- `../../../go-inference/docs/inference/inference.md` — Backend + TextModel contract this implements
- [../model/memory_plan.md](../model/memory_plan.md) — sizing input to LoadModel
- [../model/model_pack.md](../model/model_pack.md) — pre-load validation
- [local_autotune.md](local_autotune.md) — UI-facing discovery and optional tuning flow
- [../inference/README.md](../inference/README.md) — capability interfaces mounted onto metaladapter
- [../memory/agent_memory.md](../memory/agent_memory.md) — Wake/Sleep on top of metaladapter
- [../cmd/violet.md](../cmd/violet.md) — sidecar daemon that boots this
