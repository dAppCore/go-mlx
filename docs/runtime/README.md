<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# runtime/ — boot + adapter + API entry

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **load-and-call surface** of the package. How Metal gets registered with go-inference, how a loaded model is wrapped into the runtime, what entry points callers use.

## File map

| File | Doc | Role |
|------|-----|------|
| `register_metal.go` | [register_metal.md](register_metal.md) | Backend registration + metaladapter + Metal allocator controls |
| `production_lane.go` | [2026-05-16-gemma4-e2b-driver-profile.md](2026-05-16-gemma4-e2b-driver-profile.md) | Package-owned Gemma 4 E2B q4 production target and driver-profile shape |
| `local_tuning.go` | [local_autotune.md](local_autotune.md) | Machine/model discovery + opt-in streamed autotune candidates |
| runtime benchmark artefacts | [2026-05-16-gemma4-e2b-driver-profile.md](2026-05-16-gemma4-e2b-driver-profile.md) | Persisted discovery/profile commands, environment, blockers, and next native boundary |
| native greedy rerun | [2026-05-16-gemma4-e2b-native-greedy-rerun.json](2026-05-16-gemma4-e2b-native-greedy-rerun.json) | Post-boundary profile rerun after the compiled greedy decode-tail and session path |
| archived mlx-lm stderr | [2026-05-16-mlx-lm-gemma4-e2b-parity-attempt.txt](2026-05-16-mlx-lm-gemma4-e2b-parity-attempt.txt) | Historical runner stderr for the exact Gemma 4 E2B snapshot; not an active benchmark target |
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
