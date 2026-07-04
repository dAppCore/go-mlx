<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# medium.go — model loading from io.Medium

**Package**: `dappco.re/go/mlx`
**File**: `go/medium.go`

## What this is

The integration point with `dappco.re/go/io`'s **Medium** abstraction — the universal transport that lets the same model load from local disk, S3, State video, in-memory blob, or any future backend without code changes at the call site.

## Public surface

```go
mlx.LoadModelFromMedium(medium coreio.Medium, modelPath, opts...) (*Model, error)
mlx.WithMedium(medium coreio.Medium) LoadOption
```

`WithMedium` is the option-style integration:

```go
medium, _ := coreio.OpenS3("s3://lethean-models/gemma4-e2b/")
model, err := mlx.LoadModel("gemma-4-e2b", mlx.WithMedium(medium), mlx.WithContextLength(8192))
```

`LoadModelFromMedium` is the convenience wrapper:

```go
model, err := mlx.LoadModelFromMedium(medium, "models/gemma-3-1b", mlx.WithContextLength(8192))
```

— equivalent to `LoadModel(modelPath, append(opts, WithMedium(medium))...)`.

## What's staged through the medium

- `config.json` — model architecture
- `tokenizer.json` / `tokenizer.model` — tokeniser
- `*.safetensors` — weights (multiple shards)
- `chat_template.jinja` (optional) — chat template
- `adapter_config.json` + adapter safetensors (when `WithAdapterPath` set)

Each file is fetched lazily via the Medium's `OpenFile(path)`. The loader doesn't materialise the entire model archive on disk before starting — for large models on slow mediums, weight files start downloading while the loader is parsing config.

## Why Medium not stdlib io

Two reasons:

1. **One abstraction across backends.** Local disk, S3, State video, in-memory, future Lethean-distributed all satisfy `coreio.Medium`. The model loader doesn't branch on storage type.
2. **Hot-swap.** A running session can switch its model source from one Medium to another (e.g., local → S3 fallback on disk-pressure) without restart. The Medium API is stateless enough to allow this.

The full design is in [`design_medium_universal_transport.md`](../../../core/.claude/memory/design_medium_universal_transport.md).

## Implementation note

Loading is **read-only**. The model loader doesn't write through the Medium. Bundle writes go through a different path — the `state.Store` interfaces (see [`store.md`](../../../go-inference/docs/state/store.md)). The two abstractions deliberately don't overlap: model loading reads structured files; bundle storage reads/writes opaque chunks.

## Related

- `dappco.re/go/io` — Medium contract + implementations
- [register_metal.md](../runtime/register_metal.md) — LoadModel that this hooks into
- [model_pack.md](../model/model_pack.md) — model-pack validation before load
- `design_medium_universal_transport.md` — design memory
