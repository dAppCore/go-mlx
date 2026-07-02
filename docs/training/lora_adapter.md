<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# LoRA Adapter Identity And Format

**Package**: `dappco.re/go/mlx`
**Files**: `go/lora/adapter.go`, `go/pkg/metal/lora.go`, `go/backend.go`

## What This Owns

LoRA adapter identity and the on-disk adapter package used by SFT, eval,
`WithAdapterPath`, `Model.LoadLoRA`, and pack fusion.

The live format is a directory or `.safetensors` package with:

- `adapter_config.json` -- adapter metadata such as rank/r, alpha/lora_alpha or
  scale, and target modules/keys/layers.
- one or more `*.safetensors` files -- LoRA A/B tensors only.

The current identity type is `lora.AdapterInfo`, re-exported at the root as
`mlx.LoRAAdapterInfo`:

```go
type AdapterInfo struct {
    Name       string
    Path       string
    Hash       string
    Rank       int
    Alpha      float32
    Scale      float32
    TargetKeys []string
}
```

`lora.InspectAdapter` reads `adapter_config.json`, hashes the config plus sorted
adapter weight files, and returns this identity without loading the base model.
The read-and-hash implementation itself now lives in the shared
`dappco.re/go/inference/lora` package — this package's `InspectAdapter`/`Inspect`
are thin delegates onto it, so go-mlx, go-rocm, and go-cpu share one
adapter-identity shape and one inspection implementation instead of each engine
maintaining a byte-identical copy. The local call signature stays stable for
existing callers (`fuse.go`, `WithAdapterPath`, `Model.LoadLoRA`).
Inspection preserves missing rank/alpha/scale fields so validation paths can
reject incomplete metadata where they must. Native load paths may fill loader
defaults after the adapter is actually attached; root `ModelInfo`, metrics, and
`Adapter()` merge those normalised fields back into the reported identity while
keeping the inspected path and hash stable.
There is no live `BaseModelHash` field in this identity; compatibility is
enforced by target resolution and tensor-shape validation when the adapter is
loaded or fused.

## Weight Names

The loader accepts both native and PEFT-style tensor suffixes:

```text
model.layers.0.self_attn.q_proj.lora_a
model.layers.0.self_attn.q_proj.lora_b
model.layers.0.q_proj.lora_A.weight
model.layers.0.q_proj.lora_B.weight
```

Common wrapper prefixes such as `base_model.model.` are stripped before parsing.
For Gemma 4, suffix targets such as `q_proj` resolve through the shared Gemma-4
target policy to canonical model paths such as `self_attn.q_proj`.

## Save

Training saves through the concrete Metal adapter:

```go
adapter := mlx.NewLoRA(model, &mlx.LoRAConfig{Rank: 8, Alpha: 16})
err := adapter.Save("/path/to/adapter")
```

Saving writes `adapter.safetensors` and `adapter_config.json`. Adapter weights
are only the LoRA A/B matrices, not the frozen base weights.

## Load

Load at model creation:

```go
model, err := mlx.LoadModel("/path/to/model", mlx.WithAdapterPath("/path/to/adapter"))
```

Or load onto an existing model:

```go
adapter, err := model.LoadLoRA("/path/to/adapter")
```

`WithAdapterPath` records adapter identity in `ModelInfo`, metrics, and profile
reports. `Model.LoadLoRA` updates the same root model adapter identity and
refreshes parser hints so generation and chat use the new adapter state.

## Validation

Adapter load fails before attaching anything when:

- `adapter_config.json` is missing or invalid.
- no `.safetensors` files are present.
- a target path is unsupported for the loaded model.
- A/B tensor shapes do not match the resolved base projection.
- the target is a quantized projection that cannot accept live adapter injection.

Pack-level fusion uses the same adapter identity and Gemma-4 target policy, but
it can fuse into quantized safetensors packs by dequantizing only the fused
target and writing that one target back as dense. Fusion requires an explicit
rank in adapter metadata; alpha or scale may be omitted and will use the native
rank-derived default.

## Related

- [sft.md](sft.md) -- training that produces adapters.
- [distill.md](distill.md) -- SSD can produce Gemma-4 LoRA adapters through SFT.
- [grpo.md](grpo.md) -- reasoning training reuses the adapter path.
- `../training.md` -- public training API and fuse API.
