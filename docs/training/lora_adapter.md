<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# lora_adapter.go — LoRA adapter identity + on-disk format

**Package**: `dappco.re/go/mlx`
**File**: `go/lora_adapter.go`

## What this is

The **identity + serialisation** for LoRA adapters. Holds:

- `LoRAAdapterInfo` — reproducible identity (name, path, hash, rank, alpha, target keys, base-model hash)
- Save / load helpers for adapter `.npz` files
- Validation that a loaded adapter is compatible with the current base model

The actual training is in `sft.go` / `grpo.go` / `distill.go`; the actual fusion is in `lora_fuse.go`. This file is what those operations produce / consume.

## LoRAAdapterInfo

```go
type LoRAAdapterInfo struct {
    Name       string    // human-readable
    Path       string    // file path or URI
    Hash       string    // sha256 of adapter file (identity)
    Rank       int       // decomposition rank (LoRAConfig.Rank)
    Alpha      float32   // scaling factor
    TargetKeys []string  // which projections were adapted ("q_proj", "v_proj", …)

    BaseModelHash string   // identity of the base model this adapter was trained against
    Format        string   // file format (npz / safetensors)
    Labels        map[string]string  // metadata for filtering
}
```

`BaseModelHash` is the compatibility check. A LoRA trained on Gemma-3-1B won't load onto Gemma-4-E2B; the hash mismatch is caught here, not at the first matmul.

## On-disk format

Adapters serialise as MLX `.npz` files containing per-layer pairs:

```
model.layers.0.self_attn.q_proj.lora_A   shape [rank, in_dim]
model.layers.0.self_attn.q_proj.lora_B   shape [out_dim, rank]
model.layers.0.self_attn.v_proj.lora_A   …
model.layers.0.self_attn.v_proj.lora_B   …
…
```

Plus a `adapter_config.json` sidecar carrying the `LoRAAdapterInfo` shape.

`Rank × (in_dim + out_dim)` parameters per adapted projection. For a 7B model with Rank=8 and TargetKeys=[q_proj, v_proj], that's ~50MB of adapter weights — vs ~14GB for the base. The size win is what makes "ship adapters not models" viable.

## Save

```go
info, err := mlx.SaveLoRAAdapter(adapter, path, baseModelHash)
```

Writes the `.npz` + sidecar, computes the hash, returns the populated `LoRAAdapterInfo`.

## Load

```go
adapter, info, err := mlx.LoadLoRAAdapter(path, baseModel)
```

Reads the `.npz` + sidecar, validates `BaseModelHash` matches the loaded base model's hash, materialises the adapter onto the metal model. Returns both the adapter handle and its info for record-keeping.

## Why hash-based identity

Three reasons:

1. **Verifiable provenance.** An adapter on a USB stick is identifiable without trusting the filename.
2. **Bundle compatibility check.** Wake refuses if `bundle.AdapterIdentity.Hash` ≠ live adapter's hash — see [`agent_memory.md`](../memory/agent_memory.md).
3. **Cache key.** When `core/api` serves multiple base+adapter combinations, the cache key includes the adapter hash.

## Adapter chains (planned)

Future: stacking multiple LoRAs (one for persona, one for tool-use, one for safety). Today the runtime supports one adapter at a time. `LoRAAdapterInfo.Labels` carries hints for future chain composition.

## Related

- [sft.md](sft.md) — training that produces adapters
- [grpo.md](grpo.md) — reasoning training that produces adapters
- [distill.md](distill.md) — distillation that produces adapters
- [lora_fuse.md](lora_fuse.md) — fuse adapter into base weights
- `../../../go-inference/docs/state/identity.md` — `AdapterIdentity` portable shape
- `../../../go-inference/docs/inference/training.md` — `LoRAConfig` contract
