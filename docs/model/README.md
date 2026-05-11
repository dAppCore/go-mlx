<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# model/ — model pack validation, memory planning, GGUF

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **pre-load and metadata layer**. Answers questions about a model before tensors load:

- What is it? (`model_pack.go`)
- How big? (`gguf_info.go`)
- What can my hardware handle? (`memory_plan.go`)
- What algorithms does this pack support? (`algorithm_profile.go`)
- What architecture family is this? (`architecture_profile.go`)
- What weights are present + where? (`safetensor_ref.go`)

Plus the **write-side** for GGUF quantisation (`gguf_quantize.go`) — convert a safetensors pack to GGUF in a chosen quant format.

## File map

| File | Doc | Role |
|------|-----|------|
| `model_pack.go` | [model_pack.md](model_pack.md) | Pack validation + format/arch/quant detection |
| `memory_plan.go` | [memory_plan.md](memory_plan.md) | Device-aware memory planner |
| `gguf_info.go` | (planned) | GGUF metadata reader (backend-specific) |
| `gguf_quantize.go` | (planned) | Quantise safetensors → GGUF |
| `algorithm_profile.go` | (planned) | Per-algorithm runtime status report |
| `architecture_profile.go` | (planned) | Per-architecture support status |
| `safetensor_ref.go` | (planned) | Lazy tensor reference handles |
| `hf_fit.go` | (planned) | HuggingFace Hub source metadata |

## Why a separate "model" doc area

Three distinct concerns share these files:

1. **Pre-load validation** — does the pack exist, is it well-formed, can we load it?
2. **Capability reporting** — what does the pack claim to support? what does the runtime actually support?
3. **Capacity planning** — given this hardware + this pack, what knobs land where?

All three are upstream of the runtime hot path. They run once per pack-load; the hot path takes their output as fixed input.

## Related

- [../runtime/register_metal.md](../runtime/register_metal.md) — calls these at LoadModel time
- [../moe/](../moe/README.md) — MoE arch detection lives there
- `../../../go-inference/docs/inference/discover.md` — package-level discovery
- `../../../go-inference/docs/inference/gguf.md` — package-level GGUF metadata
- `../../../go-inference/docs/inference/capability.md` — capability shape these emit
