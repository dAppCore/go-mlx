<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# moe/ — Mixture-of-Experts + advanced quant

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

The **vMLX parity Phase 1** work — native loading and dispatch for MoE-architecture models with packed JANGTQ / codebook-VQ quantisation. Pre-dates this sprint were dense models (Gemma 3/4 dense, Qwen 3, Llama 3); this area unlocks the sparse-expert class (MiniMax M2/2.7, JANG-quantised Qwen variants).

Status as of 2026-05-09: metadata + planning surface done; native MoE forward + JANGTQ load in progress; expert residency hooks present awaiting forward.

## File map

| File | Doc | Role |
|------|-----|------|
| `minimax_m2.go` | [minimax_m2.md](minimax_m2.md) | MiniMax M2-class config + detection |
| `jang.go` | [jang.md](jang.md) | JANG / JANGTQ quantisation metadata |
| `codebook_vq.go` | [codebook_vq.md](codebook_vq.md) | Vector-quantised tensor metadata |
| `expert_residency.go` | [expert_residency.md](expert_residency.md) | MoE expert VRAM management |
| `minimax_m2_native_darwin.go` | (planned) | Metal-side MoE forward pass |
| `jang_native_darwin.go` | (planned) | Metal-side JANGTQ dequant + load |
| `internal/metal/minimax_m2.go` | (planned) | CGO MoE kernels |
| `internal/metal/codebook_vq.go` | (planned) | CGO VQ dequant kernels |
| `internal/metal/jang_dequant.go` | (planned) | CGO JANG dequant kernels |

## Phase 1 goals (vMLX parity plan)

1. **MiniMax M2 + 2.7 native** — eliminate the Python detour. Tracked, in flight.
2. **JANGTQ_K weight load** — the quant scheme M2 ships with. Tracked, in flight.
3. **Expert residency** — pinned + lazy modes with LRU eviction. Metadata + hooks done.
4. **Probe coverage** — expert-load/evict events, router-decision events. Hooks present.

The combination unlocks "load M2 7B-active / 56B-total on a 96GB M3 Ultra without falling back to Python or paging to disk constantly".

## Related contracts

- `../../../go-inference/docs/inference/capability.md` — capability flags this lights up
- `docs/vmlx-feature-gap-report.md` — full Phase 1 gap analysis
- `docs/superpowers/plans/2026-05-09-vmlx-feature-parity.md` — phase plan + acceptance criteria
- `../memory/agent_memory.md` — Wake/Sleep must round-trip MoE state without losing expert routing context

## Why this is a separate doc area

Three reasons:

1. **It's the most active surface.** vMLX parity is a focused, time-bounded sprint; isolating its docs makes the progress visible.
2. **The architecture differs from dense.** MoE adds router decisions, expert dispatch, residency policy — dense-model docs don't carry those concepts.
3. **The quant schemes are new.** JANG/JANGTQ/VQ are not the same conceptual model as the GGUF Qx_K_M family; they deserve their own docs surface.
