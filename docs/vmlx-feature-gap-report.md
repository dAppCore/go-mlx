<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# vMLX Feature Gap Report

Date: 2026-05-09

Competitor source audited: `https://github.com/jjang-ai/vmlx`, cloned locally at
`/private/tmp/vmlx-audit-20260509`.

This report compares vMLX against `go-mlx` as a package-first Apple native MLX
runtime. It intentionally treats CLI, TUI, UI, and distributed compute as lower
priority unless they unlock runtime capability parity.

## Executive Summary

vMLX is broad. Its strongest feature claim is not the Electron panel; it is the
combination of a Python MLX engine, OpenAI/Anthropic/Ollama-compatible HTTP
surfaces, wide model-family dispatch, JANG/JANGTQ quantisation support, paged
cache work, tool/reasoning parser coverage, multimodal endpoints, and operational
model management.

`go-mlx` is already ahead in the areas that matter for the Core direction:
native Go APIs, model-state bundles, KV snapshots, probe bus, LoRA SFT,
distillation, GRPO, eval, memory planning, model-pack validation, GGUF work,
and low-process-overhead integration with the wider Core Go stack. The largest
gap is not "can it launch an app"; it is "can it load and serve the same weird
model zoo natively without falling back to Python".

The highest-value parity target is therefore:

1. Native JANG/JANGTQ/MXTQ loading and runtime support for MiniMax M2-class MoE.
2. Runtime scheduler/cache parity: continuous batching, cancellation, stronger
   block-prefix cache, disk-backed KV blocks, and cache observability.
3. Wire-compatibility parity: OpenAI Responses, Anthropic Messages, Ollama, model
   capabilities, cache/admin endpoints, embeddings, and rerank.
4. Parser parity: tool-call and reasoning-channel registries per model family.
5. Model-family expansion after the above substrate exists.

## Competitor Architecture

The cloned vMLX repo is primarily:

- Python engine under `vmlx_engine/`.
- FastAPI HTTP server in `vmlx_engine/server.py`.
- MLX Python ecosystem integration through `mlx`, `mlx-lm`, `mlx-vlm`,
  `mlx-embeddings`, `mflux`, and optional `mlx-audio`.
- Hard dependency on `jang` / `jang_tools` for JANG and JANGTQ paths.
- Legacy Electron/React panel under `panel/`, including Python bundling scripts.
- Apache-2.0 licensed root project.

The README points users toward a newer Swift desktop app release, but the cloned
repo still carries a legacy Electron panel. For Core, the important comparison is
the engine/API feature set, not the panel.

## Core Advantages

`go-mlx` has several advantages that vMLX does not appear to have as first-class
native concepts:

- Go-native package surface with no Python runtime on the hot path.
- Research-grade model-state APIs: `StateBundle`, `KVSnapshot`, prompt hash,
  sampler metadata, adapter identity, probe metrics, and restore compatibility.
- Probe bus and eval/bench surfaces designed as library primitives.
- Native training-oriented APIs: LoRA SFT, distillation, GRPO, dataset stream,
  eval, LoRA fuse, model merge, and model pack inspection.
- Memory planner aimed at real Apple machine classes rather than generic knobs.
- Low-overhead native-app integration in the wider Core suite.

This is the product wedge: do not copy vMLX's process shape. Close the runtime
and compatibility gaps while keeping the Go-native, package-first architecture.

## Feature Gap Matrix

| Area | vMLX Evidence | go-mlx State | Gap |
| --- | --- | --- | --- |
| OpenAI chat completions | `/v1/chat/completions` | Present as a Go adapter | Mostly aligned |
| OpenAI Responses API | `/v1/responses` | Not first-class | Add shared primitive and handler |
| Anthropic Messages API | `/v1/messages` | Not first-class | Add adapter in shared HTTP layer |
| Ollama API | `/api/chat`, `/api/generate`, `/api/tags`, etc. | Not first-class | Add compatibility package outside core runtime policy |
| Model capability endpoint | `/v1/models/{id}/capabilities` | Capability structs exist across Core work | Add HTTP exposure and runtime-backed reporting |
| Cache endpoints | Stats, entries, warm, clear | Bench/cache primitives exist | Add package HTTP handlers and richer cache state |
| Request cancellation | Cancel endpoints for chat/responses/completions/images | Not surfaced as API contract | Add context/cancel IDs to adapter layer |
| Continuous batching | Batched engine/scheduler | Batch APIs exist, not request scheduler parity | Add scheduler package around `TextModel` |
| Prefix cache | Engine prefix cache | Prompt cache exists | Upgrade to block-prefix cache with hit telemetry |
| Paged KV cache | Paged cache and block cache | Quantised/paged cache work exists | Finish no-concat page attention and disk block store |
| Disk cache | L2/block disk cache | KV snapshots exist | Add hot block cache, not only durable snapshots |
| JANG/JANGTQ | `jang_tools`, JANG profiles, JANGTQ loader | Metadata recognition underway | Need native load/dequant/dispatch path |
| MXTQ / JANG profiles | `JANG_2M`, `2L`, `3M`, `4M`, `6M` | Shape/metadata recognition only | Implement profile planner and kernels |
| MiniMax M2/M2.7 | Claimed supported | Recognised/partially planned | Need native MoE forward and JANGTQ weights |
| Smelt partial experts | Partial MoE expert loading | Not present | Add lazy expert residency after MoE works |
| Codebook kernels | VQ/codebook source and Metal kernels | Not present | Add later for JANG/codebook models |
| Speculative decoding | Claimed | Not first-class | Add draft-model decode API |
| Prompt lookup decoding | Claimed | Not first-class | Add PLD path after scheduler/cache |
| Tool-call parsers | Many model families | Limited | Add parser registry and family tests |
| Reasoning parsers | Qwen, DeepSeek, GPT-OSS, Mistral, Gemma-style | Qwen/Gemma thinking path exists | Expand parser matrix |
| Vision models | MLX-VLM path | Not native | Later model-family lane |
| Image generation/edit | mflux endpoints | Not native | Out of core runner scope unless Core app needs it |
| Audio STT/TTS | mlx-audio endpoints | Not native | Out of core runner scope initially |
| Embeddings | `/v1/embeddings`, mlx-embeddings | BERT embeddings listed as future arch | Add embeddings runtime contract |
| Rerank | `/v1/rerank` | Not first-class | Add scoring/rerank contract |
| Distributed Macs | Cluster endpoints | Explicitly lower priority | Defer |
| Native low-memory app | Electron panel plus separate Swift release | Core native app path | Core advantage |

## Highest-Risk Gaps

### JANG/JANGTQ Is The Main Runtime Gap

The vMLX JANG path delegates heavily to `jang_tools`, but from a user point of
view it is the visible differentiator for MiniMax M2.7/JANGTQ_K models. For
`go-mlx`, metadata recognition is not enough. Feature parity needs:

- JANG profile parsing.
- Packed tensor dtype and shape validation.
- Gate/up/down projection dequantisation.
- MoE router and expert dispatch support for MiniMax M2-class models.
- Memory planner estimates for compressed experts and active expert residency.
- Bench coverage showing native Go/Metal behaviour on M3-class hardware.

### API Compatibility Is A Suite Gap, Not A Runtime Gap

The HTTP protocols should not make `go-mlx` depend on `go-ai` or `core/api`.
The shared primitives should stay in `go-inference`; `go-mlx` should mount local
handlers; `go-ai` can later add providers, policy, keys, fallback, and
rate-limiting.

The parity target is a small set of reusable compatibility packages:

- OpenAI Chat/Responses.
- Anthropic Messages.
- Ollama chat/generate/tags/show.
- Embeddings and rerank.
- Cache/admin/model-capability handlers.

### Cache Parity Needs A Runtime Contract

vMLX exposes cache as a user-visible subsystem. `go-mlx` already has stronger
research-grade state objects, but parity requires a request-time cache service:

- Prefix block identity.
- Block hit/miss accounting.
- Copy-on-write fork semantics where possible.
- Disk L2 for cold KV blocks.
- Fast restore benchmarks included in reports.

### Parser Coverage Is Cheap And High-Impact

Tool-call and reasoning parsing is mostly token/text protocol work. This is one
of the fastest ways to improve compatibility with current model releases without
waiting on new kernels.

## What Not To Copy

- Do not reproduce a monolithic Python API server.
- Do not require Python, Torch, Electron, or Node for local inference.
- Do not put provider keys, routing policy, or rate limits inside `go-inference`.
- Do not chase every endpoint before the native runtime can load the target
  models.
- Do not optimise for distributed Macs until single-machine behaviour is
  measured and stable.

## Recommended Parity Order

1. Finish JANG/JANGTQ metadata, planner, and model-pack validation.
2. Implement native JANGTQ/MXTQ tensor load and dequant primitives.
3. Add MiniMax M2/M2.7 MoE forward path and LoRA/probe metadata hooks.
4. Add parser registry for tool calls and reasoning channels.
5. Add continuous request scheduler with cancellation and streaming backpressure.
6. Upgrade prompt cache to block-prefix cache with cache service metrics.
7. Add disk-backed KV block cache and binary/quantised snapshot interop.
8. Expand shared HTTP compatibility: Responses, Anthropic, Ollama, capabilities,
   cache/admin endpoints.
9. Add embeddings and rerank contracts.
10. Add speculative decoding and prompt lookup decoding.
11. Add Smelt-style lazy expert residency for MoE.
12. Expand model families one at a time using the same loader/test template.

The first three items determine whether `go-mlx` can credibly claim MiniMax
M2.7/JANGTQ parity. The next five determine whether apps and agents can use the
runner as a drop-in local backend.
