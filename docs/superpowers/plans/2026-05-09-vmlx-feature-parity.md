<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# vMLX Feature Parity Plan

Date: 2026-05-09

Target repo: `/Users/snider/Code/core/go-mlx`

Competitor audit source: `/private/tmp/vmlx-audit-20260509`

## Goal

Bring the Core native Go/MLX stack up to practical feature parity with the
runtime capabilities exposed by vMLX while preserving the Core architecture:
package-first, Go-native, no Python hot path, no Electron dependency, and no
provider policy in the low-level runtime.

CLI, TUI, UI, and distributed compute are not part of the first parity pass.
HTTP compatibility is included only as reusable package/server primitives.

## Architecture Rules

- `go-inference` owns shared model, generation, stream, capability, and HTTP wire
  primitives.
- `go-mlx` implements Apple MLX/Metal local runtime behaviour.
- `go-rocm` and future `go-cuda` mirror the same primitives where hardware allows.
- `go-ai` owns provider routing, external API keys, rate limits, fallback policy,
  and higher-level chat/research/task workflows.
- `go-ml` owns model-building workflows.
- `core/api` can host handlers, but must not become the AI policy layer.
- Use the local `go.work` during active Core development. Do not force
  `GOWORK=off` while unpublished local dev APIs are intentionally linked.

## Phase 1: MiniMax/JANGTQ Native Runtime

### 1. Finish JANG/JANGTQ Capability Metadata

Files likely involved:

- `go/jang.go`
- `go/gguf_info.go`
- `go/model_pack.go`
- `go/hf_fit.go`
- `go/memory_plan.go`
- matching `*_test.go` files

Tasks:

- Stabilise current JANG/JANGTQ metadata recognition.
- Expose JANG profile, packed dtype, group size, codebook flags, and MoE expert
  hints through `ModelPack`, `ModelInfo`, `MemoryPlan`, and benchmark reports.
- Add fixture tests for MiniMax M2.7/JANGTQ_K-style metadata without needing the
  full model.
- Add negative tests for unsupported packed shapes and missing metadata.

Validation:

- `go test ./... -run 'JANG|JANGTQ|MiniMax|ModelPack|MemoryPlan' -count=1`

### 2. Add Native Packed Tensor Loading

Files likely involved:

- `go/internal/metal/model.go`
- `go/internal/metal/*quant*`
- `go/gguf_info.go`
- `go/model_pack.go`

Tasks:

- Add a JANGTQ/MXTQ tensor descriptor independent of GGUF naming quirks.
- Implement CPU-side metadata parsing and Metal-side dequant staging for the
  first profile needed by MiniMax M2.7/JANGTQ_K.
- Keep tensor IO streaming; do not require all experts in RAM during validation.
- Emit probe events for dequant profile, source dtype, target dtype, and load
  latency.

Validation:

- Small fake packed tensor round-trip tests.
- Native Metal tests behind existing Metal test gates.

### 3. Implement MiniMax M2-Class MoE Forward

Files likely involved:

- `go/internal/metal/model.go`
- `go/model_pack.go`
- `go/memory_plan.go`
- `go/probe*.go`
- `go/lora*.go`

Tasks:

- Add MiniMax config parsing and architecture detection.
- Implement router logits, top-k expert selection, expert projection dispatch,
  and result accumulation for a minimal MiniMax M2-class block.
- Wire LoRA target mapping and probe emission for router decisions and expert
  load.
- Add memory-plan hints for active experts, resident experts, and smelt-ready
  lazy residency.

Validation:

- Deterministic fake-model forward tests.
- Native skip tests for real MiniMax/JANGTQ assets when absent.
- Bench report entries for prefill/decode/load memory.

## Phase 2: Compatibility Surface

### 4. Tool And Reasoning Parser Registry

Files likely involved:

- `go/thinking*.go`
- `go/openai*.go`
- new `go/parsers*.go`

Tasks:

- Add typed parser interfaces for reasoning spans and tool-call extraction.
- Add parser families for Qwen, Gemma, DeepSeek R1, GPT-OSS, Mistral, MiniMax,
  Kimi, GLM, Hermes, Granite, and generic XML/JSON fallback.
- Make parser selection model-aware through `ModelInfo`/capabilities.
- Ensure stream chunks can either hide, show, or separately capture reasoning.

Validation:

- Fake-tokenizer tests for each parser family.
- Streaming tests for partial tags and malformed tool JSON.

### 5. Request Scheduler, Cancellation, And Backpressure

Files likely involved:

- `go/openai*.go`
- `go/bench*.go`
- new `go/scheduler*.go`

Tasks:

- Add a package-level scheduler around `inference.TextModel` that supports queued
  prefill/decode jobs, streaming, cancellation IDs, and bounded concurrency.
- Emit queue latency, first-token latency, tokens/sec, cache hit rate, and memory
  pressure probe events.
- Keep scheduler optional so library users can still call the model directly.

Validation:

- Mock model tests for cancellation before prefill, during decode, and after
  completion.
- Backpressure tests with slow stream consumers.

### 6. Block Prefix Cache Service

Files likely involved:

- `go/prompt_cache*.go`
- `go/kv_snapshot*.go`
- `go/state_bundle*.go`
- `go/bench*.go`

Tasks:

- Move from exact prompt cache semantics toward token-block identity.
- Track block hits, misses, evictions, restore time, fork/copy-on-write events,
  and adapter/model compatibility.
- Keep compatibility with `StateBundle` and KV snapshots.
- Add cache stats structs that can be served by API layers without importing
  server code.

Validation:

- Tests for overlapping prefixes, adapter mismatch, tokenizer mismatch, and
  restored bundle cache reuse.
- Bench reports include hit rate and restore latency.

### 7. Disk-Backed KV Block Cache

Files likely involved:

- `go/kv_snapshot*.go`
- `go/prompt_cache*.go`
- `go/bench*.go`

Tasks:

- Add binary q8/q4-aware block serialisation separate from full state bundles.
- Add a bounded disk cache with content-addressed blocks and corruption checks.
- Support warm, list, stats, and clear operations at the package level.
- Ensure memory planner can choose disk cache only when restore cost beats
  recompute for the current model/context.

Validation:

- Round-trip tests for q8 and unquantised blocks.
- Fault tests for truncated/corrupt block files.

## Phase 3: Wire Compatibility

### 8. OpenAI Responses, Anthropic Messages, And Ollama Adapters

Files likely involved:

- `go/openai*.go`
- `go/server*.go`
- shared `go-inference` package in the Core workspace

Tasks:

- Add OpenAI Responses request/response/event primitives.
- Add Anthropic Messages adapter over the same `TextModel` contract.
- Add Ollama chat/generate/tags/show compatibility handlers.
- Keep provider routing and external API keys out of `go-mlx`.

Validation:

- Mock model handler tests for stop handling, stream chunks, reasoning capture,
  tool calls, model resolution, and cancellation.

### 9. Capability, Cache, And Admin Handler Set

Files likely involved:

- `go/server*.go`
- `go/model_info*.go`
- `go/memory_plan.go`
- `go/prompt_cache*.go`

Tasks:

- Expose model capability structs through reusable handlers.
- Add health, wake/sleep hooks, cache stats, cache entries, cache warm, and cache
  clear handlers.
- Keep sleep/wake as runtime callbacks so Core native GUI or `core/api` can own
  process policy.

Validation:

- Handler tests with mock runtime and cache service.

### 10. Embeddings And Rerank Contracts

Files likely involved:

- `go/model_info*.go`
- `go/dataset*.go`
- new `go/embeddings*.go`
- shared `go-inference`

Tasks:

- Add embeddings model interface and vector response structs.
- Add rerank/scoring interface for cross-encoder or decoder-score models.
- Add BERT embedding model-pack detection and memory-plan hints.
- Wire OpenAI-compatible embeddings and vLLM-style rerank handler primitives.

Validation:

- Mock embedding/rerank tests.
- Native skip tests for real embedding model packs.

## Phase 4: Decode And MoE Optimisation

### 11. Speculative Decoding And Prompt Lookup Decoding

Files likely involved:

- `go/generate*.go`
- `go/scheduler*.go`
- `go/bench*.go`

Tasks:

- Add draft-model speculative decode API with acceptance metrics.
- Add prompt lookup decoding for repeated-context workloads.
- Make both modes visible in benchmark reports.
- Do not enable by default until benchmark data proves the workload win.

Validation:

- Mock deterministic acceptance/rejection tests.
- Bench comparisons for standard decode vs speculative/PLD.

### 12. Smelt-Style Lazy Expert Residency

Files likely involved:

- `go/internal/metal/model.go`
- `go/memory_plan.go`
- `go/probe*.go`

Tasks:

- Add optional expert residency policy for MoE models.
- Load only configured hot experts at startup.
- Page cold experts in/out with explicit probe events and latency accounting.
- Integrate with memory planner for M1 16GB, M3 Ultra 96GB, and ROCm-class
  16GB devices through shared capability primitives.

Validation:

- Fake expert loader tests for residency decisions.
- Bench memory peak and first-use latency.

### 13. Codebook/VQ Kernel Lane

Files likely involved:

- `go/internal/metal/*`
- `go/model_pack.go`
- `go/bench*.go`

Tasks:

- Add codebook tensor metadata and validation.
- Implement the smallest useful codebook matvec kernel.
- Add model-pack feature flags so unsupported codebook models fail clearly.

Validation:

- Fake codebook tensor tests.
- Native Metal correctness tests with tiny matrices.

## Phase 5: Model Family Expansion

### 14. Add Families One Patch At A Time

Order:

1. MiniMax M2/M2.7.
2. Mistral/Mixtral.
3. DeepSeek V2/V3/V4.
4. Phi.
5. GLM/Kimi/StepFun.
6. Nemotron/Laguna/ZAYA.
7. BERT embeddings.
8. Vision/omni only after text runtime is stable.

Each family patch must include:

- Model-pack detection.
- Config parsing.
- Loader mapping.
- Generation or embedding tests with fake weights.
- Native skip test for real assets.
- LoRA target mapping where applicable.
- Memory-plan hints.
- Parser selection where applicable.

## Phase 6: Proof Harness

### 15. Parity Bench Report

Files likely involved:

- `go/bench*.go`
- `go/eval*.go`
- `go/probe*.go`

Tasks:

- Add a single JSON report section for competitor-parity checks:
  model load time, resident memory, prefill tok/s, decode tok/s, first-token
  latency, cache hit rate, KV restore time, adapter overhead, scheduler queue
  latency, and parser/tool-call correctness.
- Add comparison labels for `native`, `adapter`, `quantised`, `paged`, `disk-l2`,
  `speculative`, and `smelt`.

Validation:

- Deterministic mock benchmark tests.
- Optional native benchmark smoke on the local M3.

## Definition Of Done

- MiniMax M2.7/JANGTQ_K-class metadata is inspected correctly.
- At least one JANGTQ packed profile can run through native load/dequant tests.
- MiniMax-style MoE fake forward path passes deterministic tests.
- API compatibility handlers cover OpenAI Chat/Responses, Anthropic Messages,
  Ollama chat/generate/tags/show, capabilities, cache stats, and cancellation.
- Cache reports include block hit rate, disk restore time, and memory pressure.
- Parser tests cover tool calls and reasoning spans across the target families.
- Bench report data can justify any default memory/cache/scheduler decision.
