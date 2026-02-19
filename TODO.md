# TODO.md — go-mlx Task Queue

Dispatched from core/go orchestration. Pick up tasks in order.

---

## Phase 1: Standalone Package Hardening

- [x] **Verify go generate → test round-trip** — ✅ 29/29 tests pass. CMake 3.24+, AppleClang 17.0.0, macOS SDK 26.2. Build takes ~2min on M3 Ultra.
- [x] **Add missing tests for core operations** — ✅ 86 new tests across 4 files: array_test.go (25), ops_test.go (44), nn_test.go (8), fast_test.go (9). Covers: all scalar/array creation, shape ops, element-wise arithmetic, math functions, matrix ops, reductions, indexing, slicing, fused kernels (RMSNorm, LayerNorm, RoPE, SDPA), Linear, Embedding, RepeatKV. Found non-contiguous view bug in Floats()/DataInt32() — see FINDINGS.md.
- [x] **Add missing tests for model/tokenizer/sample/cache** — ✅ 33 new tests: cache_test.go (10: KVCache + RotatingKVCache lifecycle, update, bounded, reset), sample_test.go (8: greedy, temperature, topK, chain, stub pass-through), tokenizer_test.go (15: Load/error, BOS/EOS, encode/decode, DecodeToken, SentencePiece space, GPT-2 byte maps). model/ still needs tests (requires model files on disk).
- [x] **Benchmark suite** — ✅ 29 benchmarks in bench_test.go. Covers: MatMul (128² to 4096², token-shaped 1×2048→32000), Softmax (1K to 128K vocab), element-wise (Add, Mul, SiLU at 1M elements), fused kernels (RMSNorm, LayerNorm, RoPE, SDPA at various shapes), Linear, Embedding, reductions (Sum, Argmax), and full sampler chain (greedy, TopK, TopP, combined). Baselined on M3 Ultra. model.Forward and tokenizer benchmarks deferred to Phase 2 (require model files on disk).

## Phase 2: Model Support

- [x] **Gemma3-1B inference validation** — ✅ End-to-end inference works. 4-bit quantised Gemma3-1B loads and generates coherently at **46 tok/s** on M3 Ultra. Fixed: `model_type: "gemma3_text"` not matched in architecture dispatch, GPT-2 BPE false detection on 262K SentencePiece vocab (checked `Ġthe` instead of bare `Ġ`). 3 new tests: inference (greedy, timing), chat template, context cancellation.
- [x] **Model loading robustness** — ✅ 24 new tests in model_test.go covering: missing/invalid config.json, unsupported architecture, `gemma3_text` dispatch, missing tokenizer, missing safetensors (was a nil-pointer panic — fixed with early error return in both LoadGemma3 and LoadQwen3), config parsing defaults/quantization/nested text_config, `isLayerSliding`, `resolveWeight` with prefix fallback.
- [x] **Add Qwen2 model support** — ✅ Qwen2 architecture (used by DeepSeek R1) now supported. Shares Qwen3 loader with optional Q/K RMS normalization (Qwen3 has it, Qwen2 does not). Auto-detected from weight presence. DeepSeek R1 7B: **27 tok/s** on M3 Ultra. 2 new tests.
- [x] **Add Llama model support** — ✅ Llama 3 architecture shares Qwen3 loader (same decoder: pre-norm, SwiGLU, GQA, no Q/K norm). Model type detected from config.json `model_type` field. Llama 3 chat template (`<|start_header_id|>`) and EOS token (`<|eot_id|>` id=128009) added. Llama 3.1 8B 4-bit: **30 tok/s** on M3 Ultra. 2 new tests.

## Phase 3: Training Pipeline

- [x] **LoRA fine-tuning end-to-end** — ✅ Full pipeline validated: load Gemma3-1B → apply LoRA (rank=8, q_proj+v_proj, 745K params) → 5 training steps with cross-entropy loss (7.15→6.31) → save adapter (2.9MB safetensors) → reload and verify weights match. Uses ValueAndGrad + AdamW. 1 new test in train_test.go.
- [x] **Gradient checkpointing** — ✅ `Checkpoint()` validates with real model training. Wraps forward pass to recompute activations during backward. Verified: produces correct gradients (loss 7.15→7.08 in 3 steps, matching non-checkpointed initial loss). 2 new tests: unit (grad_test.go) + model (train_test.go).
- [x] **Mixed precision training** — ✅ `LoRAConfig.DType` selects training dtype for A/B matrices. BFloat16 validated: loss 7.15→6.29 in 5 steps, matches Float32 accuracy with half param memory. MLX auto-promotes for cross-dtype ops. 1 new test in train_test.go.

## Phase 4: Backend Abstraction — ✅ COMPLETE (19 Feb 2026)

Design doc: `docs/plans/2026-02-19-backend-abstraction-design.md`
Implementation plan: `docs/plans/2026-02-19-backend-abstraction-plan.md`

**All Virgil review items implemented:**

- [x] **`context.Context` on `TextModel.Generate()`** — `Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]`. Checks `ctx.Done()` in the decode loop.
- [x] **`Err() error` on `TextModel`** — Distinguishes normal stop (EOS, max tokens) from errors (OOM, ctx cancelled).
- [x] **`Chat()` on `TextModel`** — Model owns its chat template. Gemma3 and Qwen3 templates implemented.
- [x] **Memory control functions at root** — `SetCacheLimit`, `SetMemoryLimit`, `GetActiveMemory`, `GetPeakMemory`, `ClearCache` delegate to `internal/metal`.
- [x] **Backend registration** — `register_metal.go` auto-registers via build-tagged `init()`.
- [x] **All CGO moved to `internal/metal/`** — 19 source files, 10 test files, 148 tests passing.
- [x] **Public API: `TextModel`, `Backend`, functional options** — Clean root package, compiles on all platforms.
- [x] **Integration tests** — 7 tests for public API (backend registration, options, LoadModel paths).
- [x] **Error handling audit** — ✅ `checkError()` replaced with `lastError() error` (reads + clears C-level error string). Added `Eval(...*Array) error` and `EvalAsync(...*Array) error` as error-returning variants of Materialize. Generate loop propagates errors via `m.lastErr`. `LoadAllSafetensors` returns `(map, error)`. Model loaders (gemma3, qwen3) check `lastError()` after safetensors load. grad.go/lora.go now surface real MLX error messages. 4 new tests in error_test.go.
- [x] **Memory management — deterministic cleanup** — ✅ `Model.Close()` now walks the full model tree (GemmaModel/Qwen3Model) and explicitly frees all weight arrays via `Free()`. Helpers: `freeLinear`, `freeEmbedding`, `freeRMSNorm`, `freeCaches`, `closeGemma`, `closeQwen3` in close.go. Handles tied output weights (skip double-free), nil safety, idempotent Close(). 8 new tests in close_test.go.
- [x] **Documentation** — ✅ Package docs expanded with examples for all common workflows: Generate, Chat, Classify, BatchGenerate, Metrics, ModelInfo, Discover, memory controls. Both go-mlx and go-inference package docs updated with godoc heading sections.

## Phase 5: Ecosystem Integration (Virgil wishlist)

- [x] **Batch inference API** — ✅ `Classify` (prefill-only, fast path) and `BatchGenerate` (autoregressive) implemented. Added `ForwardMasked` to InternalModel interface, threaded attention masks through Gemma3 and Qwen3 decoders. Mask: `[N, 1, L, L]` combining causal + padding (0=attend, -inf=ignore). Right-padded, sorted by length. Gemma3-1B 4-bit: **152 prompts/s** classify (4 prompts), BatchGenerate produces coherent per-prompt output. Types (`ClassifyResult`, `BatchResult`, `WithLogits`) in go-inference. 6 new tests (3 mask unit, 3 model). Design doc: `docs/plans/2026-02-19-batch-inference-design.md`.
- [x] **Inference metrics** — ✅ `GenerateMetrics` type in go-inference with `Metrics()` on `TextModel`. Captures: prefill/decode timing, token counts, throughput (tok/s), peak and active GPU memory. Instrumented Generate, Classify, and BatchGenerate. Gemma3-1B 4-bit: prefill 246 tok/s, decode 82 tok/s, peak 6.2 GB. 1 new test.
- [x] **Model quantisation awareness** — ✅ `ModelInfo` type in go-inference with `Info()` on `TextModel`. Exposes architecture, vocab size, layer count, hidden dimension, quantisation bits and group size. Loader already handles quantised safetensors transparently. 1 new test.
- [x] **Embed-friendly model loading** — ✅ `Discover(baseDir)` in go-inference scans for model directories (config.json + *.safetensors). Returns `DiscoveredModel` with path, architecture, quantisation, file count. Finds 20 models across the lab. 1 new test.
- [ ] **mlxlm/ backend** — Python subprocess wrapper via `core/go/pkg/process`. Implements `mlx.Backend` for mlx_lm compatibility.

## Phase 6: Go 1.26 Modernisation

- [x] **Evaluate Go 1.26 features** — ✅ Documented in FINDINGS.md. Key wins: CGO ~30% faster (free), Green Tea GC default (10-40% less overhead, helps Array finalisers), slice stack alloc.
- [x] **Range-over-func for Array** — ✅ `Array.Iter() iter.Seq[float32]` implemented in array.go. Handles non-contiguous arrays via ensureContiguous(). Supports early break. 4 tests: basic, 2D flatten, transposed, early break.

---

## go-inference Integration — ✅ COMPLETE (19 Feb 2026)

All types (`TextModel`, `Backend`, `Token`, `Message`, options) moved to shared `forge.lthn.ai/core/go-inference` package. go-mlx is now a pure backend implementation — import `_ "forge.lthn.ai/core/go-mlx"` to register the `"metal"` backend. See FINDINGS.md for migration details.

## Upstream Dependencies

- **go-i18n Phase 2a** is blocked on this package providing working Gemma3-1B inference
- **go-ml/backend_mlx.go** needs updating to use `inference.LoadModel()` + `m.Generate()` (types from go-inference, `_ "go-mlx"` for Metal registration)
- **go-ai** has a `replace` directive pointing at `../go-mlx`. No code changes needed in go-ai itself.
- **go-rocm** — sibling backend for AMD GPUs, implements same `inference.Backend` interface
- **LEM Lab** uses `MLXBackend` via go-ml. Migration transparent once go-ml updates.

## Functional Options Convention

Virgil confirms: the `WithMaxTokens(n)` functional option pattern is the right call for this package.

## core/go/pkg/process (for mlxlm backend, Phase 5)

Virgil confirms: no changes needed. The process package provides everything needed for the mlxlm subprocess backend.

## Virgil Code Review — 19 Feb 2026

Full codebase review after Phase 4 completion + go-inference integration. Grouped by priority.

### Critical — Fix Before Phase 2

- [x] **Error handler thread safety** — ✅ `last_mlx_error` now uses `_Atomic(const char*)` with `atomic_store_explicit` (release) / `atomic_exchange_explicit` (acquire). Thread-safe even if MLX calls the error handler from background threads.

- [x] **`-mmacosx-version-min=26.0` is wrong** — ✅ Changed to `13.3` (MLX's own minimum). No longer locks out macOS 14/15 users.

- [x] **`LoadOption` is ignored in `metalBackend.LoadModel()`** — ✅ Now calls `inference.ApplyLoadOpts()`. `ContextLen` passed through to `metal.LoadConfig` → stored on `Model` → replaces unbounded `KVCache` with `RotatingKVCache(contextLen)` in generate loop. `GPULayers=0` logs a warning (Metal always uses full GPU offload). newArray test: `TestNewCaches_ContextLen`.

### Important — Should Fix

- [x] **KV cache leak between turns** — ✅ Documented in Generate() godoc: each call allocates fresh KV caches released to GC; call ClearCache() between turns for prompt reclaim. Cache reuse across turns deferred to batch inference design (Phase 5).

- [x] **`RepeatPenalty` is accepted but never applied** — ✅ Implemented `applyRepeatPenalty()` in generate.go. Tracks generated token IDs, deduplicates, then for each seen token: divides positive logits by penalty, multiplies negative logits by penalty. Applied before sampling when `RepeatPenalty > 1.0`. 2 new tests.

- [x] **`DefaultGPUStream()` / `DefaultCPUStream()` leak and mislead** — ✅ Now cached with `sync.Once` like `DefaultStream()`. No more allocation on every call.

- [x] **Tokenizer `Encode()` is character-level only** — ✅ Implemented `bpeMerge()` — standard BPE algorithm using merge rank lookup. Both SentencePiece `Encode()` and GPT-2 `encodeGPT2()` now split into characters, apply BPE merges, then look up merged symbols. Merge ranks built during tokenizer load. 3 new tests.

- [x] **`CompileShapeless` is dead code** — ✅ Removed C closure, callback, `sync.Map`, and `nextID` infrastructure. `CompiledFunc` is now a plain function wrapper with mutex. `CompileShapeless()` and `Call()` signatures unchanged (gemma3.go GELU still works).

### Minor — Nice to Have

- [x] **Rename `New()` → `newArray()`** — ✅ Renamed via IDE refactoring (112 usages updated). Unexported, signals internal-only intent.

- [x] **`Collect()` is unused** — ✅ Removed function and its test. Dead code eliminated.

- [x] **`qwen3.go` — second `json.Unmarshal` error discarded** — ✅ Now checks and returns the error. gemma3.go already handled it correctly.

- [x] **Document `AsStrided` stride formula** — ✅ Added comment explaining the stride derivation for the `[B,L,H*D]` → `[B,H,L,D]` virtual transpose.

### Questions for You to Consider

1. **Per-step intermediate freeing**: The design doc mentions `freeIntermediates(logits)` per decode step to reduce GC pressure. This isn't implemented — the generate loop creates ~500 intermediate arrays per forward pass that rely on GC finalizers. Is Go 1.26 Green Tea GC considered sufficient, or is explicit per-step freeing still planned?

2. **SentencePiece BPE**: The `merges` field is parsed but never used. For Gemma3's SentencePiece tokenizer, is character-level encoding sufficient (because the vocab contains full token strings), or is merge application a known gap for Phase 2?

3. **`nextID` in compile.go**: `nextID` is a `uintptr` used as `unsafe.Pointer` key into `sync.Map`. This works but `uintptr(0)` is never valid (starts at 1 after first increment). If `CompileShapeless` is kept, consider using `atomic.AddUint64` instead of mutex + plain increment.

## Workflow

1. Virgil in core/go writes tasks here after research
2. This repo's session picks up tasks in phase order
3. Mark `[x]` when done, note commit hash
4. newArray discoveries → add tasks, flag in FINDINGS.md
