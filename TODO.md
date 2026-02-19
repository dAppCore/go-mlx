# TODO.md — go-mlx Task Queue

Dispatched from core/go orchestration. Pick up tasks in order.

---

## Phase 1: Standalone Package Hardening

- [x] **Verify go generate → test round-trip** — ✅ 29/29 tests pass. CMake 3.24+, AppleClang 17.0.0, macOS SDK 26.2. Build takes ~2min on M3 Ultra.
- [x] **Add missing tests for core operations** — ✅ 86 new tests across 4 files: array_test.go (25), ops_test.go (44), nn_test.go (8), fast_test.go (9). Covers: all scalar/array creation, shape ops, element-wise arithmetic, math functions, matrix ops, reductions, indexing, slicing, fused kernels (RMSNorm, LayerNorm, RoPE, SDPA), Linear, Embedding, RepeatKV. Found non-contiguous view bug in Floats()/DataInt32() — see FINDINGS.md.
- [x] **Add missing tests for model/tokenizer/sample/cache** — ✅ 33 new tests: cache_test.go (10: KVCache + RotatingKVCache lifecycle, update, bounded, reset), sample_test.go (8: greedy, temperature, topK, chain, stub pass-through), tokenizer_test.go (15: Load/error, BOS/EOS, encode/decode, DecodeToken, SentencePiece space, GPT-2 byte maps). model/ still needs tests (requires model files on disk).
- [x] **Benchmark suite** — ✅ 29 benchmarks in bench_test.go. Covers: MatMul (128² to 4096², token-shaped 1×2048→32000), Softmax (1K to 128K vocab), element-wise (Add, Mul, SiLU at 1M elements), fused kernels (RMSNorm, LayerNorm, RoPE, SDPA at various shapes), Linear, Embedding, reductions (Sum, Argmax), and full sampler chain (greedy, TopK, TopP, combined). Baselined on M3 Ultra. model.Forward and tokenizer benchmarks deferred to Phase 2 (require model files on disk).

## Phase 2: Model Support

- [ ] **Gemma3-1B inference validation** — The go-i18n Phase 2a needs 1B model inference for domain classification at ~5K sentences/sec. Validate Gemma3-1B loads and generates correctly via `mlx.LoadModel()` + `m.Generate()`. Report tokens/sec.
- [ ] **Model loading robustness** — Test with missing files, corrupted safetensors, wrong dtype. Currently no error handling tests for `io.go`.
- [ ] **Add Llama model support** — Only Gemma3 and Qwen3 exist. Llama architecture would cover Meta's model family (Llama 3, CodeLlama).

## Phase 3: Training Pipeline

- [ ] **LoRA fine-tuning end-to-end** — `lora.go` has the adapter but no integration test showing: load base model → apply LoRA → train on small dataset → save adapter → reload. Critical for LEM Lab.
- [ ] **Gradient checkpointing** — `grad.go` has VJP but large models will OOM without checkpointing. Add selective recomputation.
- [ ] **Mixed precision training** — MLX supports BFloat16/Float16. Add dtype selection for training (currently inference uses model's native dtype).

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
- [ ] **Documentation** — Public API has godoc but needs examples for common workflows.

## Phase 5: Ecosystem Integration (Virgil wishlist)

- [ ] **Batch inference API** — go-i18n Phase 2a wants ~5K sentences/sec through Gemma3-1B. Single-prompt `Generate(..., WithMaxTokens(1))` works functionally for classification but won't hit 5K/sec. True batch inference (multiple prompts through one forward pass) is needed.
- [ ] **Inference metrics** — Expose tokens/sec, peak memory, GPU utilisation as structured data. LEM Lab dashboard and go-ai scoring engine both want this.
- [ ] **Model quantisation awareness** — MLX supports 4-bit and 8-bit quantised models. The loader already handles quantised safetensors (GroupSize, Bits in config).
- [ ] **Embed-friendly model loading** — Add `Discover(baseDir)` that scans for available models and returns metadata.
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

- [x] **`LoadOption` is ignored in `metalBackend.LoadModel()`** — ✅ Now calls `inference.ApplyLoadOpts()`. `ContextLen` passed through to `metal.LoadConfig` → stored on `Model` → replaces unbounded `KVCache` with `RotatingKVCache(contextLen)` in generate loop. `GPULayers=0` logs a warning (Metal always uses full GPU offload). New test: `TestNewCaches_ContextLen`.

### Important — Should Fix

- [ ] **KV cache leak between turns** — After `Generate()` returns, the KV cache arrays (allocated inside the closure at `generate.go:76`) are abandoned to GC finalizers. For multi-turn chat in LEM Lab, each turn allocates a new cache and the old one only gets freed when the GC eventually runs finalisers. Options: (a) accept a `*Cache` parameter so callers can reuse across turns, (b) add `Model.ResetCache()`, or (c) call `ClearCache()` after each turn. Document the expected pattern either way.

- [ ] **`RepeatPenalty` is accepted but never applied** — `GenerateConfig.RepeatPenalty` flows through from go-inference options and is stored in the config, but the generate loop has no token history and never applies it. Either implement it (track last N token IDs, penalise logits) or remove the field and document it as unsupported.

- [ ] **`DefaultGPUStream()` / `DefaultCPUStream()` leak and mislead** — `stream.go:32-41`: These create a new `C.mlx_stream` on every call but never free it. The names suggest they return *the* default stream (like `DefaultStream()` does with `sync.Once`), but they actually allocate. Either cache them like `DefaultStream()` or rename to `NewGPUStream()` / `NewCPUStream()` and add a `Free()` method.

- [ ] **Tokenizer `Encode()` is character-level only** — BPE merges are parsed and stored (`tokenizer.go:77-96`) but never applied in `Encode()` or `encodeGPT2()`. Both methods fall back to single-character lookup. This produces far more tokens than a real BPE encoder, drastically reducing effective context length and increasing prefill time. For Phase 2 model validation this will need fixing, especially for Gemma3-1B at 5K sentences/sec.

- [ ] **`CompileShapeless` is dead code** — `compile.go:82-89`: `Call()` bypasses the C closure entirely and just calls `cf.fn(inputs)` directly. The C closure, callback, and `sync.Map` registration are all unused overhead. The `shapeless` param is ignored. Either implement actual `mlx_compile` call-through or simplify to a plain function wrapper.

### Minor — Nice to Have

- [ ] **Rename `New()` → `newArray()`** — `array.go:32`: `New("INPUT")` is exported but only used internally as a pre-allocation pattern before C fills in `ctx`. Since this is `internal/metal`, exporting is harmless, but `newArray` better signals intent.

- [ ] **`Collect()` is unused** — `metal.go:125-133` defines `Collect()` to gather valid arrays for batch Materialize, but nothing in the codebase calls it. Dead code.

- [ ] **`qwen3.go` / `gemma3.go` — second `json.Unmarshal` error discarded** — Both model configs parse quantization with a second unmarshal whose error is silently dropped. Probably fine (same data already parsed successfully), but inconsistent with the first unmarshal which returns errors.

- [ ] **Document `AsStrided` stride formula** — `gemma3.go:354-359` uses stride manipulation for `[B,L,H*D]` → `[B,H,L,D]` virtual transpose. The formula is correct but non-obvious. A one-line comment explaining the stride derivation prevents future confusion.

### Questions for You to Consider

1. **Per-step intermediate freeing**: The design doc mentions `freeIntermediates(logits)` per decode step to reduce GC pressure. This isn't implemented — the generate loop creates ~500 intermediate arrays per forward pass that rely on GC finalizers. Is Go 1.26 Green Tea GC considered sufficient, or is explicit per-step freeing still planned?

2. **SentencePiece BPE**: The `merges` field is parsed but never used. For Gemma3's SentencePiece tokenizer, is character-level encoding sufficient (because the vocab contains full token strings), or is merge application a known gap for Phase 2?

3. **`nextID` in compile.go**: `nextID` is a `uintptr` used as `unsafe.Pointer` key into `sync.Map`. This works but `uintptr(0)` is never valid (starts at 1 after first increment). If `CompileShapeless` is kept, consider using `atomic.AddUint64` instead of mutex + plain increment.

## Workflow

1. Virgil in core/go writes tasks here after research
2. This repo's session picks up tasks in phase order
3. Mark `[x]` when done, note commit hash
4. New discoveries → add tasks, flag in FINDINGS.md
