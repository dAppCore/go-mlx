# TODO.md — go-mlx Task Queue

Dispatched from core/go orchestration. Pick up tasks in order.

---

## Phase 1: Standalone Package Hardening

- [x] **Verify go generate → test round-trip** — ✅ 29/29 tests pass. CMake 3.24+, AppleClang 17.0.0, macOS SDK 26.2. Build takes ~2min on M3 Ultra.
- [x] **Add missing tests for core operations** — ✅ 86 new tests across 4 files: array_test.go (25), ops_test.go (44), nn_test.go (8), fast_test.go (9). Covers: all scalar/array creation, shape ops, element-wise arithmetic, math functions, matrix ops, reductions, indexing, slicing, fused kernels (RMSNorm, LayerNorm, RoPE, SDPA), Linear, Embedding, RepeatKV. Found non-contiguous view bug in Floats()/DataInt32() — see FINDINGS.md.
- [x] **Add missing tests for model/tokenizer/sample/cache** — ✅ 33 new tests: cache_test.go (10: KVCache + RotatingKVCache lifecycle, update, bounded, reset), sample_test.go (8: greedy, temperature, topK, chain, stub pass-through), tokenizer_test.go (15: Load/error, BOS/EOS, encode/decode, DecodeToken, SentencePiece space, GPT-2 byte maps). model/ still needs tests (requires model files on disk).
- [ ] **Benchmark suite** — No benchmarks exist. Add: MatMul (various sizes), Softmax, model.Forward (single token), tokenizer.Encode/Decode, full Generate (tokens/sec). Baseline on M3 Ultra.

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
- [ ] **Error handling audit** — `checkError()` still logs + swallows. Needs conversion to error returns. Low priority — existing behaviour, not a regression.
- [ ] **Memory management — deterministic cleanup** — `Close()` stub in place. Needs CLion Claude research on `mlx_array_free` safety before implementing per-step cleanup. See `cpp/TODO.md`.
- [ ] **Documentation** — Public API has godoc but needs examples for common workflows.

## Phase 5: Ecosystem Integration (Virgil wishlist)

- [ ] **Batch inference API** — go-i18n Phase 2a wants ~5K sentences/sec through Gemma3-1B. Single-prompt `Generate(..., WithMaxTokens(1))` works functionally for classification but won't hit 5K/sec. True batch inference (multiple prompts through one forward pass) is needed.
- [ ] **Inference metrics** — Expose tokens/sec, peak memory, GPU utilisation as structured data. LEM Lab dashboard and go-ai scoring engine both want this.
- [ ] **Model quantisation awareness** — MLX supports 4-bit and 8-bit quantised models. The loader already handles quantised safetensors (GroupSize, Bits in config).
- [ ] **Embed-friendly model loading** — Add `Discover(baseDir)` that scans for available models and returns metadata.
- [ ] **mlxlm/ backend** — Python subprocess wrapper via `core/go/pkg/process`. Implements `mlx.Backend` for mlx_lm compatibility.

## Phase 6: Go 1.26 Modernisation

- [x] **Evaluate Go 1.26 features** — ✅ Documented in FINDINGS.md. Key wins: CGO ~30% faster (free), Green Tea GC default (10-40% less overhead, helps Array finalisers), slice stack alloc.
- [ ] **Range-over-func for Array** — `Array.Iter()` returning `iter.Seq[float32]` for cleaner iteration. Measure overhead vs direct C pointer access.

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

## Workflow

1. Virgil in core/go writes tasks here after research
2. This repo's session picks up tasks in phase order
3. Mark `[x]` when done, note commit hash
4. New discoveries → add tasks, flag in FINDINGS.md
