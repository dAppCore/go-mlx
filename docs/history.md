# Project History

Module: `forge.lthn.ai/core/go-mlx`

---

## Origin

go-mlx was extracted from `forge.lthn.ai/core/go-ai/mlx/` on 19 February 2026. The split was motivated by three concerns:

1. **Platform isolation.** mlx is darwin/arm64 only with CGO and CMake build requirements. Keeping it inside go-ai forced the entire AI package to carry platform-specific build complexity.
2. **Dependency chain.** go-i18n Phase 2a needs Gemma3-1B inference for domain classification and should not pull in all of go-ai (DuckDB, Parquet, gRPC, Ollama, etc.) as a transitive dependency.
3. **Build tag cleanliness.** Every file is `//go:build darwin && arm64`. As a standalone module this is the default; inside go-ai it was a special case requiring careful build tag management.

Initial extraction: commit `cae7ef0` — 22 files, approximately 4,354 lines, comprising core MLX bindings, Gemma3 and Qwen3 model implementations, BPE tokeniser, sampling strategies, and KV cache.

---

## Phase 1: Standalone Package Hardening

**Commits:** `37abc49` through `40cbdd7`

- **Go generate → test round-trip verified.** 29/29 tests pass. CMake 3.24+, AppleClang 17.0.0, macOS SDK 26.2. Build approximately 2 minutes on M3 Ultra.
- **86 new tests** across `array_test.go` (25), `ops_test.go` (44), `nn_test.go` (8), `fast_test.go` (9). Covers all scalar/array creation, shape operations, element-wise arithmetic, math functions, matrix operations, reductions, indexing, slicing, fused kernels (RMSNorm, LayerNorm, RoPE, SDPA), Linear, Embedding, RepeatKV.
- **33 new tests** for cache, sample, tokeniser: KVCache and RotatingKVCache lifecycle and update, greedy/temperature/TopK sampling chain, BPE load and encode/decode, DecodeToken, SentencePiece space handling, GPT-2 byte maps.
- **Non-contiguous view bug discovered:** `Floats()` and `DataInt32()` read `Size()` elements from the raw C data pointer. For non-contiguous arrays (transpose, broadcast, slice views), the physical layout does not match the logical layout. Workaround documented: `Reshape(arr, totalSize)` forces a contiguous copy. Fixed in Phase 4.
- **29 benchmarks** in `bench_test.go` baselined on M3 Ultra.

---

## Phase 2: Model Support

**Commits:** `18e8dca` through `a2493e0`

- **Gemma3-1B inference validated.** End-to-end inference: 4-bit quantised Gemma3-1B loads and generates coherently at 46 tok/s on M3 Ultra. Two bugs fixed: `model_type: "gemma3_text"` was not matched in architecture dispatch; GPT-2 BPE false detection on 262K SentencePiece vocab (fixed by checking `Ġthe` rather than bare `Ġ`).
- **24 model loading robustness tests:** missing/invalid config, unsupported architecture, `gemma3_text` dispatch, missing tokeniser, missing safetensors (was a nil-pointer panic — fixed with early error return in both `LoadGemma3` and `LoadQwen3`), config parsing defaults/quantisation/nested `text_config`, `isLayerSliding`, `resolveWeight` with prefix fallback.
- **Qwen2 support added.** Shares Qwen3 loader with optional Q/K RMS normalisation (Qwen3 has it, Qwen2 does not); auto-detected from weight presence. DeepSeek R1 7B: 27 tok/s on M3 Ultra.
- **Llama 3 support added.** Shares Qwen3 loader (same decoder: pre-norm, SwiGLU, GQA, no Q/K norm). Model type detected from `config.json` `model_type`. Llama 3 chat template and EOS token (`<|eot_id|>` id=128009) added. Llama 3.1 8B 4-bit: 30 tok/s on M3 Ultra.

---

## Phase 3: Training Pipeline

**Commits:** `fb0692b` through `e3fbc22`

- **LoRA fine-tuning validated end-to-end.** Load Gemma3-1B → apply LoRA (rank=8, q_proj+v_proj, 745K params) → 5 training steps with cross-entropy loss (7.15→6.31) → save adapter (2.9 MB safetensors) → reload and verify weights match. Uses ValueAndGrad + AdamW.
- **Gradient checkpointing validated.** `Checkpoint()` wraps forward pass to recompute activations during backward. Produces correct gradients (loss 7.15→7.08 in 3 steps, matching non-checkpointed initial loss).
- **Mixed precision training validated.** `LoRAConfig.DType` selects training dtype for A/B matrices. BFloat16: loss 7.15→6.29 in 5 steps, matches Float32 accuracy with half parameter memory. MLX auto-promotes for cross-dtype operations.

---

## Phase 4: Backend Abstraction

**Commits:** `1cf5178` through `bff97cc`  
**Design documents:** `docs/plans/2026-02-19-backend-abstraction-design.md`, `docs/plans/2026-02-19-backend-abstraction-plan.md`

This phase was a full architectural restructure. All CGO code was moved to `internal/metal/`. The root package became a clean interface layer. Completed 19 February 2026.

- **All CGO moved to `internal/metal/`.** 19 source files, 10 test files.
- **Public API:** `TextModel`, `Backend`, functional options — all from go-inference, not go-mlx.
- **`context.Context` on `Generate()`.** Checks `ctx.Done()` in the decode loop.
- **`Err() error` on `TextModel`.** Distinguishes normal stop (EOS, max tokens) from errors (OOM, context cancelled).
- **`Chat()` on `TextModel`.** Model owns its chat template.
- **Backend registration.** `register_metal.go` auto-registers via build-tagged `init()`.
- **Integration tests.** 7 tests for public API (backend registration, options, `LoadModel` paths).
- **go-inference migration.** All types (`TextModel`, `Backend`, `Token`, `Message`, `GenerateConfig`, `LoadConfig`, options) moved to `forge.lthn.ai/core/go-inference`. Import `_ "forge.lthn.ai/core/go-mlx"` to register the `"metal"` backend. Completed commit `bff97cc`.
- **Error handling audit** (`ff01175`): `checkError()` replaced with `lastError()` (reads and clears C-level error string). `Eval()` and `EvalAsync()` added as error-returning variants. Generate loop propagates errors. `LoadAllSafetensors` returns `(map, error)`. Model loaders check `lastError()` after safetensors load.
- **Deterministic `Close()`** (`f2ca7fe`): Walks full model tree and explicitly frees all weight arrays. Handles tied output weights (skips double-free), nil safety, idempotent close. 8 new tests in `close_test.go`.
- **Non-contiguous array fix** (`df0b300`): `ensureContiguous()` added. `Floats()`, `DataInt32()`, `Ints()` now call it automatically. `mlx_contiguous` and `_mlx_array_is_row_contiguous` bound from mlx-c.
- **TopP and MinP sampling implemented** (`df0b300`): Previously stubs passing logits through unchanged. Now fully implemented using cumsum, argsort, and masked scattering.
- **Virgil code review applied** (`fb0692b` through `443347a`): 12 items across critical/important/minor categories including thread-safe error handler (atomic), macOS deployment target corrected (13.3), `LoadOption` propagation, KV cache leak documented, repeat penalty implemented, stream caching, BPE merge algorithm, `CompileShapeless` dead code removed, naming cleanup.
- **29 benchmarks baselined on M3 Ultra** (`ff01175`).
- **4 new error handling tests** in `error_test.go`.
- **148 tests total in `internal/metal/`; 11 root integration tests** (159 total).

---

## Phase 5: Ecosystem Integration

**Commits:** `5644857` through `757a241`

- **Batch inference API** (`5644857`, design doc `docs/plans/2026-02-19-batch-inference-design.md`): `Classify` (prefill-only, 152 prompts/s on M3 Ultra) and `BatchGenerate` (autoregressive) implemented. `ForwardMasked` added to `InternalModel` interface, threaded through Gemma3 and Qwen3 decoders. Mask: `[N, 1, L, L]` combining causal + padding. `ClassifyResult`, `BatchResult`, `WithLogits` added to go-inference.
- **Inference metrics** (`a44e9f5`): `GenerateMetrics` type in go-inference with `Metrics()` on `TextModel`. Captures prefill/decode timing, token counts, throughput (tok/s), peak and active GPU memory. Instrumented `Generate`, `Classify`, `BatchGenerate`. Gemma3-1B 4-bit on M3 Ultra: prefill 246 tok/s, decode 82 tok/s, peak 6.2 GB.
- **Model quantisation awareness** (`ceb966b`): `ModelInfo` type in go-inference with `Info()` on `TextModel`. Exposes architecture, vocab size, layer count, hidden dimension, quantisation bits and group size.
- **Embed-friendly model loading** (`dd49b4a`): `Discover(baseDir)` in go-inference scans for model directories (config.json + *.safetensors). Returns `DiscoveredModel` with path, architecture, quantisation, file count.
- **Package documentation expanded** (`d7c8f17`): godoc examples for Generate, Chat, Classify, BatchGenerate, Metrics, ModelInfo, Discover, memory controls.
- **mlxlm subprocess backend** (`757a241`): Python subprocess wrapper implementing `inference.Backend`. Communicates via JSON Lines. Auto-registers as `"mlx_lm"` via `init()`. Build tag `nomlxlm` to opt out. Full test suite using mock bridge script (`testdata/mock_bridge.py`), no `mlx-lm` install required for tests.
- **`Array.Iter()`** (`f2ca7fe`): `iter.Seq[float32]` implemented, handles non-contiguous arrays, supports early break.

**Total tests after Phase 5:** 176+ in `internal/metal/`, 11 root integration tests, 10 mlxlm tests.

---

## Known Limitations

### Per-Step Intermediate Array Accumulation

The generate loop creates approximately 500 intermediate arrays per forward pass. These rely on GC finalisers for C-side deallocation via `mlx_array_free`. Under sustained high-throughput inference, GC may not keep pace, causing Metal memory to grow until a collection cycle frees the finalisers.

Go 1.26's Green Tea GC partially mitigates this (10–40% less GC overhead, more timely finaliser execution). Explicit per-step freeing of intermediate arrays (`freeIntermediates`) was discussed but not implemented; the generate loop calls `ClearCache()` per step for prompt cache reclaim but does not explicitly free computation intermediates.

### KV Cache Per-Turn Allocation

Each call to `Generate()` allocates a fresh set of KV caches. Cross-turn cache reuse (for multi-turn chat without re-encoding the full history) is not implemented. Call `ClearCache()` between turns to reclaim Metal memory promptly.

### SentencePiece BPE Merges

The `merges` field is parsed and the merge rank table is built during tokeniser load. For Gemma 3's SentencePiece tokeniser, the vocabulary contains pre-merged token strings, so character-level splitting followed by BPE merge application is the correct approach and is implemented. The correctness boundary: if a segment contains characters not present individually in the vocabulary, they will be silently dropped. This has not caused issues with Gemma3-1B or Gemma3-4B in practice.

### mlxlm Backend Limitations

The Python subprocess backend (`mlxlm`) does not support `Classify`, `BatchGenerate`, or inference metrics. It requires Python 3 and `mlx-lm` installed in the active Python environment. There is no way to target a specific virtual environment without setting `PATH` before import.

### macOS Version Minimum

The CMake build sets `CMAKE_OSX_DEPLOYMENT_TARGET=13.3`, which is MLX's stated minimum. Testing has been performed on macOS 26.2 (Tahoe beta). Behaviour on macOS 13.x or 14.x has not been validated.

---

## Future Considerations

The following items were identified during development but deferred:

- **Per-step intermediate freeing.** Binding `mlx_array_free` directly and calling it on known-dead intermediates at the end of each decode step would reduce peak Metal memory and GC pressure. Requires careful bookkeeping to avoid double-free.
- **Cross-turn KV cache reuse.** The `Cache` interface already supports `Reset()` and `State()`. A generation session object that preserves caches across `Chat` turns would enable prefix caching without re-encoding.
- **Go 1.26 `testing.ArtifactDir()`** for storing benchmark results alongside test runs.
- **go-ml backend migration.** `go-ai/ml/backend_mlx.go` (253 LOC) should be updated to use `inference.LoadModel()` and `m.Generate()` rather than direct tensor manipulation, reducing it to approximately 60 LOC.
- **mlxlm virtual environment support.** Accept a Python interpreter path in `LoadOption` to target a specific venv.
