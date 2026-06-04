<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx — GOAL (benchmark Gemma4 + clean the slop)

Production Apple Silicon runtime for agentic + coder workflows: native Go/Metal
model loading, generation, training — **no Python in the production path**.
Floor: macOS Tahoe 26.0+ on Apple Silicon (Metal 4).

**North star:** sustained **≥100 tok/s decode** on the Gemma4 coder packs
(E2B/E4B + quantised mid-size). **The job now is to RESOLVE the slop, not
generate more.** Two hard rules:

1. **No new `GO_MLX_ENABLE_*` gates.** A proven win becomes a model-declared
   field on `metal.EngineFeatures` (`DefaultEngineFeatures()` /
   `gemma4.EngineFeatures()`) or goes always-on. A loss gets **deleted** — gate,
   kernel branch, tests. Gate count only goes **down**.
2. **No test-per-micro-step.** One `Test<Kernel>_ParityAndSpeed` per kernel,
   never a `_Good`/`_Bad`/`_RuntimeGate` triplet. Don't re-add the
   `coverageTokens` ritual (5,297 lines of it were just deleted).

Measure with `lthn-mlx driver-profile` on a real gemma4 model. **Parity** =
identical greedy token hash. **Win** = parity AND lower decode wall-time.

---

## A. The 34 experimental runtime gates — prove or kill, one per round

The accepted 7 (`DirectGreedyToken`, `NativeMLPMatVec`, `NativeLinearMatVec`,
`NativeQ6BitstreamMatVec`, `NativeAttentionOMatVec`, `GenerationStream`,
`AsyncDecodePrefetch`) already live in `metal.EngineFeatures`, applied at serve
boot. These 34 are still gated, default-off, exercised only by the benchmark.
Per round: bench off-vs-on → **win → fold into `EngineFeatures` + delete gate**;
**lose/no-diff → delete gate + kernel branch.**

**Expert / MoE (4)**
- [ ] `EXPERT_ID_MATVEC` · `EXPERT_ID_FUSED_ACTIVATION` · `EXPERT_ID_UNROLLED_Q4` · `SORTED_EXPERT_PREFILL`

**Paged attention / cache (3)**
- [ ] `PAGED_DECODE_FAST_CONCAT` · `NATIVE_PAGED_ATTENTION` · `PAGED_KV_PREALLOC`

**GELU / MLP (2)** — direct-read init-vars, no atomic (`transformer.go`)
- [ ] `NATIVE_GELU_GATE_MUL` · `NATIVE_MLP_GELU`

**Gemma4 native layer / FFN / router (6)**
- [ ] `NATIVE_GEMMA4_FFN_RESIDUAL` · `…_ROUTER_MATVEC` · `…_ROUTER_TOPK` · `…_RESIDUAL_NORM` · `…_LAYER` · `…_MOE_LAYER`

**Fixed-owner attention (2)**
- [ ] `NATIVE_GEMMA4_FIXED_OWNER_ATTENTION` · `…_FIXED_OWNER_ATTENTION_RESIDUAL`

**Compiled (2)** — `COMPILED_GEMMA4_PER_LAYER_INPUTS` is a direct-read init-var
- [ ] `COMPILED_GEMMA4_LAYER` · `COMPILED_GEMMA4_PER_LAYER_INPUTS`

**Fixed cache / mask / sliding (4)** — diagnostic-only; ignore ambient env
- [ ] `FIXED_GEMMA4_CACHE` · `FIXED_GEMMA4_SLIDING_CACHE_BOUND` · `FIXED_GEMMA4_SHARED_MASK` · `NATIVE_FIXED_SLIDING_ATTENTION`

**Wide / row attention (3)** — diagnostic-only; read directly via `core.Env`
- [ ] `FIXED_WIDE_SDPA_ATTENTION` · `FIXED_WIDE_MATMUL_ATTENTION` · `FIXED_ROW_CACHE_UPDATE`

**Misc fast paths (4)**
- [ ] `LAST_LOGITS_PREFILL` · `NATIVE_GEMMA4_MODEL_GREEDY` · `GENERATION_CLEAR_CACHE` · `ZERO_COPY_PAGED_RESTORE`

**Value params, not on/off gates (4)** — take a value; move to model config /
`EngineFeatures`, do **not** "prove or kill":
- [ ] `FIXED_GEMMA4_CACHE_SIZE` → derive from model + context
- [ ] `GENERATION_CLEAR_CACHE_INTERVAL` → config default
- [ ] `KV_CACHE_DTYPE` → already a profile field; retire the env
- [ ] `PAGED_KV_PAGE_SIZE` → config default

---

## B. The yolo'd magic numbers — tune, don't guess

Where the real Gemma4 speed lives. codex slammed these in untuned. Benchmark
the actual optimum per model + context, then source from config — not literals.

- [ ] **`SlidingWindow`** `512` / `1024` → from the model's `sliding_window` config
- [ ] **`PrefillChunkSize`** `4096` / `2048` / `1024` / `512` scattered → measure the optimum, one config value
- [ ] **`PromptChunkBytes`** `4096` → measure, don't guess
- [ ] `ProductionLaneLongContextPrefillChunkSize = 512` / `…PromptChunkBytes = 4096` → confirm these are measured optima

---

## C. The real "not implemented yet" stubs (~15)

> A `grep TODO|placeholder|for now` returns ~33 hits, but ~18 are false
> positives — chapter-prompt text telling the model *not* to write placeholders,
> `\uXXXX` unicode-escape comments, the `ModelPathPlaceholder` example field.
> The genuine gaps. Each: implement it, or if out of scope, delete the stub and
> return a clean "unsupported" instead of a "not implemented yet" lie.

**Non-gemma4 architecture loaders** (`model/pack.go`, `pkg/metal/dense_config.go`)
- [ ] qwen3_moe sparse-expert routing — `qwen3.go:76`, `dense_config.go:155`
- [ ] qwen3_6 hybrid linear-attention — `dense_config.go:157`, `pack.go:684/686`
- [ ] native embedding-encoder loading — `pack.go:688`
- [ ] native rerank-scorer loading — `pack.go:690`
- [ ] codebook/VQ-quantized model loading — `pack.go:483`
- [ ] generic "native runtime loading not implemented" fallbacks — `pack.go:675/695`, `hf/hf.go:910`

**Engine**
- [ ] TurboQuant KV cache mode — kernels planned, not implemented — `pkg/metal/backend.go:57`
- [ ] sparse-merge hook — reserved, not implemented — `merge/merge.go:195`

**Admin surface**
- [ ] `/v1/admin/models/...` stub — `adminNotImplementedHandler`, `cmd/mlx/admin.go:625`

---

## D. Cladius is handling — do NOT redo

- **#55 EngineFeatures** — model-declared fast-path: `metal.EngineFeatures` +
  `DefaultEngineFeatures()` + gemma4 declares + `backend.LoadAndInit` applies +
  serve applies at boot; 11 init-var gates hollowed onto the runtime atomic.
- **coverageTokens ritual** — deleted (5,297 lines, 91 files).
- **Open (Cladius):** split `cmd/mlx/main.go` (9,378 lines) per-command; collapse
  the `_Good`/`_Bad`/`_RuntimeGate` test triplets; delete
  `pkg/metal/model/gemma4/_parked_assistant_tests/`.

---

## Parked feature goals (after the cleanup)

- Gemma4-12B "Unified" pack — hybrid attention (1024 local sliding + global,
  p-RoPE), encoder-free multimodal (linear projection, not encoder subgraphs).
- Finish non-gemma4 generation paths (shared MoE / Qwen3.6 hybrid / DeepSeek MLA
  / MiniMax sparse) — compose `EngineFeatures`, never a new monolith.
- Native hierarchical-memory pretraining (`apple/ml-memory-pretraining`, no Python).

## Verification

```sh
GOWORK=…/go.work go build -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/pkg/metal/...
go test ./go/...                                   # green
go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /tmp/lthn-mlx ./go/cmd/mlx
```

Production-claim artefacts (model path+revision, quant, context shape, command,
stderr, memory method, output sample) → `docs/runtime/`.
