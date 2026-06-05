<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx — GOAL benchmark Gemma4 + PRODUCT ready state

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

Current baseline discipline:

- Use the existing `chapter-profile` book/chapter creation bench for the main
  Gemma4 loop. Do not substitute synthetic status-note prompts for production
  claims.
- Optimise sustained tokens/sec by reducing allocs/op and bytes/op first. Do
  not stop work on small tok/s variance when allocs or bytes move in the right
  direction.
- Do not hide useful report features to make bytes look better. Keep output and
  diagnostics visible unless the command already explicitly asks otherwise.
- Bench one model at a time; broad sweeps are too noisy and can overpressure
  the MLX allocator.

Current 6-bit pack inventory (downloaded 2026-06-05):

| Pack | Local snapshot | Role |
| --- | --- | --- |
| E2B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b` | coder baseline |
| E4B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-6bit/snapshots/d786394b6a0cfb1cebb74bac11d81fcb1b3ce8c8` | coder baseline |
| 12B Unified q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-6bit/snapshots/f0d6f5d34239a612f695362750044905e6dd072c` | unified validation |
| 31B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-6bit/snapshots/938d4fb4ebff2df7f6c8200977cf82a06d20f5b9` | mid/large baseline |
| 26B A4B MoE q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-6bit/snapshots/5f81a7a6f29e280f4bd5a4ce79d07d7a67fb867b` | MoE baseline |

Current go-mlx driver-profile baselines (pre-6bit-family chapter-profile pass):

| Pack | Quant | Report | Decode tok/s | Active+cache bytes | Note |
| --- | --- | --- | ---: | ---: | --- |
| E2B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-profile-256x1.json` | 76.07 | 4,213,461,894 | usable baseline; rerun with `chapter-profile` |
| E4B | 4-bit | `/private/tmp/go-mlx-self/reports/gemma4-e4b-q4-profile-256x1.json` | 58.15 | 4,835,022,478 | superseded by downloaded E4B q6 |
| 12B Unified | 6-bit non-it mirror | `/private/tmp/go-mlx-self/reports/gemma4-12b-q6-profile-256x1.json` | 37.44 | 18,700,650,948 | functional validation; rerun with 12B-it q6 |
| 31B | 4-bit | `/private/tmp/go-mlx-self/reports/gemma4-31b-q4-profile-256x1.json` | 29.07 | 24,485,128,808 | superseded by downloaded 31B q6 |

Invalidated chapter-profile runs:

- Any report with `-chapter-max-tokens 256` is a harness smoke only, not a
  Gemma4 baseline. It proves the CLI can stream 256 tokens, not that the model
  or runtime completes a chapter.

Current go-mlx chapter-profile baselines:

| Pack | Quant | Report | Generated tokens | Decode tok/s | Active+cache bytes | Peak bytes | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| E2B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-uncapped-native-1.json` | 1,499 | 68.76 | 9,400,629,338 | 4,028,025,290 | pre-cleanup report shows internal `chapter_max_tokens:32768`; no explicit CLI cap; natural stop before budget; rerun after Metal load recovers so report shows request `0` |
| E4B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-e4b-q6-chapter-profile-uncapped-native-1.json` | 1,495 | 47.09 | 12,927,586,884 | 6,411,030,952 | pre-cleanup report shows internal `chapter_max_tokens:32768`; no explicit CLI cap; natural stop before budget; rerun after Metal load recovers so report shows request `0` |
| 12B Unified | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-word-safe-1.json` | 2,019 | 33.04 | 19,239,393,780 | 12,757,909,568 | pre-cleanup report shows internal `chapter_max_tokens:32768`; no explicit CLI cap; completed after repeated-word safety was added; rerun after Metal load recovers so report shows request `0` |

Current failed chapter-profile probes:

| Pack | Quant | Report | Generated tokens | Decode tok/s | Active+cache bytes | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 12B Unified | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-1.json` | 16,000 | 30.45 | 19,698,793,748 | manually aborted after visible output collapsed into repeated `order-` / `0` runs; not a baseline |
| 12B Unified | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-loop-safe-1.json` | 7,390 | 31.95 | 19,417,208,104 | manually aborted after visible output collapsed into repeated `neighbors`; token-id loop safety alone was insufficient |
| 31B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-31b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 96 | 13.52 | 32,173,312,424 | stopped by repeated-word safety on `same`; not a baseline |
| 26B A4B MoE | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-26b-a4b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 841 | 38.53 | 27,781,603,808 | stopped by repeated-word safety on `termination`; not a baseline |

Next benchmark pass: `chapter-profile`, one 6-bit pack at a time, no synthetic
max-token cap. Record decode tok/s, `go_total_alloc_delta_bytes`,
`go_mallocs_delta`, `go_bytes_per_generated_token`,
`go_allocs_per_generated_token`, active+cache bytes, peak resident memory,
command, output sample, and stderr in `docs/runtime/`.

Runtime artefact: `docs/runtime/2026-06-05-gemma4-6bit-chapter-profile.md`.
Post-cleanup E2B rerun failed before model load because current discovery reports
`load_available=false` despite seeing `Apple M3 Ultra`; the failed report is
not a baseline but confirms the CLI report keeps `chapter_max_tokens:0`.

Cleanup progress:

- `driver-profile` runtime-gate CLI coverage now uses one table-driven test for
  the remaining benchmark gate flags instead of one `_Good` test per flag.
  Multi-flag fast-lane/fixed-cache tests remain only where they cover
  interactions.
- `serve` no longer applies `DefaultEngineFeatures()` at process boot. The
  authoritative fast-path selection is the loaded model declaration via
  `metal.EngineFeaturesFor`, applied inside `metal.LoadAndInit`, so lazy load
  and `/v1/admin/serve/reload` use the same model-owned path.
- `KV_CACHE_DTYPE` is no longer a runtime/env gate. Long-context Gemma4 fast
  lane applies `kv_cache_storage_dtype` through typed load/profile settings,
  and native cache allocation / prompt-cache restore read the model config.
- `GENERATION_CLEAR_CACHE_INTERVAL` is no longer a runtime/env gate. Clear-cache
  interval is a typed generate option, exposed to `driver-profile` as
  `-generation-clear-cache-interval`, and native decode uses the request config.
- `PAGED_KV_PAGE_SIZE` is no longer a runtime/env gate. Paged cache constructors
  are pure again, model-created paged caches read typed load config, and
  `driver-profile` / `chapter-profile` expose `-paged-kv-page-size`.
- `FIXED_GEMMA4_CACHE_SIZE` is no longer a runtime/env gate. Default fixed
  Gemma4 caches derive from request/context shape; the diagnostic CLI override
  now flows through typed load config and reports as `fixed_gemma4_cache_size`.
- `ProductionLaneLongContextPrefillChunkSize=512` and
  `ProductionLaneLongContextPromptChunkBytes=4096` were removed instead of
  promoted. Fast Gemma4 lane defaults no longer inject unmeasured chunk sizes;
  callers must opt in with explicit diagnostic flags until the optimum is
  measured.
- `ZERO_COPY_PAGED_RESTORE` is no longer a runtime gate. Streamed paged KV block
  restore appends page arrays directly and the legacy coalescing opt-out path
  was deleted.
- `GENERATION_CLEAR_CACHE` is no longer a runtime gate. It is now an explicit
  per-request generate option with the existing typed interval field.
- `LAST_LOGITS_PREFILL` is no longer a runtime/env gate. Models that implement
  `LastTokenLogitsModel` use the last-token prefill path automatically for long
  prompts once the built-in threshold is reached.
- `NATIVE_GELU_GATE_MUL` and `NATIVE_MLP_GELU` are direct package-init reads in
  `transformer.go`; the native MLP GELU hot path no longer calls `core.Env`.
- `NATIVE_GEMMA4_MODEL_GREEDY` was killed after an E2B q6 `driver-profile`
  off/on check showed parity but no decode win: off 71.130 tok/s, on 71.101
  tok/s, identical output token hash
  `18ce8de9f6f972df6c916b362591ea6765a740fff258b4ffc25ee192a8c3dd87`.
  The runtime gate, CLI flag, Gemma4 branch, native wrapper, C++ bridge, and
  branch-only tests were removed.
- `PAGED_KV_PREALLOC` is no longer a runtime/env gate. An E2B q6
  `driver-profile` off/on check showed parity and lower MLX active+cache
  residency, but no decode win and worse Go allocation counts: off 71.416
  tok/s at 5,576,000,330 active+cache bytes, on 70.433 tok/s at
  4,308,684,758 active+cache bytes, identical output token hash
  `18ce8de9f6f972df6c916b362591ea6765a740fff258b4ffc25ee192a8c3dd87`.
  It is now an explicit typed memory-mode load option
  `PagedKVPrealloc`, exposed through `driver-profile` and `chapter-profile`
  as `-paged-kv-prealloc`; it is not a default speed path.
- `FIXED_WIDE_SDPA_ATTENTION`, `FIXED_WIDE_MATMUL_ATTENTION`, and
  `FIXED_ROW_CACHE_UPDATE` no longer read process env in the Go/native
  attention paths. They use typed in-process diagnostics set by
  `metal.SetFixedAttentionDiagnostics`; `driver-profile` exposes
  `-fixed-wide-sdpa-attention`, `-fixed-wide-matmul-attention`, and
  `-fixed-row-cache-update`, then restores the typed state after the run.

Follow: RFC-CORE-008-AGENT-EXPERIENCE.md AX-11 especially.

See changes from start to end, dont hedge, gate, stage, or otherwise dodge   around delivering usable code.


# GOALS

- production ready Gemma-4
- LoRA + SSD training for Gemma-4 - no python
- MTP -assistant model support for all gemma-4 models

---

## A. uncomplete work, to finish.

The accepted 7 (`DirectGreedyToken`, `NativeMLPMatVec`, `NativeLinearMatVec`,
`NativeQ6BitstreamMatVec`, `NativeAttentionOMatVec`, `GenerationStream`,
`AsyncDecodePrefetch`) already live in `metal.EngineFeatures`, applied by
`metal.LoadAndInit` from the loaded model declaration. These 34 are still gated,
default-off, exercised only by the benchmark.
Per round: bench off-vs-on → **win → fold into `EngineFeatures` + delete gate**;
**lose/no-diff → delete gate + kernel branch.**
**Expert / MoE (4)**
- [ ] `EXPERT_ID_MATVEC` · `EXPERT_ID_FUSED_ACTIVATION` · `EXPERT_ID_UNROLLED_Q4` · `SORTED_EXPERT_PREFILL`
**Paged attention / cache (3)**
- [ ] `PAGED_DECODE_FAST_CONCAT` · `NATIVE_PAGED_ATTENTION`
- [x] `PAGED_KV_PREALLOC` → typed memory-mode load option; runtime gate removed; not default
**GELU / MLP (2)** — direct-read init-vars, no atomic (`transformer.go`)
- [x] `NATIVE_GELU_GATE_MUL` · `NATIVE_MLP_GELU`
**Gemma4 native layer / FFN / router (6)**
- [ ] `NATIVE_GEMMA4_FFN_RESIDUAL` · `…_ROUTER_MATVEC` · `…_ROUTER_TOPK` · `…_RESIDUAL_NORM` · `…_LAYER` · `…_MOE_LAYER`
**Fixed-owner attention (2)**
- [ ] `NATIVE_GEMMA4_FIXED_OWNER_ATTENTION` · `…_FIXED_OWNER_ATTENTION_RESIDUAL`
**Compiled (2)** — `COMPILED_GEMMA4_PER_LAYER_INPUTS` is a direct-read init-var
- [ ] `COMPILED_GEMMA4_LAYER` · `COMPILED_GEMMA4_PER_LAYER_INPUTS`
**Fixed cache / mask / sliding (4)** — diagnostic-only; ignore ambient env
- [ ] `FIXED_GEMMA4_CACHE` · `FIXED_GEMMA4_SLIDING_CACHE_BOUND` · `FIXED_GEMMA4_SHARED_MASK` · `NATIVE_FIXED_SLIDING_ATTENTION`
**Wide / row attention (3)** — diagnostic-only; no process env
- [x] `FIXED_WIDE_SDPA_ATTENTION` · `FIXED_WIDE_MATMUL_ATTENTION` · `FIXED_ROW_CACHE_UPDATE` → typed `SetFixedAttentionDiagnostics`; driver-profile flags exposed
**Misc fast paths (4)**
- [x] `NATIVE_GEMMA4_MODEL_GREEDY` → parity, no decode win; gate and branch deleted
- [x] `LAST_LOGITS_PREFILL` → automatic `LastTokenLogitsModel` capability path; env/report gate removed
- [x] `GENERATION_CLEAR_CACHE` → typed per-request generate option; runtime gate removed
- [x] `ZERO_COPY_PAGED_RESTORE` → always-on streamed paged KV block restore; gate and legacy coalescing path removed
**Value params, not on/off gates (5)** — take a value; move to model config /
`EngineFeatures`, do **not** "prove or kill":
- [x] `FIXED_GEMMA4_CACHE_SIZE` → derive from request/context by default; typed diagnostic override; env retired
- [x] `GENERATION_CLEAR_CACHE_INTERVAL` → typed generate/config default; env retired
- [x] `KV_CACHE_DTYPE` → typed load/profile field; env retired
- [x] `PAGED_KV_PAGE_SIZE` → typed load/config default; env retired
- [x] **`SlidingWindow`** `512` / `1024` → native Gemma4 `sliding_window`
  metadata is authoritative; removed root/Metal load-time override and clamp
  path.
- [ ] **`PrefillChunkSize`** `4096` / `2048` / `1024` / `512` scattered → measure the optimum, one config value
- [ ] **`PromptChunkBytes`** `4096` → measure, don't guess
- [x] `ProductionLaneLongContextPrefillChunkSize = 512` / `…PromptChunkBytes = 4096` → removed; fast lane no longer injects unmeasured chunk defaults
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
