<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx — GOAL Gemma-4 Support + LoRA

Production Apple Silicon runtime for agentic + coder workflows: native Go/Metal
model loading, generation, adapter training, and evaluation — **no Python in the
production path**. Floor: macOS Tahoe 26.0+ on Apple Silicon (Metal 4).

## Active Goals

1. **Production-ready Gemma-4 family support.** All five Gemma-4 packs below
   should load, generate, stream, retain state, benchmark, and fail cleanly when
   the local runtime cannot support the requested shape.
2. **Gemma-4 LoRA support, no Python.** LoRA target resolution,
   adapter attach/load/save, SFT smoke, eval, fuse, and clear failure modes
   should work through go-mlx APIs and CLI flows for Gemma-4 text and MoE
   shapes.

Supporting work is allowed only when it moves one of those two goals forward:
SPOR cleanup, MTP assistant support, SSD, performance work, and dead-code
removal should all feed back into Gemma-4 family quality or the Gemma-4 LoRA
loop.

## Working Rules

- **No Python** in production runtime, training, LoRA, SSD, eval, or benchmark
  paths. Python is acceptable only for unavoidable external comparison tooling,
  and not for go-mlx correctness.
- **No artificial output caps** in production benchmarks. Do not add default max
  tokens to make a run finish. A benchmark may stop on EOS, end marker, or a
  real safety stop.
- **No new `GO_MLX_ENABLE_*` gates.** A proven runtime feature becomes typed
  config, model-declared `metal.EngineFeatures`, or always-on. A loss is
  deleted with its branch and dead tests.
- **No hidden env feature paths.** CLI/profile options must flow through typed
  Go config/state, not process env mutation.
- **Use go-mlx only** for verification. Do not substitute other programs for
  tests against this codebase.
- **SPOR means Single Point of Responsibility.** Gemma-4 prompt/chat formatting,
  adapter target naming, and model metadata should each have one shared owner
  used by serving, training, eval, benchmark, and adapter code.
- **No fake green tests.** Tests must prove the live contract they name, cover
  real failure modes, and be deleted when the code path they exercised is
  deleted.
- **Bench one model at a time.** Broad sweeps are noisy and overpressure MLX
  allocation.
- **Use `chapter-profile` for production claims.** `driver-profile` remains
  useful for narrow off/on diagnostics, but book/chapter creation is the main
  Gemma-4 quality and sustained throughput loop.
- **Remove dead code as it is discovered.** Do not keep tests for deleted paths,
  parked branches, or fake compatibility surfaces.

## Gemma-4 Pack Inventory

Downloaded 2026-06-05:

| Pack | Local snapshot | Target status |
| --- | --- | --- |
| E2B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b` | primary coder baseline |
| E4B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-6bit/snapshots/d786394b6a0cfb1cebb74bac11d81fcb1b3ce8c8` | primary coder baseline |
| 12B Unified q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-6bit/snapshots/f0d6f5d34239a612f695362750044905e6dd072c` | unified validation |
| 31B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-6bit/snapshots/938d4fb4ebff2df7f6c8200977cf82a06d20f5b9` | mid/large validation |
| 26B A4B MoE q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-6bit/snapshots/5f81a7a6f29e280f4bd5a4ce79d07d7a67fb867b` | MoE validation |

## Current Baselines

`chapter-profile` baselines are the production reference. Older `driver-profile`
numbers are retained only as quick diagnostics.

| Pack | Quant | Report | Generated tokens | Decode tok/s | Active+cache bytes | Peak bytes | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| E2B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-uncapped-native-1.json` | 1,499 | 68.76 | 9,400,629,338 | 4,028,025,290 | pre-cleanup report shows internal `chapter_max_tokens:32768`; natural stop before budget |
| E4B | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-e4b-q6-chapter-profile-uncapped-native-1.json` | 1,495 | 47.09 | 12,927,586,884 | 6,411,030,952 | pre-cleanup report shows internal `chapter_max_tokens:32768`; natural stop before budget |
| 12B Unified | 6-bit | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-word-safe-1.json` | 2,019 | 33.04 | 19,239,393,780 | 12,757,909,568 | completed after repeated-word safety was added |

Failed but useful probes:

| Pack | Report | Generated tokens | Decode tok/s | Outcome |
| --- | --- | ---: | ---: | --- |
| 12B Unified q6 | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-1.json` | 16,000 | 30.45 | manually aborted after repeated `order-` / `0` output |
| 12B Unified q6 | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-loop-safe-1.json` | 7,390 | 31.95 | manually aborted after repeated `neighbors`; token-id safety alone was insufficient |
| 31B q6 | `/private/tmp/go-mlx-self/reports/gemma4-31b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 96 | 13.52 | stopped by repeated visible word `same`; load/generate worked, quality did not |
| 26B A4B MoE q6 | `/private/tmp/go-mlx-self/reports/gemma4-26b-a4b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 841 | 38.53 | stopped by repeated visible word `termination`; load/generate worked, quality did not |

Runtime artefact: `docs/runtime/2026-06-05-gemma4-6bit-chapter-profile.md`.
Fresh accepted reports should show `chapter_max_tokens: 0` when the command is
run without `-chapter-max-tokens`.

## Workstream A — Gemma-4 Family Support

- [ ] E2B q6: rerun uncapped `chapter-profile` with current code and record
  tok/s, allocs/token, bytes/token, active+cache, resident peak, command, stderr,
  and output sample.
- [ ] E4B q6: same accepted `chapter-profile` record.
- [ ] 12B Unified q6: same accepted `chapter-profile` record, preserving the
  1024 local sliding window and global owner-layer shape.
- [ ] 31B q6: make the generation quality failure actionable; distinguish model
  quality/safety failure from runtime/cache failure.
- [ ] 26B A4B MoE q6: make the MoE generation quality failure actionable;
  confirm router/shared-KV behaviour and cache layout.
- [ ] Confirm Gemma-4 native metadata is authoritative for context length,
  sliding window, shared KV owners, local/global attention layout, stop tokens,
  and tokenizer chat template.
- [ ] Keep 256K context support uncut. Do not reintroduce 8K/32K defaults as
  hidden runtime limits.
- [ ] Keep text, 12B Unified, and MoE model names routed through the Gemma-4
  loader without standalone assistant-model confusion.
- [ ] MTP assistant path: target/assistant pair loading, draft-token policy,
  target-only fallback, prompt-cache interaction, and report metrics.

## Workstream B — Gemma-4 LoRA + SPOR

- [x] Confirm Gemma-4 LoRA target resolution and attach for standard attention
  targets: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`,
  `self_attn.o_proj`, plus suffix adapter keys `q_proj`, `k_proj`, `v_proj`,
  `o_proj`.
- [x] Confirm extended Gemma-4 targets are explicit and safe:
  `router.proj`, `per_layer_input_gate`, and `per_layer_projection`.
- [x] SPOR: route Gemma-4 serving prompts, dataset/training prompts, eval
  prompts, and benchmark prompts through the shared chat formatter; remove
  duplicate prompt renderers or reduce them to thin delegations.
- [x] SPOR: keep Gemma-4 adapter target naming in one resolver used by
  attach/load/train/fuse paths instead of per-flow target maps.
- [x] Load PEFT-style adapter config + safetensors into Gemma-4 through
  go-mlx APIs and `WithAdapterPath`, including adapter identity in `ModelInfo`
  and profile reports. PEFT metadata parsing, native safetensors injection,
  public `WithAdapterPath` identity, report `adapter_path`, and a real
  Gemma-4 E2B q6 reload/generate proof are covered.
- [x] Train a small Gemma-4 LoRA SFT smoke with Go-native training only; save an
  adapter that reloads and changes generation/eval output.
- [ ] Wire SSD training for Gemma-4 using existing distillation APIs; expose the
  sampled teacher/student generate configs without Python.
- [x] Eval base vs adapter on a JSONL dataset with the existing eval harness;
  record loss/perplexity and adapter identity.
- [x] Fuse a Gemma-4 LoRA adapter into a model pack and verify reload/generate.
- [x] Make LoRA failure modes clear: unsupported target, shape mismatch, missing
  adapter config, missing safetensors, unsupported quantized target.
- [ ] Keep adapter code reusable across E2B/E4B/12B/31B/26B MoE rather than
  special-casing one checkpoint.

Progress 2026-06-05:

- Gemma-4 `ApplyLoRA` now canonicalises suffix and full-path target names through
  the model resolver before attaching adapters, so attach uses the same target
  naming surface as adapter load/save metadata.
- Gemma-4 adapter target canonicalisation now has a shared metal helper used by
  config normalisation and model attach; PEFT MLP suffix aliases
  `gate_proj`/`up_proj`/`down_proj` stay valid without extended-target opt-in
  and attach as `mlp.*` paths.
- Gemma-4 SFT now normalises training LoRA targets through the same shared metal
  policy as adapter attach/load; loaded Gemma-4 training defaults include
  `o_proj`, while generic SFT defaults remain unchanged.
- Resolver failure modes now return nil for nil models, negative/out-of-range
  layers, missing layer parts, and unknown target paths instead of panicking.
- SPOR prompt coverage now pins `dataset.MessagesToSample` Gemma-4 training
  prompts byte-for-byte against `chat.Format`; serving already delegates through
  `formatGemma4Chat`.
- SPOR benchmark prompt coverage now routes Gemma-4 `chapter-profile` and
  `state-ramp-profile` initial/continuation prompts through `chat.Format`,
  including the 26B/31B large-variant empty thought-channel suppressor derived
  from native head-count metadata.
- SPOR inference adapter chat-template coverage now derives Gemma-4 large
  variant formatting from loaded model metadata before delegating to
  `chat.Format`, so shared-inference callers do not lose the 26B/31B
  thought-channel suppressor.
- SFT eval prompts now render Gemma-4 prompt strings through the same shared
  `chat.Format` path before generation while preserving the original prompt
  identity in `SFTEvalResult`.
- Admin SFT JSONL loading now derives its chat-template config from loaded
  model metadata, so Gemma-4 message-shaped training rows use the same
  large-variant formatter as serving and eval.
- Native adapter load now accepts PEFT aliases (`r`, `lora_alpha`, `scale`,
  `target_modules`, `target_keys`) as well as mlx-lm `rank`, `alpha`, and
  `lora_layers`; loaded adapter config and attached LoRA scale preserve the
  PEFT metadata.
- Native adapter load now accepts PEFT safetensors tensor names
  `.lora_A.weight` / `.lora_B.weight`, strips common PEFT wrapper prefixes, and
  resolves Gemma-4 suffix targets such as `q_proj` into canonical
  `self_attn.q_proj` adapter layers.
- `WithAdapterPath` now has PEFT-style identity coverage in `ModelInfo` and
  metrics, and profile load settings preserve the resolved adapter path from
  loaded model info.
- Native adapter load now validates LoRA A/B tensor shapes against the resolved
  base projection before attaching anything; shape mismatches fail at load time
  with the target path named and leave the model unmodified.
- Native adapter load now rejects unsupported target paths during pre-attach
  validation; mixed valid/invalid adapters fail with the unsupported target
  named and leave already-resolved projections unmodified.
- Native adapter load failure coverage now names missing `adapter_config.json`,
  missing `.safetensors` files, unsupported target paths, LoRA shape
  mismatches, and unsupported quantized target metadata without retaining a
  partial adapter attach.
- Pack-level LoRA fusion now resolves Gemma-4 PEFT suffix targets through the
  shared adapter target policy before looking up base safetensors keys; generic
  model families keep their existing model-local suffix behaviour.
- Go-ignored parked Gemma-4 assistant scratch tests were removed; future
  assistant coverage must live in real package tests that compile in the normal
  `go test ./go/...` surface.
- Strict Metal runtime verification now runs with `MLX_METALLIB_PATH` and
  `GO_MLX_RUN_METAL_TESTS=1`: stale cache-only chunk prefill and paged block
  restore expectations were corrected, and cacheless retained-logit session
  generation no longer fails the readiness guard.
- Real Gemma-4 LoRA reload proof: `/private/tmp/go-mlx-self/gemma4_lora_smoke`
  loaded the E2B q6 snapshot, saved a rank-2 adapter to
  `/private/tmp/go-mlx-self/gemma4-e2b-lora-smoke-adapter`, reloaded with
  `WithAdapterPath`, confirmed adapter identity in `Info` and metrics, and
  generated 47 tokens with `model=gemma4_text` and targets
  `[self_attn.o_proj self_attn.q_proj self_attn.v_proj]`.
- Go-native Gemma-4 SFT smoke now runs from the checked-in Go test surface when
  `GO_MLX_RUN_METAL_TESTS=1` and the E2B q6 snapshot is present:
  `TestSFTNativeSmoke_Gemma4Q6SavesReloadableAdapter_Good` loads message-shaped
  JSONL through `DatasetConfigForModel`, trains three native LoRA steps, saves
  `adapter_config.json`, `adapter.safetensors`, and `sft_checkpoint.json`,
  reloads the saved rank-2 adapter through `WithAdapterPath`, confirms adapter
  identity in eval reports, and changed JSONL eval loss from `10.653769` to
  `3.527476` and perplexity from `42351.939379` to `34.037950` in the focused
  Metal proof run.
- The documented root fusion API is live again: `FuseLoRAIntoModelPack`
  validates the source pack through the shared model-pack inspector, calls the
  existing pack-level `lora.FuseIntoPack`, then validates the fused output pack.
  `TestFuseLoRAIntoModelPack_Gemma4SuffixTargetValidatesOutput_Good` runs with
  Metal enabled, uses PEFT-style Gemma-4 `q_proj` suffix tensors, proves the
  canonical fused key `model.layers.0.self_attn.q_proj.weight`, and verifies the
  fused tensor values. The real E2B q6 proof
  `TestFuseLoRAIntoModelPack_Gemma4Q6RealPackReloadGenerate_Good` fuses the
  saved rank-2 adapter into the local q6 snapshot, reloads the fused pack
  without a live adapter, and generates successfully.
- Gemma-4 text weight-name canonicalisation now lives in the shared metal
  package via `metal.Gemma4CanonicalWeightName`; the Gemma-4 loader delegates to
  it, and pack-level LoRA fusion builds a per-shard canonical index from it.
  Dense Gemma-4 safetensors with MLX-community wrapper keys such as
  `language_model.model.layers.*.self_attn.q_proj.weight` now fuse under the
  original source key instead of missing the base weight or writing duplicate
  canonical keys.
- Pack-level Gemma-4 fusion now handles q6 affine base targets by dequantizing
  only the fused target, adding the LoRA delta, writing that target back as
  dense, and dropping its `.scales` / `.biases` sidecars so the Gemma-4 loader
  treats it as dense while untouched q6 tensors remain quantized. The root
  `FuseLoRAIntoModelPack` proof now validates the output pack with real q6
  sidecars and the full local E2B q6 pack reload/generate proof passed with
  105 fused q/v/o projections.
- Gemma-4 fuse architecture detection now delegates to the shared
  `profile.ArchitectureID` resolver instead of carrying a local model-family
  switch. The root `FuseLoRAIntoModelPack` test now uses an official-style
  `model_type:"gemma4"` wrapper config with `Gemma4ForConditionalGeneration`,
  `text_config.model_type:"gemma4_text"`, q6 metadata, and a
  `language_model.model.*` source key, so the public API proof covers the same
  metadata and key-shape SPOR path used by real E2B/E4B/31B packs.
- Native adapter load now uses the same `profile.ArchitectureID` Gemma-4 family
  check as fuse, so suffix adapter target canonicalisation recognises official
  Gemma-4 Transformers architecture names and unified aliases without a second
  local switch. The assistant architecture remains excluded from the standalone
  Gemma-4 adapter path.
- Gemma-4 chat/SFT family detection now delegates to `profile.ArchitectureID`
  as well: official Transformers names and unified aliases select the shared
  Gemma-4 formatter for dataset rows, SFT eval prompts, and SSD's downstream
  SFT config, while the standalone assistant architecture remains excluded.

## Workstream C — Performance And Memory

- [ ] Optimise sustained decode by reducing `go_total_alloc_delta_bytes`,
  `go_mallocs_delta`, `go_bytes_per_generated_token`, and
  `go_allocs_per_generated_token`. Do not stop on small tok/s variance when
  allocation movement is clearly better.
- [ ] Measure `PrefillChunkSize` instead of guessing. Remove scattered
  `4096` / `2048` / `1024` / `512` assumptions or make one measured config
  value.
- [ ] Measure `PromptChunkBytes` instead of defaulting to `4096`.
- [ ] Recheck paged KV defaults after the accepted model-family baselines are
  current.
- [ ] Keep useful report output visible. Do not hide diagnostics to improve
  apparent memory numbers.

## Workstream D — Cleanup That Still Matters

Resolved cleanup:

- [x] `KV_CACHE_DTYPE` → typed load/profile field; env retired.
- [x] `PAGED_KV_PAGE_SIZE` → typed load/config default; env retired.
- [x] `PAGED_KV_PREALLOC` → typed memory-mode load option; runtime gate removed;
  not default.
- [x] `FIXED_GEMMA4_CACHE_SIZE` → derived by default; typed diagnostic override.
- [x] `GENERATION_CLEAR_CACHE` and interval → typed per-request generate options.
- [x] `ZERO_COPY_PAGED_RESTORE` → always-on streamed paged KV block restore.
- [x] `LAST_LOGITS_PREFILL` → automatic `LastTokenLogitsModel` capability path.
- [x] `NATIVE_GELU_GATE_MUL` / `NATIVE_MLP_GELU` → direct package-init vars.
- [x] `NATIVE_GEMMA4_MODEL_GREEDY` → deleted after E2B q6 parity/no-win bench.
- [x] `FIXED_WIDE_SDPA_ATTENTION` / `FIXED_WIDE_MATMUL_ATTENTION` /
  `FIXED_ROW_CACHE_UPDATE` → typed `SetFixedAttentionDiagnostics`; no live
  process-env selection.

Remaining cleanup backlog, only if it supports the active Gemma-4/LoRA goals:

- [ ] Expert/MoE diagnostics:
  `EXPERT_ID_MATVEC`, `EXPERT_ID_FUSED_ACTIVATION`,
  `EXPERT_ID_UNROLLED_Q4`, `SORTED_EXPERT_PREFILL`.
- [ ] Paged attention diagnostics:
  `PAGED_DECODE_FAST_CONCAT`, `NATIVE_PAGED_ATTENTION`.
- [ ] Gemma-4 native layer/router diagnostics:
  `NATIVE_GEMMA4_FFN_RESIDUAL`, `NATIVE_GEMMA4_ROUTER_MATVEC`,
  `NATIVE_GEMMA4_ROUTER_TOPK`, `NATIVE_GEMMA4_RESIDUAL_NORM`,
  `NATIVE_GEMMA4_LAYER`, `NATIVE_GEMMA4_MOE_LAYER`.
- [ ] Fixed-owner attention diagnostics:
  `NATIVE_GEMMA4_FIXED_OWNER_ATTENTION`,
  `NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL`.
- [ ] Compiled diagnostics:
  `COMPILED_GEMMA4_LAYER`, `COMPILED_GEMMA4_PER_LAYER_INPUTS`.
- [ ] Fixed cache/mask/sliding diagnostics:
  `FIXED_GEMMA4_CACHE`, `FIXED_GEMMA4_SLIDING_CACHE_BOUND`,
  `FIXED_GEMMA4_SHARED_MASK`, `NATIVE_FIXED_SLIDING_ATTENTION`.

## Verification

Before claiming a Gemma-4 or LoRA item is done:

```sh
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GO_MLX_RUN_METAL_TESTS=1 GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/... -count=1
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx
```

Production-claim artefacts must include model path+revision, quant, context
shape, command, stderr, memory method, output sample, and report path under
`docs/runtime/`.
