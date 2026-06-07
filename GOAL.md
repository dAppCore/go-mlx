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
- [x] Wire SSD training for Gemma-4 using existing distillation APIs; expose the
  sampled teacher/student generate configs without Python.
- [x] Eval base vs adapter on a JSONL dataset with the existing eval harness;
  record loss/perplexity and adapter identity.
- [x] Fuse a Gemma-4 LoRA adapter into a model pack and verify reload/generate.
- [x] Make LoRA failure modes clear: unsupported target, shape mismatch, missing
  adapter config, missing safetensors, unsupported quantized target.
- [x] Keep adapter code reusable across E2B/E4B/12B/31B/26B MoE rather than
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
- The inference-facing training adapter no longer pre-fills generic q/v LoRA
  defaults before native model attach. Empty `inference.LoRAConfig` now reaches
  the native model as empty so Gemma-4 can apply its shared q/v/o default, while
  `inference.DefaultLoRAConfig()` still forwards explicit q/v targets for the
  generic interface contract.
- The root `NewLoRA(model, nil)` wrapper now follows the same no-override
  contract as the inference adapter path, so Gemma-4 model normalisation owns
  nil/default target selection across both public LoRA entry points. Passing
  `DefaultLoRAConfig()` explicitly still forwards the generic q/v default.
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
- Adapter config parsing is now SPOR too:
  `internal/loraadapter.ParseConfig` owns `rank`/`r`,
  `alpha`/`lora_alpha`/`scale`, and target-field precedence
  (`target_keys`, then `target_modules`, then `lora_layers`) for public
  adapter inspection and native Metal adapter load. Public inspection preserves
  missing rank/alpha/scale metadata so fusion validation can reject incomplete
  adapters; `NormalizeForNativeLoad` applies mlx-lm-style rank 8 / alpha 16 /
  scale 2 defaults only at the native load boundary. The old public helper
  benches for deleted private functions now benchmark the live shared parser
  and normaliser instead.
- Root adapter identity now merges native-normalised adapter metadata after
  `WithAdapterPath` and `Model.LoadLoRA`: public inspection keeps stable
  path/hash and missing-field visibility, while loaded rank/alpha/scale/targets
  fill the reported `ModelInfo`, metrics, and `Adapter()` identity.
- Pack-level fusion now has explicit rank-only adapter coverage: missing rank
  still rejects, while adapters with rank and no alpha/scale use the native
  alpha/scale default before provenance is written. The LoRA fuse guide now
  matches that live contract instead of incorrectly requiring `scale`.
- Native adapter load now accepts PEFT safetensors tensor names
  `.lora_A.weight` / `.lora_B.weight`, strips common PEFT wrapper prefixes, and
  resolves Gemma-4 suffix targets such as `q_proj` into canonical
  `self_attn.q_proj` adapter layers.
- Native adapter load now proves that PEFT `q_proj` suffix adapters resolve
  through the shared Gemma-4 family policy for `gemma4`, `gemma4_text`,
  `gemma4_unified`, `gemma4_unified_text`, `Gemma4ForConditionalGeneration`,
  `Gemma4UnifiedForConditionalGeneration`, `Gemma4ForCausalLM`, and
  `Gemma4TextForCausalLM`; the same safetensors load path also attaches
  MoE/PLE-style `router.proj`, `per_layer_input_gate`, and
  `per_layer_projection` adapters without an E2B-only branch.
- Gemma-4 training attach coverage now proves the same extended-target boundary
  from the other side: `ApplyLoRA` attaches standard/MLP targets, only attaches
  `router.proj`, `per_layer_input_gate`, and `per_layer_projection` when
  `AllowGemma4ExtendedTargets` is set, and keeps those projections unmodified
  otherwise.
- Gemma-4 LoRA normalisation now also proves the RFC `TargetLayers` alias goes
  through the same safe-target policy: MLP aliases stay allowed without opt-in,
  while router and per-layer embedding targets are filtered unless
  `AllowGemma4ExtendedTargets` is set. The public training docs and Metal
  config comment now describe router/PLE opt-in instead of the stale
  "non q/v/o" wording.
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
- Gemma-4 assistant speculative dispatch now goes through the optional
  `nativeGemma4AssistantGenerator` capability before falling back to the real
  `*metal.Model` assistant path, so fake native models can exercise the
  package-level MTP contract. The formerly skipped speculative pair and
  fast-eval assistant tests now run and prove native assistant dispatch plus the
  production draft-token default.
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
  `3.740026` and perplexity from `42351.939379` to `42.099095` in the focused
  Metal proof run.
- The old env-only `TestRunModelEval_RealModelLoRASkip_Ugly` coverage was
  removed; Gemma-4 LoRA eval evidence now comes from the checked-in SFT smoke
  that trains, reloads, records adapter identity, and compares base vs adapter
  metrics.
- Stale LoRA adapter docs that described a non-live `go/lora_adapter.go`,
  `.npz` saves, `BaseModelHash`, and `SaveLoRAAdapter` / `LoadLoRAAdapter`
  APIs were replaced with the current `go/lora/adapter.go` +
  `go/pkg/metal/lora.go` safetensors adapter package, `WithAdapterPath`,
  `Model.LoadLoRA`, and shape/target validation contracts.
- The documented root fusion API is live again: `FuseLoRAIntoModelPack`
  validates the source pack through the shared model-pack inspector, calls the
  existing pack-level `lora.FuseIntoPack`, then validates the fused output pack.
  `TestFuseLoRAIntoModelPack_Gemma4SuffixTargetValidatesOutput_Good` runs with
  Metal enabled, uses PEFT-style Gemma-4 `q_proj` suffix tensors, proves the
  canonical fused key `model.layers.0.self_attn.q_proj.weight`, and verifies the
  fused tensor values. The real E2B q6 proof
  `TestFuseLoRAIntoModelPack_Gemma4Q6RealPackReloadGenerate_Good` fuses the
  saved rank-2 adapter into the local q6 snapshot, reloads the fused pack
  without a live adapter, and generated 256 tokens at 78.55 tok/s in the latest
  Metal proof run.
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
- The root package no longer carries an SFT-named Gemma-4 family predicate:
  `isGemma4ModelArchitecture` owns target/text/unified-but-not-assistant
  routing for dataset chat config, SFT eval prompt rendering, and Gemma-4 SFT /
  SSD LoRA target normalisation.
- Architecture profile metadata now advertises Gemma-4 target/text/unified LoRA
  targets from the same q/k/v/o, MLP, router, per-layer input gate, and
  per-layer projection policy used by adapter code, while `gemma4_assistant`
  advertises no standalone LoRA targets. The checked-in
  `TestArchitectureProfile_Gemma4LoRATargetsUseSharedPolicy_Good` pins this
  SPOR contract.
- Gemma-4 LoRA target metadata and Metal adapter resolution now share one
  policy owner in `profile`: `Gemma4LoRATargets`,
  `Gemma4DefaultLoRATargets`, `Gemma4LoRATargetPath`, and
  `Gemma4SafeLoRATarget` feed architecture metadata, safe default SFT/SSD
  targets, Metal wrapper resolution, and default target filtering instead of
  carrying per-flow lists/switches. The profile test now checks exact metadata
  equality against the shared policy, proves the safe default set is defensive
  and excludes explicit targets, and separately proves canonical
  suffix/full-path mapping plus the extended-target boundary.
- Gemma-4 target-vs-assistant architecture selection now has the same SPOR
  owner. `profile.IsGemma4TargetArchitecture` decides target/text/unified
  membership and explicitly excludes `gemma4_assistant`; root SFT/SSD family
  detection, Metal adapter-load target canonicalisation, and pack-level LoRA
  fusion now delegate to it instead of each carrying a local three-case switch.
  Focused tests cover official Transformers names, `gemma4_unified_text`, the
  attached assistant exclusion, Metal wrapper parity, and fuse suffix-key
  behaviour.
- Metal serving/runtime Gemma-4 detection now delegates to the same profile
  owner. `isGemma4RuntimeModelType` no longer carries a separate local switch;
  chat formatting, chunked chat formatting, and the fixed Gemma-4 paged-cache
  gate share `profile.IsGemma4TargetArchitecture`, so official Gemma-4 target
  class names route through the shared Gemma-4 formatter while the attached
  assistant stays excluded from target cache/prompt behaviour.
- The Gemma-4 large-variant prompt suppressor rule is now profile-owned too.
  `profile.IsGemma4LargeVariant` requires both a Gemma-4 target architecture
  and at least 16 attention heads; root dataset/SFT eval prompt config and
  Metal serving prompt config delegate to it instead of repeating the
  `NumHeads >= 16` rule locally. Tests now prove official large target/unified
  names enable the suppressor, while small Gemma-4, non-Gemma, and attached
  assistant metadata do not.
- Chat-template default selection now delegates to profile metadata instead of
  carrying a second architecture switch in `chat`. `profile.ChatTemplateName`
  owns the metadata/default lookup, while `chat.TemplateName` filters that
  result to renderers that actually exist today (`gemma4`, `gemma`, `qwen`,
  `llama`). Staged Qwen aliases remain supported through the shared profile
  fallback, and MiniMax/DeepSeek profile entries still return no chat renderer
  until real formatters are implemented.
- LoRA example coverage is no longer placeholder output for the live adapter
  path: Metal LoRA examples now assert real default config, Gemma-4 target
  canonicalisation, stable adapter names, unload, and merge behaviour; root
  `NewLoRA` now proves adapter config delegation into the native model and
  `MergeLoRA(nil)` proves the public no-op contract. The remaining Metal
  wrapper, Gemma3, and Qwen3 LoRA examples no longer print placeholder names;
  Gemma3/Qwen3 loaded-model examples are compile-only where weights are
  required, while executable examples prove cache layout, layer count, model
  type fallback/identity, and LoRA `TargetLayers` normalisation. Training docs
  now distinguish go-inference `BFloat16` compatibility from root/Metal `DType`
  and prefer reloadable adapter directories over stale single-file examples.
- Root API examples no longer echo their own function names for load/generate
  config options. `WithAdapterPath` now prints the actual adapter directory
  carried by `LoadConfig`, and the neighbouring option examples assert real
  config state or compile-only snippets when running would require Metal.
- Root backend examples no longer echo public `Model` method names. The examples
  now call `Generate`, `Chat`, stream, classify, batch, metrics, info,
  attention, KV capture, cache clear, tokenizer, close, and LoRA surfaces against
  the same fake native model used by root package tests; tensor-only helper
  examples are compile-only instead of fake computation output.
- SFT examples no longer echo method names for batch construction or checkpoint
  metadata. `BuildSFTTrainingBatches` now prints actual tokens, shifted targets,
  and loss mask from the shared fake tokenizer fixture; checkpoint save/load and
  resume examples write and read real metadata in a temporary adapter directory.
- Dataset-stream examples no longer echo method names. `BuildDatasetBatches` now
  proves packed prompt/response examples preserve response masks and shifted EOS
  targets through the same fake tokenizer fixture used by the SFT tests.
- Fast-eval examples no longer echo runner names. They now run a synthetic
  `bench.Run` path through `RunFastEval`, call `RunFastEvalBench` against the
  fake-backed root model, and prove `NewModelFastEvalRunner` preserves
  Gemma-4 adapter metadata plus generate options.
- Speculative/MTP examples no longer echo method names. They now run the
  target/draft accept-reject path, load a fake-backed speculative pair with a
  real tokenizer compatibility probe, and prove pair generation and close
  ownership contracts.
- Root training adapter examples no longer fake `Encode`, `Decode`,
  `NumLayers`, `InternalModel`, or `TrainingModel` output. They now show the
  real `inference.LoadTrainable` path and call the actual trainable model /
  Metal internal-model APIs, returning early only when no local model is loaded.
- Root training primitive examples no longer echo wrapper names. `ValueAndGrad`
  and `Checkpoint` now construct real Metal autograd closures, `NewAdamW`
  exposes live optimizer defaults, loss examples materialize scalar Metal
  losses, and `FromValues` / `Materialize` / `Free` / `Zeros` prove tensor
  lifecycle through the public root wrappers used by LoRA SFT.
- Metal AdamW examples no longer echo optimiser names. `DefaultAdamWConfig` and
  `NewAdamW` now expose live config/default state, `AdamW.Step` performs a real
  tensor update, and `AdamW.Reset` proves moment/step cleanup against the same
  optimiser used by the checked-in LoRA SFT path.
- Metal autograd/loss examples no longer echo primitive names. `VJP`, `JVP`,
  `ValueAndGrad`, `GradFn.Apply`, `GradFn.Free`, `Checkpoint`,
  `CrossEntropyLoss`, `MaskedCrossEntropyLoss`, `MSELoss`, `Log`, `SumAll`,
  `MeanAll`, and `OnesLike` now run real Metal array/autograd/loss operations
  and materialize values from the primitive surface used by LoRA SFT.
- Metal array examples no longer echo tensor helper names. `FromValue`,
  `FromValues`, `Zeros`, metadata accessors, scalar/data reads,
  `Set`/`Clone`, `SetFloat64`, shape/raw-shape access, row-contiguous
  conversion, `Free`, and `Iter` now materialize real MLX arrays and prove the
  tensor lifecycle used by LoRA weights, gradients, and AdamW state.
- Metal vector helper examples no longer echo vector wrapper names.
  `VectorArray` examples now construct, append, replace, retrieve, materialize,
  and free real MLX array vectors; `VectorString` examples now carry concrete
  Gemma-4/LoRA-style target names through append, slice, get, size, and free
  contracts.
- Metal safetensors IO examples no longer echo loader/writer names.
  `LoadSafetensors`, `LoadAllSafetensors`, custom reader load, and custom writer
  save now round-trip tiny Gemma-4 LoRA-style `q_proj` adapter tensors through
  disk and memory buffers, and the fake `MapGet` example was removed instead of
  documenting an unused C-map bridge with placeholder output.
- Core Metal ops examples no longer fake the primitive math most relevant to
  Gemma-4 projection and LoRA delta paths. Elementwise add/mul/scalar
  bridges, subtraction/division, activation helpers, matmul, softmax, reductions,
  reshape/transpose/expand/squeeze, concatenate/broadcast, and `Where` now
  materialize real MLX tensors and print stable values instead of generated
  method names.
- Additional Metal selection/masking ops examples no longer echo generated
  names. `Argmax`, `TopK`, dtype casts, strided views, gather/take,
  `Argpartition`, packed affine `Dequantize`, put/take-along-axis,
  `LogSumExp`, cumulative sums, sort/argsort, comparisons, boolean reductions,
  `Arange`, and `IsNaN` now materialize real tensors from the sampler and mask
  surface used by Gemma-4 generation. The dequantize example uses packed
  `uint32` weights with a metallib-supported affine group size instead of an
  unpacked `uint8` fixture.
- Metal slice examples no longer echo wrapper names. `Slice`, `SliceAxis`, and
  `SliceUpdateInplace` now materialize real tensor views/updates, including the
  cache-shaped update path that sits under Gemma-4 KV-cache and projection
  plumbing.
- Metal KV-cache examples no longer echo cache method names. `KVCache` and
  `RotatingKVCache` examples now update rank-4 key/value tensors, prove
  offset/length/state/reset/detach contracts, and show rotating cache output
  preserving full prompt attention while storing a bounded sliding window for
  Gemma-4 long-context state retention.
- Metal fused fast primitive examples no longer echo kernel names. `RMSNorm`,
  `RMSNormNoScale`, `LayerNorm`, `RoPE`, explicit-frequency RoPE, causal SDPA,
  and masked SDPA now materialize real tensors through the same norm/position
  embedding/attention surface used by Gemma-4 text and LoRA-forward paths.
- Metal sampler examples no longer echo sampler names. Greedy and chained
  sampling now return real token IDs, while temperature/top-k/top-p/min-p
  examples materialize filtered logits and prove retained-vs-masked candidates
  through the same generation controls used by Gemma-4 benchmarks and LoRA eval.
- Metal neural-network examples no longer echo layer names. `NewLinear`,
  quantized/dense `Linear`, expert `SwitchLinear`, `Embedding`, `AsLinear`,
  `RMSNormModule`, and `RepeatKV` now construct real layers, materialize
  forwards, and prove the base layer surface that Gemma-4 projections and LoRA
  adapters wrap.
- Metal training/model wrapper examples no longer echo `Model_*` or
  `InternalModel_*` method names. They now reuse the real tokenizer fixture,
  prove model encode/decode/tokenizer/layer/internal delegation, exercise the
  `Model.ApplyLoRA` wrapper into adapter identity state, and prove
  `InternalModel` forward/cache/LoRA contracts with a stateful in-package
  model.
- The package-level `metal.InternalModel` example now assigns a real
  in-package model to the interface and proves model type, layer count, and
  LoRA `TargetLayers` normalisation instead of printing the interface name.
- Metal backend/adapter registration examples no longer print generated method
  names. Stable contracts assert real wrapper state (`Name`, availability
  delegation); model-dependent adapter examples now compile against
  `LoadModelAsTextModel`, generation/chat/classify/batch/metrics/info/attention
  methods, and return early if the local pack is absent.
- Root `NewMLXBackend` example no longer echoes the constructor name. It now
  registers a stub inference backend, calls the real constructor, and proves the
  returned adapter name, wrapped model identity, and backend load path.
- Bundle examples no longer mix real adapter coverage with generated helper-name
  echoes. They now construct/save/load real portable Gemma-4 state bundles,
  prove defensive snapshot copies, validation, compatibility with required LoRA
  adapter identity, file/string hashes, tokenizer metadata hashes, SAMI export,
  memvid URI rendering, and defensive `TargetKeys` cloning used by portable
  state replay.
- The chat SPOR owner no longer has placeholder public examples:
  `chat.Format` now prints a real Gemma-4 large-variant prompt including the
  empty thought-channel suppressor, `TemplateName` proves official Gemma-4
  architecture routing plus explicit template override, and `NormaliseRole`
  proves live role alias canonicalisation.
- Legacy Gemma prompt examples in both tokenizer packages now print the actual
  template output instead of method-name placeholders; no production Gemma-4 /
  SPOR caller uses that helper as its formatter owner.
- Root tokenizer examples no longer echo method names. `LoadTokenizer` and the
  shared `Tokenizer` examples now load the BPE fixture and prove BOS stripping,
  decode, token lookup, `IDToken`, `BOS`, and `EOS` behaviour used by SPOR and
  SFT dataset paths.
- Internal and Metal tokenizer examples now do the same instead of echoing
  `Tokenizer_*` method names: both packages load their tiny BPE fixture and
  prove encode/decode, `DecodeToken`, BOS/EOS aliases, special-token flags, and
  vocab reverse lookup across the tokenizer surfaces used below Gemma-4 SPOR.
- Gemma-4 assistant MTP decode examples no longer echo method names. They now
  exercise real public validation paths for nil/invalid draft-step, draft-block,
  and verify calls, plus the caller-owned `Close` cleanup contracts for
  draft-step, draft-block, and verify results.
- Gemma-4 model examples no longer echo method names for the core text model
  surface. Load/forward/cache/tokenizer examples now compile against real
  `LoadGemma4`, `Forward`, `ForwardMasked`, `NewCache`, and tokenizer APIs,
  while metadata examples assert live `NumLayers` and `ModelType` behaviour.
- Gemma-4 multimodal/vision examples no longer echo method names. They now
  compile against `ForwardMultiModal`, the vision tower, patch embedder,
  encoder/layer/attention/MLP/pooler, and multimodal projector APIs using the
  real loaded-model surface, returning early only when the local pack lacks
  vision assets.
- Training docs no longer mark live LoRA fuse, fast eval, dataset stream, HF
  fit, model merge, or root training exports as planned; broken
  `lora_fuse.md`, `dataset_stream.md`, and `hf_fit.md` related links now point
  at the live `FuseLoRAIntoModelPack` docs, existing examples, or concrete code
  owners.
- SSD now carries model metadata through `SimpleSelfDistillationRunner.ModelInfo`;
  `Model.RunSimpleSelfDistillation` supplies `m.Info()` automatically, so the
  generated SFT step uses `normalizeSFTConfigForModel` and the shared Gemma-4
  LoRA target policy instead of generic q/v defaults. The checked-in
  `TestRunSimpleSelfDistillation_Gemma4ModelInfoUsesSharedLoRATargetPolicy_Good`
  proves Gemma-4 defaults include `q_proj`, `v_proj`, and `o_proj`, preserves
  decode temperature for student eval, and exposes `SampleGenerateConfig` /
  `DecodeGenerateConfig` without Python.

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
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go test -tags 'metal_runtime model_eval' -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/... -count=1
MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx
```

Production-claim artefacts must include model path+revision, quant, context
shape, command, stderr, memory method, output sample, and report path under
`docs/runtime/`.
