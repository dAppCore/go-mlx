<!--
SPDX-Licence-Identifier: EUPL-1.2
Co-Authored-By: Virgil <virgil@lethean.io>
-->

# llama.cpp Baseline — Feature / Method / Algorithm Gap Matrix

**Status:** Living document. llama.cpp is the **baseline we measure against**; vLLM / MLX / mlx-lm / mlx-vlm / mlx-engine are idea-mines only (see `2026-06-06-competitive-runner-research.md`).
**Last updated:** 2026-06-06.
**Companions:** `2026-06-06-gguf-native-metal.md` (the GGUF plan in full) · `2026-06-06-state-kv-architecture.md` (the lens).

> Framing: every gap is expressed in the **config-led idiom** — a typed declaration the engine reacts to (`Features` / `AttentionClass` / `EngineFeatures` axes, capability interfaces), never a model-name branch. Targets: **go-mlx** = Metal first + the Apple-CPU-only driver; the **HIP++ sibling** compiles the same model code to ROCm / CUDA / CPU (arm + x86), so llama.cpp's CUDA/CPU layers are *its* blueprint.
> C001 yardstick applies: prioritise what bends the retained-multi-turn curve.

---

## 1. Headline verdicts

**Where go-mlx is AHEAD of the baseline** (don't import — advertise):
- **State.** llama.cpp's `llama_state_*` is byte-copy restore with caller-driven prefix diffing and a re-prefill fallback; its server "sleep" *discards* state (wake = full reload + re-prefill); restore compat checks are self-admittedly incomplete (`// TODO: add more model-specific info…`); recurrent/hybrid checkpoints have open bugs (#22384, #24055). Your no-replay Wake/Sleep mount is the stronger model.
- **Config-led design.** llama.cpp dispatches per-arch graph builders off an `llm_arch` enum — its own maintainers describe scheduler heuristics as accumulated empirical patches. The typed `Features`/`EngineFeatures` surface is genuinely ahead; what we import is their *capability-predicate plumbing*, not their dispatch.
- **KV compression.** TurboQuant 3.5-bit has no baseline equivalent (their floor is iq4_nl/q4_0 KV).

**The biggest gap clusters** (detail in §2):
1. **Sampling & constrained generation** — we ship ~6 of their ~17 samplers, fixed order, no grammar engine, no logprobs, no stop strings, ban-only logit bias.
2. **GGUF native execution** — solvable mostly in pure Go (companion doc).
3. **Tokenizer/template breadth** — 4-ish pre-tokenizer families vs their 6 algorithms × 56 pre-types; no tool-call/reasoning parsing.
4. **Server observability & breadth** — no `/slots`, Prometheus, logprobs surface, rerank/poolings, FIM.
5. **Multimodal** — no projector runtime (mtmd equivalent); `gemma4.Features` already declares `Vision`/`Audio`, so the config surface anticipates it.

---

## 2. Domain matrices

### A. Sampling & constrained generation (baseline: `llama_sampler_chain`, everything is a vtable'd sampler composed as data — matches our idiom)

| Baseline capability | go-mlx | Typed declaration to add | Effort |
|---|---|---|---|
| Chain-as-config, user-ordered (`penalties→dry→top_n_sigma→top_k→typ_p→top_p→min_p→xtc→temp→dist`) | fixed order | `GenerateConfig.Samplers []SamplerSpec` (ordered) | M |
| logit bias (signed float, ban via −inf; `ignore_eos` = bias on EOG set) | ban-only suppression | `Features.LogitBias` — generalise suppression to signed map | **S — cheapest win** |
| stop strings w/ partial-match holdback; EOG *set* (EOS/EOT/EOM); time-based stop | stop tokens only | `StopStrings`, `EOGSet`, `TMaxPredict` | S |
| logprobs: `n_probs` top-N + post-sampling probs | none | `LogProbs{TopN, PostSampling}` (candidates already exist pre-argmax) | S–M |
| min_keep guard on all truncators | none | param on truncation samplers | S |
| typical-p · top-n-sigma (2025) · dynatemp · XTC (2024) | none | one sampler module each (top-n-sigma = mean/σ pass, trivial in MLX) | S each |
| DRY repeat suppression (2024) | none | needs shared token-history ring buffer + suffix matcher | M |
| penalties: repeat **+ freq + presence** over `penalty_last_n` window | repeat only | `Penalties{Repeat, Freq, Presence, LastN}` | S |
| mirostat v1/v2 · adaptive-p (2026) — stateful terminal selectors | none | terminal-selector slot in chain | M |
| **GBNF grammar engine** + JSON-schema→GBNF + lazy/triggered grammars (tool calls) + token-terminal rules (`<[1000]>`, 2025–26) | none | `Constraint{GBNF\|JSONSchema, Lazy, Triggers}`; copy their *validate-sampled-token-first, mask-only-on-reject* fast path | **L — highest product value (guaranteed tool-calls)** |
| GPU backend sampling (`llama_set_sampler`, 2025–26) | partial (native greedy) | extends our fused-sampler Tier-C work; note baseline asserts grammar ∉ GPU path | gated |

### B. Server / runtime surface

| Baseline | go-mlx | Typed declaration | Effort |
|---|---|---|---|
| slots + continuous batching + similarity routing; `/slots`, Prometheus `/metrics` | sessions (stronger) but no observability | `Features.SlotObservability{Slots, Prometheus}` | M |
| `/slots/{id}?action=save\|restore\|erase`, `--cache-ram` host prompt-cache tier (2025) | Wake/Sleep (stronger), no HTTP exposure | `Features.SlotStateEndpoints` — drop-in client compat | S–M |
| embeddings poolings {none,mean,cls,last,rank} + `/rerank` | stubs in daemon | `Features.Embeddings{Poolings, Rerank}` | M |
| `/infill` FIM with repo-level `input_extra` | none | `Features.FIM{RepoLevel}` (FIM token set comes free from GGUF vocab) | M |
| speculative in serve layer: draft + **model-free n-gram** (`--spec-type ngram-*`, 2025-26), chained drafters, per-request n_max/n_min/p_min | lib-level MTP only | `Features.Speculation{Draft, NGram}` — n-gram = pure Go, no second model; **gated by parity harness** | M (gated) |
| draft vocab-compat validator (type equal, size Δ≤128, token-text equal from id 5; `--spec-replace`) | none | `VocabCompatible(tgt,dft) error` | S |
| multi-model router (`--models-dir`, presets, load/unload, 2025-26) | none | `EngineFeatures.MultiModelRouter` — fits violet daemon | M–L |
| LoRA hot-swap + per-request scale + aLoRA invocation tokens (2025) | LoRA train/fuse; runtime swap partial | `Features.AdapterRuntime{HotSwap, PerRequestScale, ALoRA}` | M |
| control vectors (per-layer additive steering, GGUF format) | none | `Features.ControlVectors{LayerRange}` | S–M |

### C. KV / memory & state (read alongside the state-kv doc — this is where we're mostly ahead)

| Baseline | go-mlx | Verdict / declaration |
|---|---|---|
| memory kinds: KV / iSWA (dual sub-cache) / **recurrent** (Mamba/RWKV) / **hybrid** (Jamba, Qwen3-Next) behind `llama_memory_i` | KV + sliding + shared-KV; no recurrent/hybrid | `EngineFeatures.MemoryKinds` — add **only when a target model needs it**; the abstraction slot costs little now, kernels later |
| seq algebra: `seq_rm/cp/keep/add/div`, pos_min/max; `seq_add` = position shift (RoPE re-rotation) powering context-shift and `--cache-reuse` chunk reuse | prefix-only block cache | `Features.KVSeqOps{Remove, Copy, Keep, Shift, Divide}` per memory kind — **Shift is the one that buys something** (mid-context edit reuse) |
| per-seq state save/restore + `ON_DEVICE` flag (in-VRAM checkpoints, recent); session files embed token transcript + arch string | Wake/Sleep mount (ahead) | parity bits worth taking: arch/dims/KV-dtype **fingerprint in snapshot header**, embedded token transcript, an `OnDevice` snapshot tier |
| SWA/recurrent context checkpoints (`-ctxcp`, 2025) — replay-minimising approximation | native no-replay (ahead) | declare `Features.StateCheckpoints`; nothing to import |
| KV-quant: 9 K/V dtypes; quantised V requires FA; defrag **removed** (2025) | TurboQuant + q8/kq8vq4 (ahead) | declare `Features.KVCacheTypes`; **do not build defrag** |

### D. Tokenizer / templates / output parsing

| Baseline | go-mlx | Declaration | Effort |
|---|---|---|---|
| 6 tokenizer algorithms; **56 pre-tokenizer variants** keyed by `tokenizer.ggml.pre` | SPM + GPT-2 BPE, ~4 families | pre-tokenizer **registry keyed by config** | M (grow as models demand) |
| native Jinja engine (minja removed, late 2025), caps introspection, default-on | hard-coded per-arch templates | pragmatic path: typed per-family `ChatFormat` decls (SPOR: `chat.Format`) — full Go-Jinja is a huge lift, defer | M |
| **PEG autoparser** generates tool-call parsers from the template itself (PR #18675; `PEG_GEMMA4` specialisation) | none | `ToolCallFormat{TriggerToken, ArgsSchema}` feeding lazy grammar + stream parser → `{content, reasoning_content, tool_calls}` | M–L |
| reasoning: `reasoning_content` extraction, `--reasoning-budget` (force-close think tag at N tokens) | none | `ReasoningConfig{Tags, Budget, Format}` — decoupled from Jinja, very buildable; budget = stop-logic (mlx-vlm has same trick) | S–M |
| token healing | **baseline lacks it too** (open issues #4778/#5765) | not a gap — skip | — |

### E. Multimodal (baseline: mtmd/libmtmd — deliberately *outside* libllama)

Text GGUF + `mmproj` sidecar (encoder+projector); prompt split on media marker into chunks; media chunks carry **content-hash ids so prompt caching covers images**; embeddings enter the sequence at positions (M-RoPE aware). Maps beautifully onto the retained-KV model — encoded media is just more mounted state, and our hash-keyed blockcache extends to it directly.
→ `EngineFeatures.Modalities{Vision, Audio}` + config-led projector loader. **Natural first target: Gemma-4 vision/audio** — the decoder side is done and `gemma4.Features` already declares the flags. Effort: L.

### F. Backends — blueprints for the HIP++ sibling and the Apple-CPU driver

For **HIP++** (rocm/cuda/cpu) — llama.cpp proves the shape:
- **One kernel tree, vendor-mapping header** (`ggml-hip` compiles the CUDA sources via macro hipify; AMD deltas confined to per-gfx launch tables). Don't fork kernels per vendor.
- **Capability predicates as the load-bearing abstraction**: `supports_op` / `supports_buft` / `offload_op` per device + a scheduler that places ops by *weight residency* (`-ngl` = buffer placement, nothing more) and **demotes unsupported ops to CPU instead of erroring**. → `EngineFeatures.OpCoverage`, `Placement{LayerOffload, TensorOverride}`, `HostOffload{minBatch}`.
- **Kernel inventory**: MMQ (quantised mat-mat, int8 dp4a/tensor-core, per-quant-type instantiations) + MMVQ (quantised mat-vec for decode) + batch-size dispatch between them; FlashAttention in tiers (tensor-core / vector-per-KV-quant / tile); CUDA Graphs decode capture (~10–15%, **NVIDIA-only — do not chase on HIP**); VMM memory pool; pinned host buffers.
- Worst-case `reserve` + graph-plan reuse + mmap zero-copy weights = their per-token overhead story. → `EngineFeatures{GraphPlanReuse, WorstCaseReserve, ZeroCopyWeights}`.

For the **Apple-CPU-only driver** (derived from go-mlx):
- Runtime ISA detection via `sysctlbyname` → `Features.CPU{DotProd, I8MM, FP16, SME}`; **KleidiAI is the only route to M4-class SME matmul throughput — wrap it, don't rewrite it**.
- **Runtime weight repack** (Q4_0 → interleaved ×4/×8 blocks) implemented as a buffer-type transform at load (their on-disk repack types were deleted in favour of this — copy the lesson).
- `vec_dot` table per quant type with activations pre-quantised to Q8; spin-wait pinned threadpool sized to performance cores.

For **go-mlx/Metal**: mostly verification, since MLX owns the layer — confirm residency-set behaviour on our pinned v0.31.1, keep per-step graph shape stable (their plan-reuse lesson ≈ our fixed-chunk prefill note in the competitive doc).

---

## 3. Don't-chase list

KV defrag (removed upstream) · self-extend/group-attention in the server (removed, PR #9860) · TFS sampler (removed) · token healing (baseline lacks it) · CUDA-graph capture on HIP (buggy upstream) · their server sleep semantics (state-discarding — ours is better) · full Go-Jinja engine as a prerequisite (typed templates first).

---

## 4. Priority tiers (proposal — through the state-engine lens, respecting the repair)

1. **Tier 1 — pure-Go, ungated, do during/after config-led repair:** GGUF items 1–3 (name remap → k-quant repacker → tokenizer-from-GGUF; companion doc) · logit bias · stop strings + EOG set · logprobs · min_keep · sampler-chain-as-config scaffolding.
2. **Tier 2 — product surface:** GBNF/JSON-schema grammar + lazy triggers (tool calls) · reasoning parser + budget · new samplers (top-n-sigma, typical, dynatemp, XTC, DRY, penalties split, mirostat/adaptive-p) · embeddings poolings + rerank · vocab-compat validator.
3. **Tier 3 — engine internals (parity-harness-gated):** n-gram speculation in the serve layer · `KVSeqOps.Shift` (position re-rotation → cache-reuse) · GPU backend sampling (joins Tier-C decode-tail work).
4. **Tier 4 — strategic:** Gemma-4 vision/audio projector runtime (mtmd-shaped) · multi-model router in violet · adapter runtime (hot-swap/aLoRA) + control vectors · recurrent/hybrid memory kinds when a target model demands · HIP++ blueprint adoption (§2F).

---

## 5. Sources (key)

deepwiki ggml-org/llama.cpp (backend system 4.2, CUDA 5.1, CPU 4.3, Metal 5.2, memory 3.6, chat templates 3.9) · `tools/server/README.md` (read in full) · `include/llama.h` state/memory/sampler APIs · `common/sampling.{h,cpp}`, `common/chat.h`, `common/speculative.{h,cpp}`, `grammars/README.md`, `docs/function-calling.md`, `docs/multimodal.md` + `tools/mtmd/` · PRs: #6766 CUDA graphs, #9921/#10446 runtime repack, #11427 Metal residency sets, #13194 SWA-full, #14363 per-stream KV, #15293 context checkpoints, #16391 cache-ram, #9639 lazy grammars, #11016 Jinja, #18675 PEG autoparser, #21418 Gemma-4 parser, #9742 XTC, #6839 DRY, #11896 top-n-sigma, #17927 adaptive-p, #10455 server speculative · slaren on `ggml_backend_sched` (discussion #10182) · NVIDIA CUDA-graphs blog · issues #22384/#24055 (checkpoint bugs), #4778/#5765 (token healing, open). Full URL lists live with the four research passes that produced this matrix (conversation 2026-06-06).
