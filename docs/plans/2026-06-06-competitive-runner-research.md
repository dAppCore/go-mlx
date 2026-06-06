<!--
SPDX-Licence-Identifier: EUPL-1.2
Co-Authored-By: Virgil <virgil@lethean.io>
-->

# Competitive Runner Research — vLLM · llama.cpp · MLX/mlx-lm · mlx-vlm

**Status:** Living document — candidate ideas, not committed work.
**Last updated:** 2026-06-06.
**Owner:** Snider.
**Purpose:** Mine open-source runners for techniques worth importing into go-mlx, filtered for a *single-machine, Apple-Metal, unified-memory, Go+CGO* engine. Every entry is rated for fit and effort and checked against our guardrails and our already-rejected probes.

> How to use this doc: it is a backlog of *candidates*, ranked. Nothing here is accepted until it lands in `GOAL.md`. Prune freely. When an idea graduates, move it to a dated plan and link the commit. Items are dated so this doubles as a prior-art trail (see §7).

---

## 0. Guardrails this research respects

These are lifted from `GOAL.md` / `TODO.md` / `IDEAS.md` so recommendations don't fight the project:

- **No Python** in production runtime/training/eval/benchmark paths. Python only for external comparison tooling.
- **No new `GO_MLX_ENABLE_*` env gates.** Proven features become typed config / `metal.EngineFeatures` / always-on; losers are deleted with their branch + tests.
- **darwin/arm64 only**, macOS Tahoe 26.0+ (Metal 4); **M3 Ultra** is the bench reference. EUPL-1.2, SPDX header per file, **UK English**, conventional commits, Co-Author trailer `Virgil <virgil@lethean.io>`.
- **No fake-green tests / no artificial output caps** in benchmarks; bench one model at a time.
- **256k context stays uncut** — context size may pick chunking/overflow limits but must not swap K/V family or invent a fixed-cache budget for bench convenience.
- **SPOR** (single owner) for prompt/chat formatting, adapter naming, model metadata.

### Areas you have already decided / parked — do NOT re-litigate

- **Native paged attention stays opt-in** until a *retained-workflow* win is measured (a 32k smoke moved decode 110.28 → 109.68 tok/s for ~67 MB — not worth promoting).
- **Sampler / lookahead changes are the most-gated area in the repo.** A long list of probes already regressed and were rejected *with data*: prepared-sampler prefetch (→81.3 tok/s), C++ sampler/suppression wrapper (91.6→86.3), sampled-token lookahead in prefetch boundary (empty output), scalar sampled-token sync (91.0→89.2), zero-key random handle (→90.1), yield-before-prefetch (→88.0). **Rule: no sampler/lookahead change without first extending the retained-session state-advance parity guard** (`TestSample_PrefetchTokenEvalParity_Good`, `TestModelSession_PrefetchTokenStateAdvanceParity_Good`).
- **Distributed/multi-Mac serving is deferred** until single-machine behaviour is stable.
- **TurboQuant KV is research-only**, never auto-selected by `NewPlan` until quality gates pass.

Implication for sequencing: while the codebase is mid-repair, prefer **additive, non-core-invasive** wins first (§3 tier A); save **structural / core / gated** bets (§3 tier C) for after the repair settles and the parity harness is extended.

---

## 1. TL;DR — what the survey actually found

You already own the table stakes: paged + quantized + TurboQuant KV, hash block-prefix cache, a scheduler with cancellation, OpenAI/Anthropic/Ollama HTTP, GGUF k-quants Q2_K–Q8_K, AutoRound, Gemma-4 MTP speculative decoding, and a mature sampler chain. Most "obvious" vLLM/llama.cpp ideas are **built, on your parity order, or explicitly parked.**

The genuinely useful, non-duplicative opportunities cluster into five themes:

1. **Quantisation quality multipliers you don't have yet** — an `imatrix`-style importance pass, the FP4 micro-scaled `mode` (mxfp4/nvfp4), and per-layer mixed bit-width loading.
2. **Draft-model-free speculative decoding** — prompt-lookup / suffix / Cacheback n-gram drafting: pure Go, no second model, 2–4× on RAG/code/agentic, composes with your MTP verifier. (Gated area — see §0.)
3. **The decode tail** (your stated `prefetch_logits` ~6.7 ms/token bottleneck) — fused on-device argmax/sample + single-eval boundary + `mlx::compile`. (Most-gated area — see §0.)
4. **Cache/serving refinements** — leaf-first LRU eviction for the block cache, contiguous all-layer KV block layout, unified per-step token budget (continuous batching), and an `position_ids` model-call change that unlocks *all* tree spec-decode on Metal.
5. **Cheap surface wins** — JSON-schema/grammar constrained decoding via a logits-processor hook; mlx-vlm's APC warm-disk tier and Vision Feature Cache (VLM is an embedding front-end, not a new engine).

---

## 1.5 The state-engine lens (how to weight everything below)

go-mlx is a **temporally-aware, CONT (no-replay) retained-state** engine, not a stateless role-play context window — see `docs/plans/2026-06-06-state-kv-architecture.md`. That changes what "improvement" means. Weight every idea below by whether it serves **retained multi-turn, mount-don't-replay** work. The yardstick is the C001 run — **~83 s vs llama.cpp's ~133 s over 10 turns / 9 wake-sleep restarts** — that curve is what we're bending, not cold single-shot tok/s.

Re-weighted through that lens:

- **Matters MORE than its generic rank:** contiguous all-layer KV block layout (B2 — makes CaptureKV/Sleep/Wake + spill cheap, the hot path of a retained engine); APC warm-disk block store (B1 — durable prefix tiers = more Wake hits across sessions); prompt-lookup / suffix decoding (C1 — agentic multi-turn is exactly where it pays); per-step async + single-eval boundary (C3 — shrinks the per-*tick* cost, and a tick is the unit of time here); imatrix (A1 — quality on the quantised states that get persisted and re-mounted).
- **Matters, but must round-trip through state:** any quantized-KV / fused-sampler / spec-decode change must survive `CaptureKV → Sleep → Wake → RestoreKV` **losslessly**, and must cope with a model that is *woken into mounted state* rather than re-prefilled. Speculative draft models and tree attention especially must work under CONT. This is *why* the parity-harness extension (`2026-06-06-parity-harness-extension.md`) gates them, and why its Layer 1 asserts KV-state-hash equality across all six cache families.
- **Matters LESS / skip:** anything whose only win is cold-start prefill throughput or stateless batching that ignores state continuity; any replay-assuming optimisation; multi-node disaggregation (already skipped, §2).
- **Model-capability caveat:** CONT is a radically different regime and some models can't handle it, so TRAD/replay must always remain a graceful fallback. A feature that *only* helps under CONT is still worth it — but nothing may assume CONT is always on.

## 2. Honest "skip these" list (so we don't chase them)

Unified memory + single machine dissolves several headline features of the big runners:

- **Prefill/decode disaggregation, NIXL/distributed KV transfer, DMA-vs-kernel tradeoffs** (vLLM) — multi-GPU/multi-node concerns. No second GPU to disaggregate onto. **Skip.**
- **Radix-tree prefix cache rewrite** (SGLang) — vLLM's own docs show leaf-first LRU over hash blocks is *equivalent* for full-attention models, and your hash design handles LoRA/multimodal identity more cleanly. Take the *leaf-first eviction rule* (§4.1), not the tree.
- **FA3 / FlashInfer CUDA kernels** — not portable. Steal the *idea* (one fused Metal SDPA over a mixed prefill+decode paged batch), not the code.
- **Ternary TQ1_0/TQ2_0 / 1.25-bit** — only relevant if you host BitNet-class ternary-trained models; Gemma/Qwen/Llama aren't. Defer.
- **EAGLE-3 as a quick win** — the only published Apple-Silicon number is **1.05×** on M3 Ultra (Llama-3.1-8B 4-bit), gated by tree attention + small-model economics. Your MTP path is the stronger Metal bet today. Revisit after the `position_ids` change (§4.4) and for larger/less-quantised targets.

---

## 3. Ranked candidate backlog

Effort/fit are for *our* engine. "Gated?" flags whether it touches a parked/rejected area (§0) and therefore needs a parity-harness extension or a measured retained-workflow win before it can land.

### Tier A — additive, non-core-invasive, do-able during repair

| # | Idea | Source | Fit | Effort | Gated? | Net-new since 05-09? |
|---|------|--------|-----|--------|--------|----------------------|
| A1 | **`imatrix` importance-weighted quantisation** — collect per-channel `Σ(act²)` diagonals on an MLX forward over calibration text; feed as weights into the existing k-quant/AutoRound minimiser. Mandatory below ~3-bit. | llama.cpp | High | Med | No | imatrix→GGUF format is recent |
| A2 | **FP4 micro-scaled `mode` param** (mxfp4 g32 / mxfp8 / nvfp4 g16) threaded through `mlx_quantize`/`mlx_quantized_matmul` CGO + `QuantizedLinear` loader. Structurally ideal for Gemma-4 MoE experts. | MLX | High | Med | No | Yes — gate nvfp4 (signed-E4M3 scale bug #2962) |
| A3 | **Per-layer mixed bit-width loading** — let one model carry different bits/group per layer. Unlocks dynamic-quant / DDWQ checkpoints. | mlx-lm | Med | Med | No | Yes (dynamic_quant) |
| A4 | **JSON-schema / grammar constrained decoding** via a logits-processor hook in front of the sampler (build token mask in Go, add to logits). Guaranteed valid tool-calls. | mlx-lm / mlx-vlm | High | Low-Med | No¹ | — |
| A5 | **Leaf-first LRU eviction** for `blockcache` (today: no active LRU; blocks persist until explicit clear). Closes most of the radix-tree gap. Optionally fold LoRA/multimodal IDs into block hash. | vLLM | Med | Low-Med | No | — |
| A6 | **Recommend DWQ/AWQ/GPTQ checkpoints** — they emit standard affine weights your loader already reads; ~+0.6 effective bpw from DWQ (4-bit DWQ ≈ 5-bit). Doc + CLI presets only. | mlx-lm | High | Low | No | — |
| A7 | **Quantized-KV hardening** — ensure the fused MLX SDPA path engages with quantized KV; prefer **symmetric K/V** (asymmetric falls off the fused path on Metal); add **sink-head protection** / KVarN-style variance-normalisation for long-context reasoning. | llama.cpp / research | Med | Low-Med | No | KVarN is post-05-09 |

¹ A4 is a *new* hook ahead of the sampler, not a change to the sampler's token-eval path, so it sits outside the gated boundary — but confirm it doesn't perturb first-token/RNG parity before enabling by default.

### Tier B — infra steals, medium structural

| # | Idea | Source | Fit | Effort | Gated? | Net-new? |
|---|------|--------|-----|--------|--------|----------|
| B1 | **APC warm-disk block store** — block-level (16-tok) prefix cache with warm-memory + warm-disk safetensors tiers, capacity caps, LRU disk eviction, per-tenant isolation. Maps directly onto your *disk L2 block store* amber item + existing kv-snapshot. | mlx-vlm | High | Med | No | Yes (shipping 2026) |
| B2 | **Contiguous all-layer KV block layout** — pack a logical block's K+V for *all layers* into one contiguous span. vLLM measured ~10× cheaper block moves; makes your kv-snapshot, eviction, and any spill far cheaper. Independent of offloading. | vLLM | High (design) | Med-High² | Touches KV core | Jan 2026 deep-dive |
| B3 | **Unified per-step token budget (continuous batching)** — one `max_num_batched_tokens` budget per step, mixing one prefill chunk + many decodes into a single graph eval; reconcile run/wait queues each iteration. Your parity-order item 5; pure Go control flow. | vLLM | High | High | No (extends scheduler) | async-by-default is Apr 2026 |
| B4 | **Chunked prefill** — split long prompts into fixed-size chunks co-batched with decodes; fixed chunk size keeps the Metal graph shape stable (no re-trace). Bounds the 32k-prompt stall. | vLLM | High | Med | No | — |
| B5 | **Vision Feature Cache + VLM front-end** — VLM = vision tower + projector + image-token splice + LRU feature cache on top of your existing text decode/KV/samplers. mlx-vlm shards the *LLM only*. Strategic optionality. | mlx-vlm | High (strategic) | Med-High | No | — |

² B2 touches the KV core — hold until the Claude-Code repair settles.

### Tier C — high-leverage but gated / most-invasive (post-repair, parity-harness first)

| # | Idea | Source | Fit | Effort | Gated? | Net-new? |
|---|------|--------|-----|--------|--------|----------|
| C1 | **Prompt-lookup / suffix / Cacheback n-gram drafting** — training-free, no second model, single-path verify needs no tree, pure Go string-matching; 2–4× on RAG/code/summarisation, ~1× on open-ended (so it never *hurts*). Composes with your MTP verify loop. | llama.cpp / vLLM (Arctic Suffix, NeurIPS'25) | High | Med | **Yes** — spec-decode (parity-order item 10); extend parity guard | Suffix/Cacheback are late-2025+ |
| C2 | **Fused on-device last-token argmax/sample** — FlashInfer dual-pivot *rejection* sampler ported to a Metal kernel (`mx.fast.metal_kernel`, same tooling as your TurboQuant kernels): no full 256k sort, no materialise→host→sample round-trip. Doubles as the spec-decode verifier. Directly attacks `prefetch_logits`. | FlashInfer (MLSys'25) | High | Med-High | **Yes** — most-gated area | sampler approach is 2025 |
| C3 | **Single-eval boundary + `mx.async_eval` pipelining + `mlx::compile`** — collapse draft+verify+sample into one eval; plan step N+1 while step N's GPU work runs; fuse per-step kernel launches. Your stated optimisation target. | MLX | High | Med | **Yes** — your prefetch probes already regressed here; needs the parity guard + a real measured win | mx.compile via mlx-c may need a new binding |
| C4 | **`position_ids` in model `__call__` + KV caches** — the structural prerequisite that unlocks *any* tree-based spec-decode (EAGLE/Medusa/lookahead) on Metal, because the single-integer RoPE `offset` can't express tree depths. Highest-leverage *enabler*. | MLX EAGLE-3 prototype | Med (enabler) | Med | Enables gated work | Feb-2026 finding |
| C5 | **Sampling-aware verification** — replace greedy-only verify with **modified rejection sampling** (bit-exact lossless under temp/top-p) or **typical acceptance** (Medusa-style; *gains* speed at higher temperature). Shares one kernel with C2. | research | Med-High | Med | **Yes** — spec-decode | — |

---

## 4. Per-area notes (the "why" behind the table)

### 4.1 Paged attention & KV cache

What you have is strong and largely *correct by current best practice* — mlx-vlm independently arrived at the same heterogeneous cache taxonomy you built (full-attn layers quantised, sliding-window layers on a rotating cache, **last deep full-attention layer left unquantised** — that last heuristic is a cheap 5-line tweak worth stealing, A7-adjacent).

Real gaps: **(1)** the block cache has no active eviction — add leaf-first LRU (A5); **(2)** per-layer KV is stored separately, so any block move/snapshot/spill touches `2·num_layers` fragments — a contiguous all-layer block span (B2) makes that ~10× cheaper and reinforces your page-native KV / zero-copy-restore direction in `GOAL_STRECH.md`; **(3)** for disk L2, mlx-vlm's APC warm-disk tier (B1) is a ready blueprint that maps onto your kv-snapshot surface.

### 4.2 Continuous batching / serving

Your scheduler + cancellation is production. The missing piece is the vLLM V1 *iteration-level* model: a single per-step token budget that packs one prefill chunk plus many 1-token decodes into a single MLX `Eval()` (B3 + B4). On unified memory you skip the host/device split that complicates vLLM, and fixed chunk sizes keep the Metal graph shape stable so you don't re-trace each step. Pair with **async-by-default scheduling** (plan next step during current eval) — vLLM made this the default in Apr 2026 and it cuts TTFT.

### 4.3 Quantization & formats

Three concrete adds, none gated:

- **`imatrix` (A1)** is the single biggest quality multiplier you're missing — negligible at Q6/Q8, meaningful below 4-bit, *mandatory* at 2-bit. It's a quantiser-side pass (collect `Σ(act²)` diagonals, weight the RMSE), no kernel work. AutoRound is already importance-style, so this is a natural extension.
- **FP4 `mode` (A2)** is the only way to load the new mxfp4/nvfp4 checkpoints the ecosystem is shipping; FP4 is structurally ideal for MoE experts (large resident, small active path) — relevant to Gemma-4 MoE. Gate nvfp4 behind a quality check (open MLX scale bug).
- **Per-layer mixed bits (A3)** unlocks dynamic-quant / DDWQ checkpoints — one loader change.

Don't bother re-implementing AWQ/GPTQ/DWQ as runtime ops — they emit affine weights you already load; just recommend the checkpoints and add CLI presets (A6). Note your **TurboQuant is ahead of upstream** — MLX issue #3404 tracks pulling quantized-KV-in-SDPA into core; when it lands you may be able to drop some custom-kernel maintenance. Watch post-05-09 KV-quant research (KVarN, OCTOPUS, OScaR) as possible TurboQuant successors.

### 4.4 Speculative decoding & sampling

This is your most-guarded area for good reason (§0). Two framings keep us safe:

- **The lowest-risk, highest-value spec idea is draft-model-free (C1).** Prompt-lookup / suffix / Cacheback is pure Go, needs no GPU draft pass, single-path verification needs no tree, and is lossless. It pays off exactly on the local agentic/coding/RAG workloads a single-user Mac runs, and degrades to baseline (never slower) elsewhere. It still touches the spec-decode path, so the parity guard must be extended first — but it sidesteps the sampler-boundary probes that regressed.
- **The decode-tail work (C2/C3) is your stated target but also your graveyard of rejected probes.** The research points at a *specific* shape that your earlier probes didn't try: fuse argmax/sample **on-device** in one kernel and collapse to a **single eval**, rather than host-side *prefetch* of a prepared sampler (which regressed). Treat C2/C3 as "extend the parity harness, then microbench one change at a time," not a sweep.

The **`position_ids` change (C4)** is the quiet keystone: it's modest work, isn't itself a sampler change, and unlocks every tree-based method later. Worth doing early in the gated track.

For *correctness* when you do sample-verify, use modified rejection sampling (bit-exact) or typical acceptance (faster at temp>0) — C5 — which shares the C2 kernel.

---

## 5. Suggested sequencing (proposal, not a commitment)

1. **Now / during repair (Tier A):** A1 imatrix, A2 FP4 mode, A5 leaf-first eviction, A4 constrained decoding, A6 DWQ presets, A7 quantized-KV hardening. All additive, none touch the gated cores.
2. **After repair settles (Tier B):** B1 APC disk tier, B4 chunked prefill → B3 continuous batching, B2 contiguous KV layout, A3 mixed-bit loading. Then B5 VLM front-end if it's a product direction.
3. **Gated track, parity-harness first (Tier C):** C4 `position_ids` → C1 prompt-lookup → C2/C3 fused decode tail (one microbenched change at a time) → C5 sample-aware verify → revisit EAGLE-3 for large models.

---

## 6. Open questions for Snider (steer here)

1. Of Tier A, which two do you want fleshed into a dated implementation plan first? (My pick: **A1 imatrix** + **A2 FP4 mode** — biggest quality/compat leverage, zero gated-area risk.)
2. Is **VLM (B5)** a direction you want optionality for, or out of scope for the core runner? It's cheap *if* we keep the text engine clean for it.
3. For the decode tail (C2/C3): do you want me to first draft the **parity-harness extension** spec (the retained-session state-advance guard) so the gated work has a safety net before any kernel change?
4. Should this doc track **upstream watch items** (KVarN, OCTOPUS, EAGLE-3.1, MLX #3404 TurboQuant-in-core) as a standing section you can glance at?

---

## 7. Prior-art / timestamp trail

You flagged that a KV-state idea you posted publicly showed up in others' work a week or two later. Worth converting that into a defensible trail: this repo is EUPL-1.2 and every design note here is dated and attributed. Recommend a short `docs/plans/prior-art.md` (or a section here) that timestamps each original design — page-native KV substrate, prefix DAG + copy-on-write states, TurboQuant KV layout, retained-session state-advance — with the commit hash and any public post date. Cheap to maintain, and it makes the "we shipped/described it first" claim checkable. (Happy to draft it.)

---

## 8. Sources

**vLLM / serving:** KV-offloading connector + contiguous-block layout (blog.vllm.ai/2026/01/08/kv-offloading-connector.html, Jan 2026) · scheduling/token-budget/chunked-prefill (docs.vllm.ai · audreywongkg medium) · prefix caching design + leaf-first LRU (docs.vllm.ai/en/stable/design/prefix_caching) · SGLang RadixAttention (lmsys.org/blog/2024-01-17-sglang) · layered prefill (arXiv 2510.08055) · async-by-default v0.19 (Apr 2026) · suffix decoding (snowflake.com/blog · suffix-decoding.github.io, NeurIPS'25) · EAGLE-3.1 (vllm.ai/blog/2026-05-26-eagle-3-1) · vAttention (arXiv 2405.04437) · FlashInfer (arXiv 2501.01005, github.com/flashinfer-ai/flashinfer).

**llama.cpp / ggml:** imatrix (github.com/ggml-org/llama.cpp tools/imatrix/README · PR #9400) · IQ vs k-quants + bpw (kaitchup substack) · unified quant eval (arXiv 2601.14277) · quantized-KV + FA coupling (discussions #22411 · issues #21450 #21385) · Metal backend (deepwiki ggml-org/llama.cpp 5.2) · Gemma-4 head_dim=256 SWA fix (issue #22527) · NVFP4/MXFP4 landing + Apple caveat (insiderllm.com) · KVarN (hf.co/papers/2606.03458, Jun 2026).

**MLX / mlx-lm / mlx-vlm:** learned quants DWQ/AWQ/GPTQ/dynamic (github.com/ml-explore/mlx-lm LEARNED_QUANTS.md · n8programs substack) · quantized_matmul modes (ml-explore.github.io · deepwiki ml-explore/mlx 7 · issue #2962) · custom Metal kernels (ml-explore.github.io dev/custom_metal_kernels) · TurboQuant-in-SDPA (issue #3404) · mlx-vlm APC/Vision-Feature-Cache/continuous-batching/EAGLE-3/DFlash (github.com/Blaizzy/mlx-vlm) · WWDC25 MLX (developer.apple.com/videos/play/wwdc2025/315).

**Decode fusion / spec-decode:** FlashInfer sampling (flashinfer.ai/2025/03/10/sampling.html) · FlashHead (arXiv 2603.14591) · VQ-Logits (arXiv 2505.10202) · Liger fused CE (arXiv 2410.10989) · async_eval (github.com/ml-explore/mlx discussions/1571) · MLX EAGLE-3 prototype (mlx-lm discussions/890) · speculative sampling (arXiv 2302.01318 · jaykmody.com) · Medusa typical acceptance (arXiv 2401.10774) · MTP/DeepSeek-V3 (arXiv 2412.19437) · prompt-lookup (github.com/apoorvumang/prompt-lookup-decoding) · Cacheback (arXiv 2511.21699) · Mirror SD/Apple (arXiv 2510.13161) · MLX comparative perf (arXiv 2511.05502).
