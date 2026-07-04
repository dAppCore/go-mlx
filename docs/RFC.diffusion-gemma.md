# RFC: DiffusionGemma-26B-A4B — block diffusion on the LEM Engine

Status: spec distilled from first-party sources (2026-06-11). Implementation pending.
Task: #69. Model cached: `mlx-community/diffusiongemma-26B-A4B-it-4bit` (snapshot 0d2cee4a).

DeepMind's launch guidance: "you'll want a dedicated accelerator (GPU or TPU) to see
real speedups… we love our MacOs AI developers, but this model may not be best for
you." That prices in the PyTorch interpreter and a dense compute model. This engine
brings neither: the trunk is the 26B-A4B MoE we already serve compiled at 114 tok/s
(~4B active params), and the diffusion inner loop is prefill-shaped work.

## Sources (verified, first-party)

- `google-deepmind/gemma` → `gemma/diffusion/` — the authoritative JAX sampler
  (`_sampler.py`, `_transformer.py`, `_early_stopping.py`).
- `huggingface/transformers` → `models/diffusion_gemma/` — the port (generation,
  modular, conversion); transformers ≥ 5.8.0.dev0.
- vLLM blog 2026-06-10 — engine-integration perspective.
- HF checkpoint config + safetensors index (tensor map below).

## The algorithm (DeepMind `_sampler.py`, exact)

Outer loop — autoregressive ACROSS canvases, one `_sample_step` per canvas:
1. `sample_next_canvas` (inner denoising loop, below) → 256 tokens.
2. Truncate at the first stop token (rest → PAD 0); per-batch done flags.
3. `append_tokens_to_cache`: ONE causal forward over the accepted canvas writes it
   into the KV cache (standard prefill shape; positions = cache_end + arange).
4. step += canvas_length; repeat until done/limit.

Inner loop — `sample_next_canvas`, ≤ `max_denoising_steps` (HF default **48**):
- Initial canvas = **uniform-random token ids** (multinomial diffusion, NOT masks).
- Linear schedule: `noise_proportions[i] = 1 − i/S`.
- Per step (`sample_step`):
  1. Forward canvas through the trunk **with self-conditioning** (below).
     Positions are the SAME every step: cache_end + arange(L). Canvas K/V are NOT
     cached during denoising — each step concats fresh canvas K/V after the
     read-only prompt cache.
  2. Attention masks: global layers = canvas attends to all valid cache + full
     bidirectional canvas self-attention. Sliding layers = **block-local**: a fixed
     context window [cache_end − window, cache_end) SHARED by every canvas token,
     plus full canvas self-attention. (Two masks, both [B, L, cache+L].)
  3. Logits → **annealing temperature**: t = min + (max−min)·(1 − (1−noise)^exp);
     defaults max 0.8 → min 0.4, exp 1 (so t decays 0.8→0.4 as noise 1→0).
  4. **Entropy-bound acceptance** (`SampleFromPredictions`, entropy_bound 0.1):
     categorical-sample tokens from shaped logits; per-token entropy; sort
     ascending; accept the k most confident where cumsum(H)−H ≤ bound; ALL other
     positions are re-randomised to uniform tokens. Accepted + renoised = next canvas.
  5. Next self-conditioning signal = `embedder.encode_logits(shaped_logits)`:
     `softmax(logits) @ embedding_table × √d` — the expected embedding.
  6. Early stop (per batch): canvas unchanged / stability heuristics
     (`_early_stopping.py`); typical effective steps ≪ 48 on easy text.

Self-conditioning block (`_transformer.py` SelfConditioning, weights
`model.decoder.self_conditioning.*`):
```
result = RMSNorm_noscale( canvas_embeddings + FFW(RMSNorm_scaled(sc_signal)) )
```
- pre_norm carries a scale weight; post_norm is scale-FREE (pure normalisation —
  it applies even on step 0 when sc=0).
- FFW = standard gemma gate/up/down GELU MLP (`gate_proj/up_proj/down_proj`).
- PLE is ignored for canvas forwards (`ignore_ple_tokens=True`).

## Encoder/decoder (HF `modular_diffusion_gemma.py`)

- **Weight-tied**: one trunk serves both roles ("ties the text encoder with the
  decoder"). The HF split is organisational, not parametric — except:
- **Per-role layer scalars**: every layer multiplies hidden by `layer_scalar`
  (ones-init buffer). The checkpoint carries TWO sets:
  `model.encoder.language_model.layers.N.layer_scalar` (prompt-encode role) and
  `model.decoder.layers.N.layer_scalar` (denoise role).
- The encoder runs the PROMPT causally and fills the KV cache; the decoder
  denoises canvases reading that cache as read-only context, concatenating fresh
  canvas K/V per step.

## Tensor map (HF 4bit index, 1647 tensors)

- `model.decoder.layers.N.*` → exactly our gemma4 MoE layer pieces (fused
  experts.gate_up/down, router proj/scale/per_expert_scale, the four norms +
  `_2` variants, q/k/v/o + q/k norms, layer_scalar). 30 layers, hidden 2816,
  128 experts, window 1024, ctx 262144 — config-identical to gemma-4-26B-A4B.
  v_proj on 75/90 (KEqV-style on some layers, as our loader already handles).
- `model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}` — new.
- `model.encoder.language_model.layers.N.layer_scalar` — the encoder-role scalars.
- `model.encoder.vision_tower.*` (27L) + `embed_vision.embedding_projection` —
  vision; OUT OF SCOPE for the first unit (text-only).
- `model.decoder.embed_tokens` — tied embeddings (`tie_word_embeddings: true`);
  also the `encode_logits` table.
- Top-level config: `canvas_length: 256`, boi/eoi/image ids, transformers 5.8 dev.

## Engine mapping — exists vs new

| Piece | Engine status |
|---|---|
| MoE trunk forward (30L, A4B, fused experts) | EXISTS — compiled closures (#68) serve it |
| 256-token canvas forward vs static prefix | EXISTS in shape — prefill/chunk machinery; needs the bidirectional-canvas masks |
| Causal append-to-cache forward | EXISTS — prefill append |
| Block-local + global canvas masks | NEW (two explicit [L, cache+L] masks; we build masks already for MTP verify) |
| Per-role layer scalars | EXISTS (LayerScalar in the compiled key) — needs role switching (two scalar sets, same trunk) |
| Self-conditioning FFW block | NEW (tiny gemma MLP + 2 norms; reuse TracedGELUMLPForward) |
| encode_logits | NEW (softmax @ embed table × √d — one matmul) |
| Entropy-bound acceptance + annealing temp + renoise | NEW (sampler-side, host or small graphs) |
| Loader: `diffusion_gemma` model_type + `decoder.*` remap + sc block + scalar pairs | NEW (mechanical; gemma4 loader extension) |
| Generation loop (canvas outer + denoise inner + early stop) | NEW (the real work — its own generate path, NOT the AR session loop) |

## Cost model (honest)

Per 256-token canvas ≈ S_eff × T_forward(256, A4B, vs cache) + T_append(256).
- Worst case S_eff = 48; tok/s = 256 / (48·T_fwd + T_app).
- T_fwd is a 256-token MoE prefill step against the cache — measure first
  (`generate -trace` prefill rate on the 26B gives the ballpark today).
- Early stopping + entropy acceptance make S_eff content-dependent — easy text
  converges in far fewer steps; THE lever for Mac-competitive rates.
- The canvas forward is compute-parallel (DeepMind's "needs an accelerator"
  assumption) but A4B active params + compiled closures + zero interpreter
  overhead is precisely our shape. Measure before claiming.

## Implementation units

- **A — loader**: register `diffusion_gemma`; remap `model.decoder.*` onto the
  gemma4 structures; load sc block + both scalar sets + tied embed; vision
  SKIPPED. Smoke: loads + one bidirectional canvas forward returns sane logits.
- **B — denoise step**: masks (global + block-local), self-conditioning forward,
  encode_logits, annealing temp, entropy acceptance, renoise. Probe: one step on
  a tiny canvas reproduces reference shapes/dtypes.
- **C — generation loop**: outer canvas loop + early stop + stop-token truncate +
  append-to-cache; wire to a `diffuse` CLI verb with per-step trace timers
  (steps, accept-rate, ms/step — the instrument IS the demo).
- **D — serve/template**: chat template, serve route, streaming (canvas-at-a-time
  yield), MaxTokens semantics.
- **E — perf**: compiled-closure reuse for the canvas forward (L=256 trace key),
  batched acceptance on-GPU, step-count tuning, the video numbers.

## Unit E results (measured, M3 Ultra, 4bit checkpoint)

**Wave 1 — convergence semantics** (8fd93d7): reference convergence (argmax
stable `stability_threshold` consecutive steps AND mean entropy <
`confidence_threshold` 0.005; COMMIT the clean argmax always) replaced the
renoised-canvas comparison: 37 → 17-19 steps. Compiled-closure reuse KILLED as
a lever: build 1.7 ms vs eval 322 ms — the step is GPU-bound at the 26B MoE
prefill rate.

**Wave 2 — decode-profile sweep** (sky-blue prompt, seed 42, ~256-token budget):

| canvas | max steps | entropy | steps | tok/s |
|-------:|----------:|--------:|------:|------:|
| 256 | 48 | 0.3 | 18 | 24.3 |
| 256 | 24 | 0.3 | 13 | 32.8 |
| 128 | 24 | 0.3 | 22 | 38.3 |
| **64** | **16** | **0.3** | **25** | **52.3** |
| 64 | 12 | 0.3 | 30 | 44.4 |
| 32 | 12 | 0.3 | 49 | 40.8 |

Winner probes: Go linked-list code **83.3 tok/s** (7 steps total — confident
text is diffusion's best case); 588-token long-form holds **52.0** across 10
canvases. Within the gemma4 family band (12B AR = 51.8; 26B AR = 114).

Mechanics: `MaxSteps` paces the anneal (`noise = 1 − step/MaxSteps`), so
lowering it is a speed dial — until ~12, where the canvas destabilises and
re-converges (steps go UP). Entropy 0.5+ backfires the same way. Canvas cost
fits ~60 ms fixed + ~0.85 ms/token per step; the fixed floor is kernel-level.
Shipped as defaults: `DefaultCanvasLength` 64 / `DefaultMaxSteps` 16 /
`EntropyBound` 0.3 (lib zero-values, serve bridge, diffuse CLI). Banked next:
Gumbel-max sampling, bf16 sampler chain, prefix-cache reuse for commits,
kernel-level forward, batch>1.

## Verification discipline

AX-11 holds: bounded `-max-tokens`/steps, one model at a time, Snider present for
live loads. Exactness: the reference is stochastic (rng-driven) — verification is
shape/dtype/step-trace fidelity + greedy-ised variants (entropy_bound → ∞,
temp → const) for determinism probes, not byte-parity with JAX.
