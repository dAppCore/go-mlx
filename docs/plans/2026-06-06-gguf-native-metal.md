<!--
SPDX-Licence-Identifier: EUPL-1.2
Co-Authored-By: Virgil <virgil@lethean.io>
-->

# GGUF → Metal, First-Class — Feasibility & Implementation Plan

**Status:** Researched plan, awaiting the config-led repair to settle before implementation.
**Last updated:** 2026-06-06.
**Companion:** `2026-06-06-llamacpp-baseline-gap-matrix.md`.

> Goal: load any ecosystem GGUF and run it natively on Metal — no llama.cpp, no Python, no sidecar files. **Verdict: achievable almost entirely in pure Go with zero new Metal kernels** for ~95% of HF-shipped GGUFs (Q4_K_M, Q5_K_M, Q6_K, Q8_0, Q4_0).

---

## 1. Where we are today

GGUF load rides MLX core's `mlx_load_gguf_arrays` (`go/pkg/metal/gguf.go:42`, vendored `lib/mlx/mlx/io/gguf.cpp` + antirez gguflib). Three tiers, per tensor:

| GGUF type | What happens now |
|---|---|
| F32/F16, I8/16/32 | direct copy |
| Q4_0, Q4_1, Q8_0 | → MLX affine 4/8-bit g32 (**lossless**, runs on tuned quant kernels) |
| Q2_K, Q4_K, Q6_K, BF16 | **dequantised to fp16** — ~3.5× file size resident, no quant speedup (an 8B Q4_K_M ≈ 4.7 GB file → ≈ 15 GB) |
| Q3_K, Q5_K, Q5_0/1, Q8_1/K, all IQ*, TQ*, MXFP4 | **load throws** — file unusable |

Two hard gaps beyond quant handling:

1. **Tensor-name binding.** Decoders bind HF names (`model.layers.N.self_attn.q_proj.weight`); ecosystem GGUFs use `blk.N.attn_q.weight`. No remap exists in `pkg/metal` — today only our own `SaveGGUF` exports (HF names preserved) round-trip. *This blocks everything else.*
2. **Tokenizer sidecar requirement.** `go/model/pack.go:502` hard-requires `tokenizer.json`; a bare `.gguf` can't chat — even though the file embeds vocab, merges, scores, special ids, pre-tokenizer selector, and `tokenizer.chat_template`, and our pure-Go parser (`go/gguf/info.go`) already walks all those keys (it currently only counts them). Note: the CGO bridge discards MLX-side metadata (`gguf_bridge.cpp:17` `(void)metadata;`) — moot, since the Go parser is the right extraction point.

---

## 2. The conversion mathematics (why this is mostly free)

MLX affine quant (CGO-reachable: `mlx_quantize` / `mlx_quantized_matmul`) supports bits {2,3,4,5,6,8} × groups {32,64,128} + modes mxfp4/nvfp4/mxfp8.

| GGUF type | Map | Fidelity |
|---|---|---|
| Q4_0 / Q4_1 / Q8_0 | affine g32 (`bias=−8d` / copy / `q⊕0x80, bias=−128d`) | **exact** (already done by MLX) |
| Q5_0 / Q5_1 | affine(5, g32) | **exact** — MLX supports 5-bit; the loader just never implemented it (~60 lines) |
| **Q4_K** | affine(4, g32): 8 sub-blocks of 32 ↔ groups of 32, `scale=d·sc`, `bias=−dmin·m` | **structurally exact** (bit-exact with fp32 scales; ≤½-ULP-fp16 otherwise — below quant noise) |
| **Q5_K** | affine(5, g32) | same — effectively exact |
| **Q6_K** | ⚠ affine(6, g32) merges its 16-element sub-scales → requantise (approx). **But:** our existing q6 bitstream kernel (`dense_matvec_q6.go`) is group-size-parameterised — **repack Q6_K at group 16 = lossless, zero new kernel** | exact via repack |
| Q2_K / Q3_K | group-16 mismatch; low-traffic | dequant to fp16 (acceptable) |
| IQ* / TQ* | codebook/LUT — cannot map to affine | dequant (needs Go-side dequant funcs; gguflib lacks them) or skip |
| MXFP4 (type 39) | MLX mode="mxfp4" (both 32-elem groups, E8M0 scale, e2m1) | likely exact — **verify scale byte encoding first** |
| BF16 | direct copy to native MLX bfloat16 (bypass gguflib's fp16 cast) | exact, trivial |

Q6_K matters more than it looks: it appears *inside every Q4_K_M file* (output / `ffn_down` / `attn_v` tensors).

Also flag: our `gguf.QuantizeQ8_K` export — llama.cpp treats Q8_K as a dot-product intermediate, never weight storage. Review for ecosystem compat.

---

## 3. Work items (dependency order)

1. **Tensor-name remap** `blk.*` ↔ HF — port the mapping table (llama.cpp `gguf-py/gguf/tensor_mapping.py`; ~40 entries covers llama/qwen/gemma). Blocking; pure Go.
2. **K-quant repacker** — Q4_K/Q5_K → MLX affine; Q6_K → q6 bitstream @ g16. Includes the 6-bit interleaved scale decoder (gguflib `gguflib.c:593–619` is the reference; our inverse already exists in `go/gguf/quantize.go`). Streams tensor-by-tensor at load. Pure Go, zero Metal.
3. **Tokenizer + config + chat template from GGUF KV** — extend `go/gguf/info.go` extraction → existing tokenizer constructors (`tokenizer.ggml.model` selects our SentencePiece vs GPT-2 BPE engines — constructor mapping, not a new tokenizer); honour `tokenizer.ggml.pre` (wrong pre-regex = silently degraded tokenisation); feed `tokenizer.chat_template` into `pack.ChatTemplate`. Drops the sidecar requirement. Precedent: mlx-examples `gguf_llm/utils.py` builds a full tokenizer purely from these keys.
4. **Long tail** — Q5_0/Q5_1 repack; Q2_K/Q3_K dequant; IQ* Go dequant funcs; MXFP4→mxfp4 mode (after verifying #2962-adjacent scale semantics); BF16 direct copy.

Config-led fit: this lands as a load-path capability, not a model change — e.g. `Features.WeightSource{GGUF{TypesNative, TypesRepacked, TypesDequant}}` declared by what the *file* contains, with the engine reacting per tensor. No model-name branches anywhere.

---

## 4. When native block kernels *would* pay (path b, later)

Only where conversion is lossy AND the type is hot: candidate = IQ4_NL/IQ4_XS (LUT nibble formats, popular at 4-bit). Reference: llama.cpp `ggml-metal.metal` per-type fused matvec with per-type threadgroup tunings (Q4_0 N_R0=4/N_SG=2, Q8_0 2/4, IQ4_NL 2/2). Our machinery exists (`metal_kernel.go` wrapping `mlx_fast_metal_kernel`, same pattern as TurboQuant/q6). Decode matvec alone leaves prefill slow — prefill via dequant-then-qmm is the pragmatic split.

---

## 5. Sources

GGUF spec (github.com/ggml-org/ggml docs/gguf.md) · block layouts (`ggml/src/ggml-common.h`) · llama.cpp Metal kernels (deepwiki 5.2) · MLX loader (vendored `lib/mlx/mlx/io/gguf.cpp`, `gguf_quants.cpp`, `ops.cpp` quantize; `lib/gguflib/gguflib.c`) · mlx-examples `llms/gguf_llm` (first-party GGUF-on-MLX precedent) · mlx-lm issue #353 · gguf2mlx (community converter) · in-repo: `go/pkg/metal/gguf.go`, `gguf_bridge.cpp`, `dense_matvec_q6.go`, `metal_kernel.go`, `go/gguf/info.go`, `go/gguf/quantize.go`, `go/model/pack.go`.
