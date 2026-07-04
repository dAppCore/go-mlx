// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package gemma4 is the Gemma 4 model family on the go-mlx metal SDK (Apple
// Metal, darwin/arm64) — and the reference example for how a model package is
// built on the SDK. New families should mirror the patterns established here.
//
// # Variants
//
// One package serves the whole family; the registered loader id selects which
// path config.json drives (see load.go):
//
//   - "gemma4_text"          — text-only decoder (Gemma4ForCausalLM).
//   - "gemma4" / "gemma4_unified" — the unified multimodal model: the text
//     decoder plus multimodal projection into the text hidden size. Encoder-style
//     packs can include a SigLIP-derived vision tower; 12B Unified uses the
//     encoder-free direct vision/audio projection path.
//   - gemma4_assistant       — an attached MTP drafter, NOT a standalone model;
//     load it through LoadGemma4AssistantPair / the speculative-pair path with
//     a Gemma 4 target (loadModel rejects it as a standalone).
//
// # Config (the SPOR pattern)
//
// Every config embeds the architecture-neutral metal.TransformerConfig core
// (model_type, hidden_size, num_hidden_layers, intermediate_size, the head
// counts, head_dim, vocab_size, rms_norm_eps, max_position_embeddings) and adds
// only its family/tower-specific fields on top:
//
//   - Gemma4TextConfig   — core + token ids, sliding-window pattern, per-layer
//     inputs, MoE, partial-rotary (p-RoPE) and the unified token ids.
//   - Gemma4VisionConfig — core + SigLIP fields (image/patch/channels, the MM
//     projector dims, pooling).
//   - Gemma4AudioConfig  — audio projection metadata (kept flat: it is not a
//     full transformer config).
//   - Gemma4AssistantConfig — wraps a *Gemma4TextConfig backbone + the drafter
//     centroid fields.
//
// Architecture identification is NOT done here — the gguf/hf/model config
// probes and metal's loader dispatch all route through the single classifier in
// package profile (NormalizeArchitecture / ArchitectureFromTransformersName).
//
// # Compute
//
// gemma4 is deliberately bespoke: it composes the low-level metal primitives
// (Array, Linear, RMSNormModule, the KV caches) and its own NativeGemma*
// fused kernels rather than the shared dense DenseDecoderLayer/GQAAttention
// path, because the hybrid local/global attention, per-layer input embeddings
// and the unified multimodal fusion have no dense-family equivalent.
package gemma4
