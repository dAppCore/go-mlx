// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Gemma4AudioConfig is the Gemma 4 audio tower, defined exactly as the model's
// audio_config declares it: a chunked-attention encoder (NumHiddenLayers ×
// NumAttentionHeads, chunked at AttentionChunkSize with a [ContextLeft,
// ContextRight] window) fed by a strided conv subsampler, then projected into
// the decoder embedding space. The model is the source of truth — no guessed
// dimensions; absent fields stay zero so a consumer fails loud rather than
// running a fabricated encoder.
type Gemma4AudioConfig struct {
	ModelType               string  `json:"model_type"`
	HiddenSize              int32   `json:"hidden_size"`
	NumHiddenLayers         int32   `json:"num_hidden_layers"`
	NumAttentionHeads       int32   `json:"num_attention_heads"`
	AttentionChunkSize      int32   `json:"attention_chunk_size"`
	AttentionContextLeft    int32   `json:"attention_context_left"`
	AttentionContextRight   int32   `json:"attention_context_right"`
	AttentionLogitCap       float32 `json:"attention_logit_cap"`
	ConvKernelSize          int32   `json:"conv_kernel_size"`
	SubsamplingConvChannels []int32 `json:"subsampling_conv_channels"`
	ResidualWeight          float32 `json:"residual_weight"`
	HiddenAct               string  `json:"hidden_act"`
	UseClippedLinears       bool    `json:"use_clipped_linears"`
	OutputProjDims          int32   `json:"output_proj_dims"`
	RMSNormEps              float32 `json:"rms_norm_eps"`
	// GradientClipping clamps activations between Conformer sub-blocks
	// (training-stability carry-over the reference applies at inference too).
	GradientClipping float32 `json:"gradient_clipping"`
	// AttentionInvalidLogitsValue replaces masked attention logits.
	AttentionInvalidLogitsValue float32 `json:"attention_invalid_logits_value"`
}

// normalizeGemma4AudioConfig fills only the family-universal RMS-norm epsilon
// when a config carries none (the one genuine Gemma invariant, 1e-6); every
// dimension is left as the model declared it. It does NOT invent hidden sizes
// or sample rates — a model that declares an audio tower declares its shape.
func normalizeGemma4AudioConfig(cfg *Gemma4AudioConfig) *Gemma4AudioConfig {
	if cfg == nil {
		return nil
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_unified_audio"
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	// Non-dimensional knobs absent from a checkpoint config take the HF
	// Gemma4AudioConfig defaults (configuration_gemma4.py) — published spec,
	// not invention. Dimensions stay zero and fail loud at encoder build.
	if cfg.GradientClipping == 0 {
		cfg.GradientClipping = 1e10
	}
	if cfg.AttentionInvalidLogitsValue == 0 {
		cfg.AttentionInvalidLogitsValue = -1.0e9
	}
	if cfg.HiddenAct == "" {
		cfg.HiddenAct = "silu"
	}
	if cfg.ResidualWeight == 0 {
		cfg.ResidualWeight = 0.5
	}
	return cfg
}

// Gemma4AudioProjector maps encoder-free Gemma 4 Unified audio token features
// into the decoder embedding space.
type Gemma4AudioProjector struct {
	Projection *metal.Linear
	Eps        float32
}

func sanitizeGemma4AudioWeights(raw map[string]*metal.Array) map[string]*metal.Array {
	audio := make(map[string]*metal.Array)
	for name, arr := range raw {
		canonical, ok := canonicalGemma4AudioWeightName(name)
		if !ok {
			continue
		}
		if prev, exists := audio[canonical]; exists && prev != arr {
			metal.Free(prev)
		}
		audio[canonical] = arr
		delete(raw, name)
	}
	return audio
}

func canonicalGemma4AudioWeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := trimGemma4WrapperPrefix(trimmed)
		if !changed {
			break
		}
		trimmed = next
	}
	if !core.HasPrefix(trimmed, "embed_audio") && !core.HasPrefix(trimmed, "audio_tower") {
		return "", false
	}
	return trimmed, true
}

func buildGemma4AudioProjector(cfg *Gemma4TextConfig, weights map[string]*metal.Array) *Gemma4AudioProjector {
	if len(weights) == 0 {
		return nil
	}
	audioCfg := normalizeGemma4AudioConfig(&Gemma4AudioConfig{})
	if cfg != nil && cfg.AudioConfig != nil {
		audioCfg = normalizeGemma4AudioConfig(cfg.AudioConfig)
	}
	var quantization *metal.QuantizationConfig
	if cfg != nil {
		quantization = cfg.Quantization
	}
	projector := &Gemma4AudioProjector{
		Projection: gemma4Linear(weights, "embed_audio.embedding_projection", quantization),
		Eps:        audioCfg.RMSNormEps,
	}
	if projector.Projection == nil {
		return nil
	}
	return projector
}

func (p *Gemma4AudioProjector) Forward(x *metal.Array) *metal.Array {
	if p == nil {
		return x.Clone()
	}
	normed := metal.RMSNormNoScale(x, p.Eps)
	if p.Projection == nil {
		return normed
	}
	out := p.Projection.Forward(normed)
	metal.Free(normed)
	return out
}

func closeGemma4AudioProjector(projector *Gemma4AudioProjector) {
	if projector == nil {
		return
	}
	metal.FreeLinear(projector.Projection)
}
