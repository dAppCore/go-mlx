// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Gemma4AudioConfig holds the Gemma 4 Unified audio projection metadata.
type Gemma4AudioConfig struct {
	ModelType            string  `json:"model_type"`
	AudioEmbedDim        int32   `json:"audio_embed_dim"`
	AudioSamplesPerToken int32   `json:"audio_samples_per_token"`
	HiddenSize           int32   `json:"hidden_size"`
	OutputProjDims       int32   `json:"output_proj_dims"`
	RMSNormEps           float32 `json:"rms_norm_eps"`
}

func normalizeGemma4AudioConfig(cfg *Gemma4AudioConfig) *Gemma4AudioConfig {
	if cfg == nil {
		return nil
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_unified_audio"
	}
	if cfg.HiddenSize == 0 {
		cfg.HiddenSize = 640
	}
	if cfg.AudioEmbedDim == 0 {
		cfg.AudioEmbedDim = cfg.HiddenSize
	}
	if cfg.AudioSamplesPerToken == 0 {
		cfg.AudioSamplesPerToken = 640
	}
	if cfg.OutputProjDims == 0 {
		cfg.OutputProjDims = cfg.HiddenSize
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
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
