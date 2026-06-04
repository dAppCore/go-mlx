// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

// Gemma4AudioConfig holds the Gemma 4 Unified audio tower metadata.
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
