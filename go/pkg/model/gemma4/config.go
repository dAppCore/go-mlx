// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
)

// Config is the backend-agnostic gemma4 model configuration: the architecture-
// relevant subset of the HF config.json. The json tags match config.json so a raw
// config unmarshals straight into it (core.JSONUnmarshal), and Arch() fills a complete
// backend-agnostic Arch — the dims-from-config step a loader needs so it never
// hand-assembles transformer dims. pkg/metal's Gemma4TextConfig carries the same
// fields (plus backend/runtime extras); this is the neutral, all-platforms mirror.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`

	FinalLogitSoftcapping float32  `json:"final_logit_softcapping"`
	SlidingWindow         int      `json:"sliding_window"`
	NumKVSharedLayers     int      `json:"num_kv_shared_layers"`
	LayerTypes            []string `json:"layer_types"`
	AttentionKEqV         bool     `json:"attention_k_eq_v"`

	VocabSizePerLayerInput  int `json:"vocab_size_per_layer_input"`
	HiddenSizePerLayerInput int `json:"hidden_size_per_layer_input"`

	EnableMoEBlock      bool `json:"enable_moe_block"`
	NumExperts          int  `json:"num_experts"`
	TopKExperts         int  `json:"top_k_experts"`
	MoEIntermediateSize int  `json:"moe_intermediate_size"`
}

// gemma4 defaults applied when a config omits the field.
const (
	defaultRopeTheta  float32 = 1_000_000 // gemma4 global RoPE base
	defaultRMSNormEps float32 = 1e-6
)

// Arch builds the backend-agnostic Arch from the config: it fills the neutral
// transformer dims + gemma4-specifics, derives the per-layer attention/KV-share specs
// (DeriveLayers over layer_types + num_kv_shared_layers), and marks every layer MoE
// when enable_moe_block is set — gemma4 applies MoE uniformly across layers, not
// interleaved (matching pkg/metal's per-layer EnableMoE = the model-wide flag).
// HeadDim defaults to hidden_size / num_attention_heads, NumKeyValueHeads to
// NumAttentionHeads (MHA), eps/rope to the gemma4 defaults, when the config omits
// them. Validates the load-bearing invariants. RopeScale is the single global scale
// the native executor consumes today; per-attention-type RoPE (sliding vs global
// theta) is a later concern.
func (c Config) Arch() (Arch, error) {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return Arch{}, core.NewError("gemma4.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}

	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return Arch{}, core.NewError("gemma4.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return Arch{}, core.NewError("gemma4.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}

	layerTypes := c.LayerTypes
	if len(layerTypes) == 0 {
		// no per-layer types declared → all global attention.
		layerTypes = make([]string, c.NumHiddenLayers)
		for i := range layerTypes {
			layerTypes[i] = "full_attention"
		}
	}
	if len(layerTypes) != c.NumHiddenLayers {
		return Arch{}, core.NewError("gemma4.Config.Arch: layer_types length must equal num_hidden_layers")
	}

	experts, topK, expertFF := 0, 0, 0
	if c.EnableMoEBlock {
		if c.NumExperts <= 0 || c.TopKExperts <= 0 {
			return Arch{}, core.NewError("gemma4.Config.Arch: enable_moe_block set but num_experts / top_k_experts not declared")
		}
		if c.TopKExperts > c.NumExperts {
			return Arch{}, core.NewError("gemma4.Config.Arch: top_k_experts must not exceed num_experts")
		}
		experts, topK = c.NumExperts, c.TopKExperts
		expertFF = c.MoEIntermediateSize
		if expertFF == 0 {
			expertFF = c.IntermediateSize // fall back to the dense FF when unspecified
		}
	}

	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}

	layers := DeriveLayers(layerTypes, c.NumKVSharedLayers)
	if c.EnableMoEBlock {
		for i := range layers {
			layers[i].MoE = true
		}
	}

	return Arch{
		Hidden:              c.HiddenSize,
		Heads:               c.NumAttentionHeads,
		KVHeads:             kvHeads,
		HeadDim:             headDim,
		FF:                  c.IntermediateSize,
		Vocab:               c.VocabSize,
		Experts:             experts,
		TopK:                topK,
		ExpertFF:            expertFF,
		Eps:                 eps,
		RopeBase:            ropeBase,
		RopeScale:           1,
		SoftCap:             c.FinalLogitSoftcapping,
		SlidingWindow:       c.SlidingWindow,
		PerLayerInputVocab:  c.VocabSizePerLayerInput,
		PerLayerInputHidden: c.HiddenSizePerLayerInput,
		AttentionKEqV:       c.AttentionKEqV,
		Layer:               layers,
	}, nil
}
