// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

// text_config.go — copied verbatim from pkg/metal (attention.go TransformerConfig + model.go
// Gemma4TextConfig/RopeParams), the parse target for the literal config port. cgo couplings
// (the TransformerConfig embed, metal.QuantizationConfig) are the only adaptations.

type TransformerConfig struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
}

// Gemma4TextConfig holds Gemma 4 text model configuration.
type Gemma4TextConfig struct {
	// Embedded neutral core — promotes ModelType/HiddenSize/NumHiddenLayers/
	// IntermediateSize/NumAttentionHeads/NumKeyValueHeads/HeadDim/VocabSize/
	// RMSNormEps/MaxPositionEmbeddings. Shared with every model on the SDK.
	TransformerConfig

	PadTokenID                int32                 `json:"pad_token_id"`
	ImageTokenID              int32                 `json:"image_token_id"`
	AudioTokenID              int32                 `json:"audio_token_id"`
	VideoTokenID              int32                 `json:"video_token_id"`
	BOITokenID                int32                 `json:"boi_token_id"`
	BOATokenID                int32                 `json:"boa_token_id"`
	EOITokenID                int32                 `json:"eoi_token_id"`
	EOATokenIndex             int32                 `json:"eoa_token_index"`
	NumGlobalKeyValueHeads    *int32                `json:"num_global_key_value_heads"`
	GlobalHeadDim             int32                 `json:"global_head_dim"`
	GlobalPartialRotaryFactor float32               `json:"global_partial_rotary_factor"`
	VocabSizePerLayerInput    int32                 `json:"vocab_size_per_layer_input"`
	SlidingWindow             int32                 `json:"sliding_window"`
	SlidingWindowPattern      int32                 `json:"sliding_window_pattern"`
	NumKVSharedLayers         int32                 `json:"num_kv_shared_layers"`
	HiddenSizePerLayerInput   int32                 `json:"hidden_size_per_layer_input"`
	AttentionKEqV             bool                  `json:"attention_k_eq_v"`
	FinalLogitSoftcapping     float32               `json:"final_logit_softcapping"`
	UseDoubleWideMLP          bool                  `json:"use_double_wide_mlp"`
	UseDoubleWideMLPDeclared  bool                  `json:"-"`
	AttentionKEqVDeclared     bool                  `json:"-"`
	EnableMoEBlockDeclared    bool                  `json:"-"`
	EnableMoEBlock            bool                  `json:"enable_moe_block"`
	NumExperts                *int32                `json:"num_experts"`
	TopKExperts               *int32                `json:"top_k_experts"`
	MoEIntermediateSize       *int32                `json:"moe_intermediate_size"`
	TieWordEmbeddings         bool                  `json:"tie_word_embeddings"`
	RopeParameters            map[string]RopeParams `json:"rope_parameters"`
	LayerTypesInput           []string              `json:"layer_types"`

	Quantization                *QuantConfig `json:"-"`
	VisionConfig                *Gemma4VisionConfig       `json:"-"`
	AudioConfig                 *Gemma4AudioConfig        `json:"-"`
	LayerTypes                  []string                  `json:"-"`
	EmbeddingScale              float32                   `json:"-"` // Computed: sqrt(hidden_size); cached to skip per-token math.Sqrt
	PerLayerInputEmbeddingScale float32                   `json:"-"` // Computed: sqrt(hidden_size_per_layer_input); cached to skip per-token math.Sqrt
	PerLayerProjectionScale     float32                   `json:"-"` // Computed: 1/sqrt(hidden_size); cached to skip per-token math.Pow in perLayerInputTensor
}

// RopeParams holds RoPE configuration for a single attention type.
type RopeParams struct {
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	Factor              float32 `json:"factor"`
}
