//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"
	"strings"

	"dappco.re/go/core"

	coreio "dappco.re/go/core/io"
)

// Gemma4TextConfig holds Gemma 4 text model configuration.
type Gemma4TextConfig struct {
	ModelType                 string                `json:"model_type"`
	HiddenSize                int32                 `json:"hidden_size"`
	NumHiddenLayers           int32                 `json:"num_hidden_layers"`
	IntermediateSize          int32                 `json:"intermediate_size"`
	NumAttentionHeads         int32                 `json:"num_attention_heads"`
	NumKeyValueHeads          int32                 `json:"num_key_value_heads"`
	NumGlobalKeyValueHeads    *int32                `json:"num_global_key_value_heads"`
	HeadDim                   int32                 `json:"head_dim"`
	GlobalHeadDim             int32                 `json:"global_head_dim"`
	GlobalPartialRotaryFactor float32               `json:"global_partial_rotary_factor"`
	VocabSize                 int32                 `json:"vocab_size"`
	VocabSizePerLayerInput    int32                 `json:"vocab_size_per_layer_input"`
	RMSNormEps                float32               `json:"rms_norm_eps"`
	SlidingWindow             int32                 `json:"sliding_window"`
	SlidingWindowPattern      int32                 `json:"sliding_window_pattern"`
	MaxPositionEmbeddings     int32                 `json:"max_position_embeddings"`
	NumKVSharedLayers         int32                 `json:"num_kv_shared_layers"`
	HiddenSizePerLayerInput   int32                 `json:"hidden_size_per_layer_input"`
	AttentionKEqV             bool                  `json:"attention_k_eq_v"`
	FinalLogitSoftcapping     float32               `json:"final_logit_softcapping"`
	UseDoubleWideMLP          bool                  `json:"use_double_wide_mlp"`
	EnableMoEBlock            bool                  `json:"enable_moe_block"`
	NumExperts                *int32                `json:"num_experts"`
	TopKExperts               *int32                `json:"top_k_experts"`
	MoEIntermediateSize       *int32                `json:"moe_intermediate_size"`
	TieWordEmbeddings         bool                  `json:"tie_word_embeddings"`
	RopeParameters            map[string]RopeParams `json:"rope_parameters"`
	LayerTypesInput           []string              `json:"layer_types"`

	Quantization *QuantizationConfig `json:"-"`
	LayerTypes   []string            `json:"-"`
}

// RopeParams holds RoPE configuration for a single attention type.
type RopeParams struct {
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	Factor              float32 `json:"factor"`
}

// Gemma4Model is the Gemma 4 text model.
type Gemma4Model struct {
	EmbedTokens         *Embedding
	EmbedTokensPerLayer *Embedding
	Layers              []*Gemma4DecoderLayer
	Norm                *RMSNormModule
	Output              *Linear
	PerLayerModelProj   *Linear
	PerLayerProjNorm    *RMSNormModule

	NormScaled             *Array
	PerLayerProjNormScaled *Array

	Tok *Tokenizer
	Cfg *Gemma4TextConfig

	PreviousKVs       []int32
	CacheIndexByLayer []int32
	modelType         string
}

// Gemma4DecoderLayer is a single transformer block.
type Gemma4DecoderLayer struct {
	InputNorm    *RMSNormModule
	Attention    *Gemma4Attention
	PostAttnNorm *RMSNormModule
	PreFFNorm    *RMSNormModule
	MLP          *MLP
	PostFFNorm   *RMSNormModule

	EnableMoE   bool
	Router      *Gemma4Router
	Experts     *Gemma4Experts
	PreFFNorm2  *RMSNormModule
	PostFFNorm1 *RMSNormModule
	PostFFNorm2 *RMSNormModule

	PerLayerInputGate     *Linear
	PerLayerProjection    *Linear
	PostPerLayerInputNorm *RMSNormModule

	LayerScalar *Array

	InputNormScaled             *Array
	PostAttnNormScaled          *Array
	PreFFNormScaled             *Array
	PostFFNormScaled            *Array
	PreFFNorm2Scaled            *Array
	PostFFNorm1Scaled           *Array
	PostFFNorm2Scaled           *Array
	PostPerLayerInputNormScaled *Array

	LayerType     string
	IsSliding     bool
	DoubleWideMLP bool
	LayerIdx      int32
}

// Gemma4Attention implements Gemma 4 attention with per-layer RoPE and K-eq-V.
type Gemma4Attention struct {
	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear
	QNorm *RMSNormModule
	KNorm *RMSNormModule
	VNorm *RMSNormModule

	QNormScaled *Array
	KNormScaled *Array

	HeadDim        int32
	NKVHeads       int32
	UseKEqV        bool
	Scale          float32
	RopeBase       float32
	RopeRotatedDim int32
	RopeFreqs      *Array
}

// Gemma4Router routes tokens to top-k experts.
type Gemma4Router struct {
	Proj           *Linear
	Scale          *Array
	PerExpertScale *Array
	ScaleScaled    *Array
	RootSize       float32
	TopK           int32
	Eps            float32
}

// Gemma4Experts holds the SwitchGLU sparse MoE block.
type Gemma4Experts struct {
	GateProj *SwitchLinear
	UpProj   *SwitchLinear
	DownProj *SwitchLinear
}

type sharedKV struct {
	Keys   *Array
	Values *Array
	Offset int
}

func defaultGemma4RopeParameters(cfg *Gemma4TextConfig) map[string]RopeParams {
	return map[string]RopeParams{
		"full_attention": {
			PartialRotaryFactor: cfg.GlobalPartialRotaryFactor,
			RopeTheta:           1000000.0,
			RopeType:            "proportional",
			Factor:              1.0,
		},
		"sliding_attention": {
			PartialRotaryFactor: 1.0,
			RopeTheta:           10000.0,
			RopeType:            "default",
			Factor:              1.0,
		},
	}
}

func mergeGemma4RopeParameters(cfg *Gemma4TextConfig) {
	defaults := defaultGemma4RopeParameters(cfg)
	if cfg.RopeParameters == nil {
		cfg.RopeParameters = defaults
		return
	}

	merged := make(map[string]RopeParams, len(defaults)+len(cfg.RopeParameters))
	for attentionType, params := range defaults {
		if override, ok := cfg.RopeParameters[attentionType]; ok {
			if override.PartialRotaryFactor == 0 {
				override.PartialRotaryFactor = params.PartialRotaryFactor
			}
			if override.RopeTheta == 0 {
				override.RopeTheta = params.RopeTheta
			}
			if override.RopeType == "" {
				override.RopeType = params.RopeType
			}
			if override.Factor == 0 {
				override.Factor = params.Factor
			}
			merged[attentionType] = override
			continue
		}
		merged[attentionType] = params
	}
	for attentionType, params := range cfg.RopeParameters {
		if _, ok := merged[attentionType]; ok {
			continue
		}
		if params.Factor == 0 {
			params.Factor = 1.0
		}
		merged[attentionType] = params
	}
	cfg.RopeParameters = merged
}

func cloneGemma4Int32Ptr(v *int32) *int32 {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func cloneGemma4RopeParameters(src map[string]RopeParams) map[string]RopeParams {
	if len(src) == 0 {
		return nil
	}
	cloned := make(map[string]RopeParams, len(src))
	for attentionType, params := range src {
		cloned[attentionType] = params
	}
	return cloned
}

func overlayGemma4RopeParameters(base, overlay map[string]RopeParams) map[string]RopeParams {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}
	merged := cloneGemma4RopeParameters(base)
	if merged == nil {
		merged = make(map[string]RopeParams, len(overlay))
	}
	for attentionType, params := range overlay {
		current := merged[attentionType]
		if params.PartialRotaryFactor != 0 {
			current.PartialRotaryFactor = params.PartialRotaryFactor
		}
		if params.RopeTheta != 0 {
			current.RopeTheta = params.RopeTheta
		}
		if params.RopeType != "" {
			current.RopeType = params.RopeType
		}
		if params.Factor != 0 {
			current.Factor = params.Factor
		}
		merged[attentionType] = current
	}
	return merged
}

func mergeGemma4ConfigMissing(dst *Gemma4TextConfig, src Gemma4TextConfig) {
	if dst.ModelType == "" && src.ModelType != "" {
		dst.ModelType = src.ModelType
	}
	if dst.HiddenSize == 0 {
		dst.HiddenSize = src.HiddenSize
	}
	if dst.NumHiddenLayers == 0 {
		dst.NumHiddenLayers = src.NumHiddenLayers
	}
	if dst.IntermediateSize == 0 {
		dst.IntermediateSize = src.IntermediateSize
	}
	if dst.NumAttentionHeads == 0 {
		dst.NumAttentionHeads = src.NumAttentionHeads
	}
	if dst.NumKeyValueHeads == 0 {
		dst.NumKeyValueHeads = src.NumKeyValueHeads
	}
	if dst.NumGlobalKeyValueHeads == nil {
		dst.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(src.NumGlobalKeyValueHeads)
	}
	if dst.HeadDim == 0 {
		dst.HeadDim = src.HeadDim
	}
	if dst.GlobalHeadDim == 0 {
		dst.GlobalHeadDim = src.GlobalHeadDim
	}
	if dst.GlobalPartialRotaryFactor == 0 {
		dst.GlobalPartialRotaryFactor = src.GlobalPartialRotaryFactor
	}
	if dst.VocabSize == 0 {
		dst.VocabSize = src.VocabSize
	}
	if dst.VocabSizePerLayerInput == 0 {
		dst.VocabSizePerLayerInput = src.VocabSizePerLayerInput
	}
	if dst.RMSNormEps == 0 {
		dst.RMSNormEps = src.RMSNormEps
	}
	if dst.SlidingWindow == 0 {
		dst.SlidingWindow = src.SlidingWindow
	}
	if dst.SlidingWindowPattern == 0 {
		dst.SlidingWindowPattern = src.SlidingWindowPattern
	}
	if dst.MaxPositionEmbeddings == 0 {
		dst.MaxPositionEmbeddings = src.MaxPositionEmbeddings
	}
	if dst.NumKVSharedLayers == 0 {
		dst.NumKVSharedLayers = src.NumKVSharedLayers
	}
	if dst.HiddenSizePerLayerInput == 0 {
		dst.HiddenSizePerLayerInput = src.HiddenSizePerLayerInput
	}
	if !dst.AttentionKEqV && src.AttentionKEqV {
		dst.AttentionKEqV = true
	}
	if dst.FinalLogitSoftcapping == 0 {
		dst.FinalLogitSoftcapping = src.FinalLogitSoftcapping
	}
	if !dst.EnableMoEBlock && src.EnableMoEBlock {
		dst.EnableMoEBlock = true
	}
	if dst.NumExperts == nil {
		dst.NumExperts = cloneGemma4Int32Ptr(src.NumExperts)
	}
	if dst.TopKExperts == nil {
		dst.TopKExperts = cloneGemma4Int32Ptr(src.TopKExperts)
	}
	if dst.MoEIntermediateSize == nil {
		dst.MoEIntermediateSize = cloneGemma4Int32Ptr(src.MoEIntermediateSize)
	}
	if len(dst.LayerTypesInput) == 0 && len(src.LayerTypesInput) > 0 {
		dst.LayerTypesInput = append([]string(nil), src.LayerTypesInput...)
	}
	if len(dst.RopeParameters) == 0 && len(src.RopeParameters) > 0 {
		dst.RopeParameters = cloneGemma4RopeParameters(src.RopeParameters)
	}
}

func parseGemma4Config(data []byte) (*Gemma4TextConfig, error) {
	var wrapper struct {
		ModelType                 string                `json:"model_type"`
		Quantization              *QuantizationConfig   `json:"quantization"`
		LayerTypes                []string              `json:"layer_types"`
		NumGlobalKeyValueHeads    *int32                `json:"num_global_key_value_heads"`
		NumKVSharedLayers         *int32                `json:"num_kv_shared_layers"`
		GlobalHeadDim             *int32                `json:"global_head_dim"`
		GlobalPartialRotaryFactor *float32              `json:"global_partial_rotary_factor"`
		HiddenSizePerLayerInput   *int32                `json:"hidden_size_per_layer_input"`
		AttentionKEqV             *bool                 `json:"attention_k_eq_v"`
		FinalLogitSoftcapping     *float32              `json:"final_logit_softcapping"`
		UseDoubleWideMLP          *bool                 `json:"use_double_wide_mlp"`
		EnableMoEBlock            *bool                 `json:"enable_moe_block"`
		NumExperts                *int32                `json:"num_experts"`
		TopKExperts               *int32                `json:"top_k_experts"`
		MoEIntermediateSize       *int32                `json:"moe_intermediate_size"`
		SlidingWindow             *int32                `json:"sliding_window"`
		TieWordEmbeddings         *bool                 `json:"tie_word_embeddings"`
		RopeParameters            map[string]RopeParams `json:"rope_parameters"`
		TextConfig                struct {
			Gemma4TextConfig
			Quantization              *QuantizationConfig   `json:"quantization"`
			LayerTypes                []string              `json:"layer_types"`
			NumGlobalKeyValueHeads    *int32                `json:"num_global_key_value_heads"`
			NumKVSharedLayers         *int32                `json:"num_kv_shared_layers"`
			GlobalHeadDim             *int32                `json:"global_head_dim"`
			GlobalPartialRotaryFactor *float32              `json:"global_partial_rotary_factor"`
			HiddenSizePerLayerInput   *int32                `json:"hidden_size_per_layer_input"`
			UseDoubleWideMLP          *bool                 `json:"use_double_wide_mlp"`
			TieWordEmbeddings         *bool                 `json:"tie_word_embeddings"`
			RopeParameters            map[string]RopeParams `json:"rope_parameters"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gemma4.parseConfig", "parse config", nil)
	}

	cfg := wrapper.TextConfig.Gemma4TextConfig
	var top Gemma4TextConfig
	if r := core.JSONUnmarshal(data, &top); !r.OK {
		return nil, core.E("gemma4.parseConfig", "parse top-level fields", nil)
	}
	if cfg.NumHiddenLayers == 0 {
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("gemma4.parseConfig", "parse top-level config", nil)
		}
	} else {
		mergeGemma4ConfigMissing(&cfg, top)
	}

	if wrapper.ModelType != "" {
		cfg.ModelType = wrapper.ModelType
	}
	cfg.Quantization = wrapper.Quantization
	if cfg.Quantization == nil {
		cfg.Quantization = wrapper.TextConfig.Quantization
	}
	switch {
	case len(wrapper.LayerTypes) > 0:
		cfg.LayerTypesInput = append([]string(nil), wrapper.LayerTypes...)
	case len(wrapper.TextConfig.LayerTypes) > 0:
		cfg.LayerTypesInput = append([]string(nil), wrapper.TextConfig.LayerTypes...)
	}
	switch {
	case wrapper.NumGlobalKeyValueHeads != nil:
		cfg.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(wrapper.NumGlobalKeyValueHeads)
	case wrapper.TextConfig.NumGlobalKeyValueHeads != nil:
		cfg.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(wrapper.TextConfig.NumGlobalKeyValueHeads)
	}
	switch {
	case wrapper.NumKVSharedLayers != nil:
		cfg.NumKVSharedLayers = *wrapper.NumKVSharedLayers
	case wrapper.TextConfig.NumKVSharedLayers != nil:
		cfg.NumKVSharedLayers = *wrapper.TextConfig.NumKVSharedLayers
	}
	switch {
	case wrapper.GlobalHeadDim != nil:
		cfg.GlobalHeadDim = *wrapper.GlobalHeadDim
	case wrapper.TextConfig.GlobalHeadDim != nil:
		cfg.GlobalHeadDim = *wrapper.TextConfig.GlobalHeadDim
	}
	switch {
	case wrapper.GlobalPartialRotaryFactor != nil:
		cfg.GlobalPartialRotaryFactor = *wrapper.GlobalPartialRotaryFactor
	case wrapper.TextConfig.GlobalPartialRotaryFactor != nil:
		cfg.GlobalPartialRotaryFactor = *wrapper.TextConfig.GlobalPartialRotaryFactor
	}
	cfg.RopeParameters = overlayGemma4RopeParameters(cfg.RopeParameters, wrapper.TextConfig.RopeParameters)
	cfg.RopeParameters = overlayGemma4RopeParameters(cfg.RopeParameters, wrapper.RopeParameters)
	switch {
	case wrapper.HiddenSizePerLayerInput != nil:
		cfg.HiddenSizePerLayerInput = *wrapper.HiddenSizePerLayerInput
	case wrapper.TextConfig.HiddenSizePerLayerInput != nil:
		cfg.HiddenSizePerLayerInput = *wrapper.TextConfig.HiddenSizePerLayerInput
	}
	switch {
	case wrapper.AttentionKEqV != nil:
		cfg.AttentionKEqV = *wrapper.AttentionKEqV
	}
	switch {
	case wrapper.FinalLogitSoftcapping != nil:
		cfg.FinalLogitSoftcapping = *wrapper.FinalLogitSoftcapping
	}
	switch {
	case wrapper.EnableMoEBlock != nil:
		cfg.EnableMoEBlock = *wrapper.EnableMoEBlock
	}
	switch {
	case wrapper.NumExperts != nil:
		cfg.NumExperts = cloneGemma4Int32Ptr(wrapper.NumExperts)
	}
	switch {
	case wrapper.TopKExperts != nil:
		cfg.TopKExperts = cloneGemma4Int32Ptr(wrapper.TopKExperts)
	}
	switch {
	case wrapper.MoEIntermediateSize != nil:
		cfg.MoEIntermediateSize = cloneGemma4Int32Ptr(wrapper.MoEIntermediateSize)
	}
	switch {
	case wrapper.SlidingWindow != nil:
		cfg.SlidingWindow = *wrapper.SlidingWindow
	}
	switch {
	case wrapper.UseDoubleWideMLP != nil:
		cfg.UseDoubleWideMLP = *wrapper.UseDoubleWideMLP
	case wrapper.TextConfig.UseDoubleWideMLP != nil:
		cfg.UseDoubleWideMLP = *wrapper.TextConfig.UseDoubleWideMLP
	}
	switch {
	case wrapper.TieWordEmbeddings != nil:
		cfg.TieWordEmbeddings = *wrapper.TieWordEmbeddings
	case wrapper.TextConfig.TieWordEmbeddings != nil:
		cfg.TieWordEmbeddings = *wrapper.TextConfig.TieWordEmbeddings
	}

	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.GlobalHeadDim == 0 {
		switch {
		case wrapper.TextConfig.GlobalHeadDim != nil:
			cfg.GlobalHeadDim = *wrapper.TextConfig.GlobalHeadDim
		case wrapper.GlobalHeadDim != nil:
			cfg.GlobalHeadDim = *wrapper.GlobalHeadDim
		default:
			cfg.GlobalHeadDim = 512
		}
	}
	if cfg.GlobalPartialRotaryFactor == 0 {
		cfg.GlobalPartialRotaryFactor = 0.25
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 262144
	}
	if cfg.VocabSizePerLayerInput == 0 {
		cfg.VocabSizePerLayerInput = cfg.VocabSize
	}
	if cfg.SlidingWindow == 0 {
		cfg.SlidingWindow = 512
	}
	if cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 5
	}
	if cfg.MaxPositionEmbeddings == 0 {
		cfg.MaxPositionEmbeddings = 131072
	}
	if cfg.NumKVSharedLayers == 0 && wrapper.NumKVSharedLayers == nil && wrapper.TextConfig.NumKVSharedLayers == nil {
		cfg.NumKVSharedLayers = 20
	}
	if cfg.FinalLogitSoftcapping == 0 {
		cfg.FinalLogitSoftcapping = 30
	}
	if cfg.HiddenSizePerLayerInput == 0 {
		switch {
		case wrapper.TextConfig.HiddenSizePerLayerInput != nil:
			cfg.HiddenSizePerLayerInput = *wrapper.TextConfig.HiddenSizePerLayerInput
		case wrapper.HiddenSizePerLayerInput != nil:
			cfg.HiddenSizePerLayerInput = *wrapper.HiddenSizePerLayerInput
		default:
			cfg.HiddenSizePerLayerInput = 256
		}
	}
	if cfg.EnableMoEBlock {
		if cfg.NumExperts == nil {
			numExperts := int32(128)
			cfg.NumExperts = &numExperts
		}
		if cfg.TopKExperts == nil {
			topK := int32(8)
			cfg.TopKExperts = &topK
		}
	}
	if !cfg.UseDoubleWideMLP {
		switch {
		case wrapper.TextConfig.UseDoubleWideMLP != nil:
			cfg.UseDoubleWideMLP = *wrapper.TextConfig.UseDoubleWideMLP
		case wrapper.UseDoubleWideMLP != nil:
			cfg.UseDoubleWideMLP = *wrapper.UseDoubleWideMLP
		default:
			cfg.UseDoubleWideMLP = true
		}
	}
	if !cfg.TieWordEmbeddings {
		switch {
		case wrapper.TextConfig.TieWordEmbeddings != nil:
			cfg.TieWordEmbeddings = *wrapper.TextConfig.TieWordEmbeddings
		case wrapper.TieWordEmbeddings != nil:
			cfg.TieWordEmbeddings = *wrapper.TieWordEmbeddings
		default:
			cfg.TieWordEmbeddings = true
		}
	}
	mergeGemma4RopeParameters(&cfg)
	if len(cfg.LayerTypesInput) > 0 {
		cfg.LayerTypes = append([]string(nil), cfg.LayerTypesInput...)
	} else {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		pattern := int(cfg.SlidingWindowPattern)
		for i := range cfg.NumHiddenLayers {
			if pattern > 1 && (int(i)+1)%pattern != 0 {
				cfg.LayerTypes[i] = "sliding_attention"
			} else {
				cfg.LayerTypes[i] = "full_attention"
			}
		}
	}
	if len(cfg.LayerTypes) < int(cfg.NumHiddenLayers) {
		return nil, core.E("gemma4.parseConfig", "layer_types shorter than num_hidden_layers", nil)
	}
	cfg.LayerTypes = cfg.LayerTypes[:cfg.NumHiddenLayers]
	return &cfg, nil
}

func gemma4QuantPredicate(path string, defaultConfig *QuantizationConfig) *QuantizationConfig {
	if strings.HasSuffix(path, "router.proj") {
		return &QuantizationConfig{GroupSize: 64, Bits: 8}
	}
	return defaultConfig
}

func splitGemma4GateUpArray(a *Array) (*Array, *Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	shape := a.Shape()
	if len(shape) == 0 {
		return nil, nil, false
	}
	axis := len(shape) - 2
	if len(shape) == 1 {
		axis = 0
	} else if len(shape) == 2 {
		// Expert tensors are typically [num_experts, 2*hidden]. Split the
		// feature axis instead of the expert axis.
		axis = 1
	}
	mid := shape[axis] / 2
	if mid <= 0 || shape[axis]%2 != 0 {
		return nil, nil, false
	}
	starts := make([]int32, len(shape))
	ends := append([]int32(nil), shape...)
	ends[axis] = mid
	left := Slice(a, starts, ends)
	starts[axis] = mid
	ends = append([]int32(nil), shape...)
	right := Slice(a, starts, ends)
	return left, right, true
}

func sanitizeGemma4Weights(raw map[string]*Array) map[string]*Array {
	sanitized := make(map[string]*Array, len(raw))
	retained := make(map[*Array]struct{}, len(raw))
	discarded := make([]*Array, 0)
	for name, arr := range raw {
		canonical, skip := canonicalGemma4WeightName(name)
		if skip {
			discarded = append(discarded, arr)
			continue
		}
		for _, suffix := range []string{".weight", ".scales", ".biases", ".bias"} {
			if strings.HasSuffix(canonical, ".experts.gate_up_proj"+suffix) {
				base := strings.TrimSuffix(canonical, suffix)
				base = strings.TrimSuffix(base, ".gate_up_proj")
				gate, up, ok := splitGemma4GateUpArray(arr)
				if !ok {
					break
				}
				sanitized[base+".switch_glu.gate_proj"+suffix] = gate
				sanitized[base+".switch_glu.up_proj"+suffix] = up
				discarded = append(discarded, arr)
				goto nextWeight
			}
			if strings.HasSuffix(canonical, ".experts.down_proj"+suffix) {
				canonical = strings.TrimSuffix(canonical, ".down_proj"+suffix) + ".switch_glu.down_proj" + suffix
				break
			}
		}
		if prev, ok := sanitized[canonical]; ok && prev != arr {
			delete(retained, prev)
			discarded = append(discarded, prev)
		}
		sanitized[canonical] = arr
		if arr != nil {
			retained[arr] = struct{}{}
		}
	nextWeight:
	}
	freed := make(map[*Array]struct{}, len(discarded))
	for _, arr := range discarded {
		if arr == nil {
			continue
		}
		if _, ok := retained[arr]; ok {
			continue
		}
		if _, ok := freed[arr]; ok {
			continue
		}
		Free(arr)
		freed[arr] = struct{}{}
	}
	return sanitized
}

func canonicalGemma4WeightName(name string) (string, bool) {
	trimmed := name
	for _, prefix := range []string{
		"model.language_model.model.",
		"model.language_model.",
		"language_model.model.",
		"language_model.",
		"model.",
	} {
		if strings.HasPrefix(trimmed, prefix) {
			trimmed = strings.TrimPrefix(trimmed, prefix)
			break
		}
	}

	if strings.HasPrefix(trimmed, "vision_tower") ||
		strings.HasPrefix(trimmed, "multi_modal_projector") ||
		strings.HasPrefix(trimmed, "audio_tower") ||
		strings.HasPrefix(trimmed, "embed_audio") ||
		strings.HasPrefix(trimmed, "embed_vision") ||
		strings.Contains(trimmed, "self_attn.rotary_emb") ||
		strings.Contains(trimmed, "input_max") ||
		strings.Contains(trimmed, "input_min") ||
		strings.Contains(trimmed, "output_max") ||
		strings.Contains(trimmed, "output_min") {
		return "", true
	}

	switch {
	case strings.HasPrefix(trimmed, "layers."),
		strings.HasPrefix(trimmed, "embed_tokens."),
		strings.HasPrefix(trimmed, "embed_tokens_per_layer."),
		strings.HasPrefix(trimmed, "norm."),
		strings.HasPrefix(trimmed, "per_layer_model_projection."),
		strings.HasPrefix(trimmed, "per_layer_projection_norm."):
		return "model." + trimmed, false
	default:
		return trimmed, false
	}
}

func gemma4Ones(shape []int32) *Array {
	base := Zeros(shape, DTypeFloat32)
	ones := AddScalar(base, 1.0)
	Free(base)
	return ones
}

func gemma4WeightAny(weights map[string]*Array, names ...string) *Array {
	for _, name := range names {
		if arr := resolveWeight(weights, name); arr != nil {
			return arr
		}
	}
	return nil
}

func inferGemma4HeadDim(weights map[string]*Array, layerTypes []string, numAttentionHeads int32, target string) int32 {
	for i, layerType := range layerTypes {
		if layerType != target {
			continue
		}
		if qProj := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); qProj != nil {
			shape := qProj.Shape()
			if len(shape) > 0 && numAttentionHeads > 0 && shape[0]%numAttentionHeads == 0 {
				return shape[0] / numAttentionHeads
			}
		}
	}
	return 0
}

func inferGemma4PerLayerInputSize(weights map[string]*Array, numHiddenLayers int32) int32 {
	if numHiddenLayers <= 0 {
		return 0
	}
	if w := gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight"); w != nil {
		shape := w.Shape()
		switch len(shape) {
		case 2:
			if shape[1]%numHiddenLayers == 0 {
				return shape[1] / numHiddenLayers
			}
		case 3:
			if shape[1] == numHiddenLayers {
				return shape[2]
			}
			if shape[2] == numHiddenLayers {
				return shape[1]
			}
		default:
			if len(shape) > 1 {
				featureSize := int32(1)
				for _, dim := range shape[1:] {
					featureSize *= dim
				}
				if featureSize%numHiddenLayers == 0 {
					return featureSize / numHiddenLayers
				}
			}
		}
	}
	if w := gemma4WeightAny(weights, "model.per_layer_model_projection.weight"); w != nil {
		shape := w.Shape()
		if len(shape) >= 2 {
			outFeatures := int32(1)
			for _, dim := range shape[:len(shape)-1] {
				outFeatures *= dim
			}
			if outFeatures%numHiddenLayers == 0 {
				return outFeatures / numHiddenLayers
			}
		}
	}
	for i := int32(0); i < numHiddenLayers; i++ {
		if w := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.per_layer_input_gate.weight", i)); w != nil {
			shape := w.Shape()
			if len(shape) >= 2 && shape[0] > 0 {
				return shape[0]
			}
		}
		if w := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.per_layer_projection.weight", i)); w != nil {
			shape := w.Shape()
			if len(shape) >= 2 && shape[len(shape)-1] > 0 {
				return shape[len(shape)-1]
			}
		}
	}
	return 0
}

func gemma4Linear(weights map[string]*Array, prefix string, defaultQ *QuantizationConfig) *Linear {
	weight := gemma4WeightAny(weights, prefix+".weight")
	if weight == nil {
		return nil
	}
	scales := gemma4WeightAny(weights, prefix+".scales")
	biases := gemma4WeightAny(weights, prefix+".biases")
	bias := gemma4WeightAny(weights, prefix+".bias")
	if scales != nil {
		if q := gemma4QuantPredicate(prefix, defaultQ); q != nil {
			return NewQuantizedLinear(weight, scales, biases, bias, q.GroupSize, q.Bits)
		}
	}
	return NewLinear(weight, bias)
}

func gemma4SwitchLinear(weights map[string]*Array, defaultQ *QuantizationConfig, prefixes ...string) *SwitchLinear {
	for _, prefix := range prefixes {
		weight := gemma4WeightAny(weights, prefix+".weight")
		if weight == nil {
			continue
		}
		scales := gemma4WeightAny(weights, prefix+".scales")
		biases := gemma4WeightAny(weights, prefix+".biases")
		bias := gemma4WeightAny(weights, prefix+".bias")
		if scales != nil {
			if q := gemma4QuantPredicate(prefix, defaultQ); q != nil {
				return NewQuantizedSwitchLinear(weight, scales, biases, bias, q.GroupSize, q.Bits)
			}
		}
		return NewSwitchLinear(weight, bias)
	}
	return nil
}

func gemma4OutputLinear(weights map[string]*Array, cfg *Gemma4TextConfig, embed *Embedding) (*Linear, error) {
	if output := gemma4Linear(weights, "lm_head", cfg.Quantization); output != nil {
		return output, nil
	}
	if cfg.TieWordEmbeddings {
		if embed == nil {
			return nil, core.E("gemma4.outputLinear", "tied output requested without embed_tokens", nil)
		}
		return embed.AsLinear(), nil
	}
	return nil, core.E("gemma4.outputLinear", "missing lm_head.weight with tie_word_embeddings=false", nil)
}

func buildGemma4CacheLayout(layers []*Gemma4DecoderLayer, numShared int32) ([]int32, []int32) {
	previous := make([]int32, len(layers))
	cacheIndexByLayer := make([]int32, len(layers))
	for i := range previous {
		previous[i] = int32(i)
		cacheIndexByLayer[i] = -1
	}
	if len(layers) == 0 {
		return previous, cacheIndexByLayer
	}
	firstShared := int32(len(layers)) - numShared
	if firstShared < 0 {
		firstShared = 0
	}
	if firstShared > int32(len(layers)) {
		firstShared = int32(len(layers))
	}
	latestByType := make(map[string]int32)
	nextCacheIndex := int32(0)
	for i := int32(0); i < int32(len(layers)); i++ {
		layerType := layers[i].LayerType
		ownsCache := i < firstShared
		if !ownsCache {
			if prev, ok := latestByType[layerType]; ok {
				previous[i] = prev
			} else {
				// Small toy configs can place the first layer of an attention type
				// in the shared-KV region. Promote it to an owner so decoding keeps
				// a persistent cache instead of silently recomputing from scratch.
				ownsCache = true
			}
		}
		if ownsCache {
			previous[i] = i
			latestByType[layerType] = i
			cacheIndexByLayer[i] = nextCacheIndex
			nextCacheIndex++
		}
	}
	return previous, cacheIndexByLayer
}

func buildGemma4PreviousKVs(layers []*Gemma4DecoderLayer, numShared int32) []int32 {
	previous, _ := buildGemma4CacheLayout(layers, numShared)
	return previous
}

func gemma4RotatedDims(headDim int32, params RopeParams) int32 {
	factor := params.PartialRotaryFactor
	if factor <= 0 {
		factor = 1
	}
	dims := int32(math.Round(float64(float32(headDim) * factor)))
	if dims <= 0 {
		dims = headDim
	}
	if dims > headDim {
		dims = headDim
	}
	if dims%2 != 0 {
		dims--
	}
	if dims <= 0 {
		dims = headDim
	}
	return dims
}

func gemma4ProportionalFreqs(headDim int32, rotatedDims int32, base float32, factor float32) *Array {
	if rotatedDims <= 0 {
		return nil
	}
	exponents := Arange(0, float64(rotatedDims), 2, DTypeFloat32)
	scale := float32(1.0 / float32(headDim))
	exponentsScaled := MulScalar(exponents, scale)
	Free(exponents)
	baseScalar := FromValue(base)
	freqs := Power(baseScalar, exponentsScaled)
	Free(baseScalar, exponentsScaled)
	if factor != 0 && factor != 1 {
		scaled := MulScalar(freqs, factor)
		Free(freqs)
		freqs = scaled
	}
	if rotatedDims < headDim {
		extra := make([]float32, (headDim-rotatedDims)/2)
		for i := range extra {
			extra[i] = float32(math.Inf(1))
		}
		inf := FromValues(extra, len(extra))
		combined := Concatenate([]*Array{freqs, inf}, 0)
		Free(freqs, inf)
		freqs = combined
	}
	return freqs
}

func gemma4AttentionScale(headDim int32) float32 {
	if headDim <= 0 {
		return 1.0
	}
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

func precomputeGemma4ScaledWeights(m *Gemma4Model) {
	if m.Norm != nil {
		m.NormScaled = AddScalar(m.Norm.Weight, 1.0)
	}
	if m.PerLayerProjNorm != nil && m.PerLayerProjNorm.Weight != nil {
		m.PerLayerProjNormScaled = AddScalar(m.PerLayerProjNorm.Weight, 1.0)
	}

	var scaled []*Array
	scaled = append(scaled, m.NormScaled, m.PerLayerProjNormScaled)

	for _, layer := range m.Layers {
		if layer.InputNorm != nil && layer.InputNorm.Weight != nil {
			layer.InputNormScaled = AddScalar(layer.InputNorm.Weight, 1.0)
		}
		if layer.PostAttnNorm != nil && layer.PostAttnNorm.Weight != nil {
			layer.PostAttnNormScaled = AddScalar(layer.PostAttnNorm.Weight, 1.0)
		}
		if layer.PreFFNorm != nil && layer.PreFFNorm.Weight != nil {
			layer.PreFFNormScaled = AddScalar(layer.PreFFNorm.Weight, 1.0)
		}
		if layer.PostFFNorm != nil && layer.PostFFNorm.Weight != nil {
			layer.PostFFNormScaled = AddScalar(layer.PostFFNorm.Weight, 1.0)
		}
		if layer.PreFFNorm2 != nil && layer.PreFFNorm2.Weight != nil {
			layer.PreFFNorm2Scaled = AddScalar(layer.PreFFNorm2.Weight, 1.0)
		}
		if layer.PostFFNorm1 != nil && layer.PostFFNorm1.Weight != nil {
			layer.PostFFNorm1Scaled = AddScalar(layer.PostFFNorm1.Weight, 1.0)
		}
		if layer.PostFFNorm2 != nil && layer.PostFFNorm2.Weight != nil {
			layer.PostFFNorm2Scaled = AddScalar(layer.PostFFNorm2.Weight, 1.0)
		}
		if layer.PostPerLayerInputNorm != nil && layer.PostPerLayerInputNorm.Weight != nil {
			layer.PostPerLayerInputNormScaled = AddScalar(layer.PostPerLayerInputNorm.Weight, 1.0)
		}
		if layer.Attention != nil {
			if layer.Attention.QNorm != nil && layer.Attention.QNorm.Weight != nil {
				layer.Attention.QNormScaled = AddScalar(layer.Attention.QNorm.Weight, 1.0)
			}
			if layer.Attention.KNorm != nil && layer.Attention.KNorm.Weight != nil {
				layer.Attention.KNormScaled = AddScalar(layer.Attention.KNorm.Weight, 1.0)
			}
			scaled = append(scaled, layer.Attention.QNormScaled, layer.Attention.KNormScaled, layer.Attention.RopeFreqs)
		}
		if layer.Router != nil && layer.Router.Scale != nil {
			layer.Router.ScaleScaled = MulScalar(layer.Router.Scale, layer.Router.RootSize)
			scaled = append(scaled, layer.Router.ScaleScaled)
		}
		scaled = append(
			scaled,
			layer.InputNormScaled,
			layer.PostAttnNormScaled,
			layer.PreFFNormScaled,
			layer.PostFFNormScaled,
			layer.PreFFNorm2Scaled,
			layer.PostFFNorm1Scaled,
			layer.PostFFNorm2Scaled,
			layer.PostPerLayerInputNormScaled,
		)
	}
	Materialize(scaled...)
}

func (m *Gemma4Model) ensureCacheLayout() {
	if len(m.PreviousKVs) == len(m.Layers) && len(m.CacheIndexByLayer) == len(m.Layers) {
		return
	}
	previous, cacheIndexByLayer := buildGemma4CacheLayout(m.Layers, m.Cfg.NumKVSharedLayers)
	m.PreviousKVs = previous
	m.CacheIndexByLayer = cacheIndexByLayer
}

// LoadGemma4 loads a Gemma 4 text model from a directory.
func LoadGemma4(modelPath string) (*Gemma4Model, error) {
	root := resolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseGemma4Config(data)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "parse config", err)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load tokenizer", err)
	}

	rawWeights, err := loadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load weights", err)
	}
	weights := sanitizeGemma4Weights(rawWeights)

	if inferred := inferGemma4HeadDim(weights, cfg.LayerTypes, cfg.NumAttentionHeads, "sliding_attention"); inferred > 0 {
		cfg.HeadDim = inferred
	}
	if inferred := inferGemma4HeadDim(weights, cfg.LayerTypes, cfg.NumAttentionHeads, "full_attention"); inferred > 0 {
		cfg.GlobalHeadDim = inferred
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.GlobalHeadDim == 0 {
		cfg.GlobalHeadDim = 512
	}

	if inferred := inferGemma4PerLayerInputSize(weights, cfg.NumHiddenLayers); inferred > 0 {
		cfg.HiddenSizePerLayerInput = inferred
	}
	if cfg.HiddenSizePerLayerInput > 0 {
		if gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight") == nil ||
			gemma4WeightAny(weights, "model.per_layer_model_projection.weight") == nil ||
			gemma4WeightAny(weights, "model.per_layer_projection_norm.weight") == nil {
			cfg.HiddenSizePerLayerInput = 0
		}
	}

	modelType := cfg.ModelType
	if modelType == "" {
		modelType = "gemma4_text"
	}

	embed := &Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens.weight")}
	if embedScales := gemma4WeightAny(weights, "model.embed_tokens.scales"); embedScales != nil && cfg.Quantization != nil {
		embed.Scales = embedScales
		embed.Biases = gemma4WeightAny(weights, "model.embed_tokens.biases")
		embed.GroupSize = cfg.Quantization.GroupSize
		embed.Bits = cfg.Quantization.Bits
	}

	var embedPerLayer *Embedding
	if cfg.HiddenSizePerLayerInput > 0 {
		embedPerLayer = &Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight")}
		if scales := gemma4WeightAny(weights, "model.embed_tokens_per_layer.scales"); scales != nil && cfg.Quantization != nil {
			embedPerLayer.Scales = scales
			embedPerLayer.Biases = gemma4WeightAny(weights, "model.embed_tokens_per_layer.biases")
			embedPerLayer.GroupSize = cfg.Quantization.GroupSize
			embedPerLayer.Bits = cfg.Quantization.Bits
		}
	}

	m := &Gemma4Model{
		EmbedTokens:         embed,
		EmbedTokensPerLayer: embedPerLayer,
		Layers:              make([]*Gemma4DecoderLayer, cfg.NumHiddenLayers),
		Norm:                &RMSNormModule{Weight: gemma4WeightAny(weights, "model.norm.weight")},
		Tok:                 tok,
		Cfg:                 cfg,
		modelType:           modelType,
	}

	if cfg.HiddenSizePerLayerInput > 0 {
		m.PerLayerModelProj = gemma4Linear(weights, "model.per_layer_model_projection", cfg.Quantization)
		m.PerLayerProjNorm = &RMSNormModule{Weight: gemma4WeightAny(weights, "model.per_layer_projection_norm.weight")}
	}

	firstShared := cfg.NumHiddenLayers - cfg.NumKVSharedLayers
	if firstShared < 0 {
		firstShared = 0
	}
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("model.layers.%d", i)
		layerType := cfg.LayerTypes[i]
		isSliding := layerType == "sliding_attention"
		headDim := cfg.HeadDim
		if !isSliding && cfg.GlobalHeadDim > 0 {
			headDim = cfg.GlobalHeadDim
		}
		nkvHeads := cfg.NumKeyValueHeads
		useKEqV := cfg.AttentionKEqV && !isSliding
		if useKEqV && cfg.NumGlobalKeyValueHeads != nil {
			nkvHeads = *cfg.NumGlobalKeyValueHeads
		}

		ropeParams := cfg.RopeParameters[layerType]
		rotatedDims := gemma4RotatedDims(headDim, ropeParams)
		var ropeFreqs *Array
		if ropeParams.RopeType == "proportional" {
			factor := ropeParams.Factor
			if factor == 0 {
				factor = 1
			}
			ropeFreqs = gemma4ProportionalFreqs(headDim, rotatedDims, float32(ropeParams.RopeTheta), factor)
		}

		layer := &Gemma4DecoderLayer{
			InputNorm:    &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".input_layernorm.weight")},
			PostAttnNorm: &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_attention_layernorm.weight")},
			PreFFNorm:    &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm.weight")},
			Attention: &Gemma4Attention{
				QProj:          gemma4Linear(weights, prefix+".self_attn.q_proj", cfg.Quantization),
				KProj:          gemma4Linear(weights, prefix+".self_attn.k_proj", cfg.Quantization),
				VProj:          gemma4Linear(weights, prefix+".self_attn.v_proj", cfg.Quantization),
				OProj:          gemma4Linear(weights, prefix+".self_attn.o_proj", cfg.Quantization),
				QNorm:          &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".self_attn.q_norm.weight")},
				KNorm:          &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".self_attn.k_norm.weight")},
				VNorm:          &RMSNormModule{},
				HeadDim:        headDim,
				NKVHeads:       nkvHeads,
				UseKEqV:        useKEqV,
				Scale:          gemma4AttentionScale(headDim),
				RopeBase:       float32(ropeParams.RopeTheta),
				RopeRotatedDim: rotatedDims,
				RopeFreqs:      ropeFreqs,
			},
			MLP: &MLP{
				GateProj: gemma4Linear(weights, prefix+".mlp.gate_proj", cfg.Quantization),
				UpProj:   gemma4Linear(weights, prefix+".mlp.up_proj", cfg.Quantization),
				DownProj: gemma4Linear(weights, prefix+".mlp.down_proj", cfg.Quantization),
			},
			LayerScalar:   gemma4WeightAny(weights, prefix+".layer_scalar", prefix+".layer_scalar.weight"),
			LayerType:     layerType,
			IsSliding:     isSliding,
			DoubleWideMLP: cfg.UseDoubleWideMLP && cfg.NumKVSharedLayers > 0 && i >= firstShared,
			LayerIdx:      i,
			EnableMoE:     cfg.EnableMoEBlock,
		}
		if layer.LayerScalar == nil {
			layer.LayerScalar = gemma4Ones([]int32{1})
		}
		if useKEqV {
			layer.Attention.VProj = nil
		}

		if cfg.EnableMoEBlock {
			routerScale := gemma4WeightAny(weights, prefix+".router.scale", prefix+".router.scale.weight")
			if routerScale == nil {
				routerScale = gemma4Ones([]int32{cfg.HiddenSize})
			}
			perExpertScale := gemma4WeightAny(weights, prefix+".router.per_expert_scale", prefix+".router.per_expert_scale.weight")
			if perExpertScale == nil && cfg.NumExperts != nil {
				perExpertScale = gemma4Ones([]int32{*cfg.NumExperts})
			}
			layer.Router = &Gemma4Router{
				Proj:           gemma4Linear(weights, prefix+".router.proj", cfg.Quantization),
				Scale:          routerScale,
				PerExpertScale: perExpertScale,
				RootSize:       float32(math.Pow(float64(cfg.HiddenSize), -0.5)),
				TopK:           valueOrDefault(cfg.TopKExperts, 0),
				Eps:            cfg.RMSNormEps,
			}
			layer.Experts = &Gemma4Experts{
				GateProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.gate_proj",
					prefix+".experts.gate_proj",
				),
				UpProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.up_proj",
					prefix+".experts.up_proj",
				),
				DownProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.down_proj",
					prefix+".experts.down_proj",
				),
			}
			layer.PreFFNorm2 = &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".pre_feedforward_layernorm_2.weight")}
			layer.PostFFNorm1 = &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm_1.weight")}
			layer.PostFFNorm2 = &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm_2.weight")}
		}

		if cfg.HiddenSizePerLayerInput > 0 {
			layer.PerLayerInputGate = gemma4Linear(weights, prefix+".per_layer_input_gate", cfg.Quantization)
			layer.PerLayerProjection = gemma4Linear(weights, prefix+".per_layer_projection", cfg.Quantization)
			layer.PostPerLayerInputNorm = &RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_per_layer_input_norm.weight")}
			if layer.PerLayerInputGate == nil || layer.PerLayerProjection == nil || layer.PostPerLayerInputNorm.Weight == nil {
				layer.PerLayerInputGate = nil
				layer.PerLayerProjection = nil
				layer.PostPerLayerInputNorm = nil
			}
		}

		m.Layers[i] = layer
	}

	m.Output, err = gemma4OutputLinear(weights, cfg, m.EmbedTokens)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "build output projection", err)
	}

	m.PreviousKVs, m.CacheIndexByLayer = buildGemma4CacheLayout(m.Layers, cfg.NumKVSharedLayers)

	var allArrays []*Array
	for _, arr := range weights {
		allArrays = append(allArrays, arr)
	}
	Materialize(allArrays...)
	precomputeGemma4ScaledWeights(m)

	return m, nil
}

func valueOrDefault(v *int32, def int32) int32 {
	if v == nil {
		return def
	}
	return *v
}

func gemma4NormalizePerLayerTensor(x *Array, batchSize, seqLen, numLayers, hiddenSize int32) *Array {
	if x == nil || !x.Valid() {
		return x
	}

	shape := x.Shape()
	switch len(shape) {
	case 4:
		if shape[2] == numLayers && shape[3] == hiddenSize {
			return x
		}
		if shape[2] == hiddenSize && shape[3] == numLayers {
			return Transpose(x, 0, 1, 3, 2)
		}
	case 3:
		if shape[2] == numLayers*hiddenSize {
			return Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
		}
	}

	return Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
}

func (m *Gemma4Model) computePerLayerInputs(tokens, hidden *Array) []*Array {
	if m.EmbedTokensPerLayer == nil || m.PerLayerModelProj == nil || m.PerLayerProjNorm == nil || m.PerLayerProjNormScaled == nil {
		return nil
	}
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	perLayer := m.EmbedTokensPerLayer.Forward(tokens)
	scale := float32(math.Sqrt(float64(m.Cfg.HiddenSizePerLayerInput)))
	scaled := MulScalar(perLayer, scale)
	Free(perLayer)
	perLayer = gemma4NormalizePerLayerTensor(scaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if perLayer != scaled {
		Free(scaled)
	}

	projected := m.PerLayerModelProj.Forward(hidden)
	projectedScaled := MulScalar(projected, float32(math.Pow(float64(m.Cfg.HiddenSize), -0.5)))
	Free(projected)
	projected = gemma4NormalizePerLayerTensor(projectedScaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if projected != projectedScaled {
		Free(projectedScaled)
	}
	projectedNormed := RMSNorm(projected, m.PerLayerProjNormScaled, m.Cfg.RMSNormEps)
	Free(projected)

	combined := Add(projectedNormed, perLayer)
	Free(projectedNormed, perLayer)
	combinedScaled := MulScalar(combined, float32(math.Pow(2, -0.5)))
	Free(combined)
	combined = combinedScaled

	perLayerInputs := make([]*Array, m.Cfg.NumHiddenLayers)
	for i := range m.Cfg.NumHiddenLayers {
		sliced := SliceAxis(combined, 2, i, i+1)
		perLayerInputs[i] = Squeeze(sliced, 2)
		Free(sliced)
	}
	Free(combined)
	return perLayerInputs
}

func buildGemma4SlidingMask(batchSize, seqLen, window int32) *Array {
	negInf := float32(math.Inf(-1))
	data := make([]float32, int(batchSize)*int(seqLen)*int(seqLen))
	for b := range batchSize {
		base := int(b) * int(seqLen) * int(seqLen)
		for i := range seqLen {
			for j := range seqLen {
				if j <= i && i-j < window {
					data[base+int(i)*int(seqLen)+int(j)] = 0
				} else {
					data[base+int(i)*int(seqLen)+int(j)] = negInf
				}
			}
		}
	}
	return FromValues(data, int(batchSize), 1, int(seqLen), int(seqLen))
}

func gemma4CombineMasks(base, extra *Array) *Array {
	if base == nil {
		return extra
	}
	if extra == nil {
		return base
	}
	combined := Minimum(base, extra)
	return combined
}

// Forward runs the Gemma 4 text model forward pass.
func (m *Gemma4Model) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

// ForwardMasked runs the forward pass with an explicit attention mask.
func (m *Gemma4Model) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	m.ensureCacheLayout()

	shape := tokens.Shape()
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)
	embeddingScale := float32(math.Sqrt(float64(m.Cfg.HiddenSize)))
	scaledH := MulScalar(h, embeddingScale)
	Free(h)
	h = scaledH

	perLayerInputs := m.computePerLayerInputs(tokens, h)
	defer Free(perLayerInputs...)

	var ownedMasks []*Array
	fullMask := mask
	slidingMask := mask
	if mask == nil {
		if L > 1 && m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
			slidingMask = buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
			ownedMasks = append(ownedMasks, slidingMask)
		}
	} else if m.Cfg.SlidingWindow > 0 && L > m.Cfg.SlidingWindow {
		windowMask := buildGemma4SlidingMask(B, L, m.Cfg.SlidingWindow)
		combined := gemma4CombineMasks(mask, windowMask)
		Free(windowMask)
		slidingMask = combined
		ownedMasks = append(ownedMasks, combined)
	}
	defer Free(ownedMasks...)

	intermediates := make([]sharedKV, len(m.Layers))
	for i, layer := range m.Layers {
		var prev sharedKV
		if prevIdx := m.PreviousKVs[i]; prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(intermediates)) {
			prev = intermediates[prevIdx]
		}

		var cache Cache
		if m.PreviousKVs[i] == int32(i) && i < len(m.CacheIndexByLayer) {
			if cacheIdx := m.CacheIndexByLayer[i]; cacheIdx >= 0 && int(cacheIdx) < len(caches) {
				cache = caches[cacheIdx]
			}
		}

		layerMask := fullMask
		if layer.IsSliding {
			layerMask = slidingMask
		}

		var pli *Array
		if len(perLayerInputs) > i {
			pli = perLayerInputs[i]
		}

		nextH, kv := layer.forward(h, cache, B, L, layerMask, pli, prev, m.Cfg)
		Free(h)
		h = nextH
		intermediates[i] = kv
	}
	defer func() {
		for i, kv := range intermediates {
			if m.PreviousKVs[i] != int32(i) {
				continue
			}
			Free(kv.Keys, kv.Values)
		}
	}()

	normed := RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		Free(out)
		out = softcapped
	}
	return out
}

func logitSoftcap(x *Array, softcap float32) *Array {
	scaled := MulScalar(x, 1.0/softcap)
	capped := Tanh(scaled)
	Free(scaled)
	out := MulScalar(capped, softcap)
	Free(capped)
	return out
}

func (l *Gemma4DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, cfg *Gemma4TextConfig) (*Array, sharedKV) {
	residual := x

	normed := RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut, kv := l.Attention.forward(normed, c, B, L, mask, prev, cfg)
	Free(normed)
	attnNormed := RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	Free(attnOut)
	h := Add(residual, attnNormed)
	Free(attnNormed)

	residual = h
	var ffResidual *Array
	if l.EnableMoE && l.Router != nil && l.Experts != nil {
		h1In := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		h1 := l.MLP.forward(h1In)
		Free(h1In)
		h1Normed := RMSNorm(h1, l.PostFFNorm1Scaled, cfg.RMSNormEps)
		Free(h1)

		h2In := RMSNorm(h, l.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := l.Router.forward(h2In)
		h2 := l.Experts.forward(h2In, topKIndices, topKWeights)
		Free(h2In, topKIndices, topKWeights)
		h2Normed := RMSNorm(h2, l.PostFFNorm2Scaled, cfg.RMSNormEps)
		Free(h2)

		// Gemma 4 MoE layers normalise each branch independently, then apply
		// the standard post-feedforward norm to the combined branch output
		// before adding it back to the residual path.
		combined := Add(h1Normed, h2Normed)
		Free(h1Normed, h2Normed)
		ffResidual = RMSNorm(combined, l.PostFFNormScaled, cfg.RMSNormEps)
		Free(combined)
	} else {
		ffIn := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		ff := l.MLP.forward(ffIn)
		Free(ffIn)
		ffResidual = RMSNorm(ff, l.PostFFNormScaled, cfg.RMSNormEps)
		Free(ff)
	}

	hNext := Add(residual, ffResidual)
	Free(h, ffResidual)

	if l.PerLayerInputGate != nil && l.PerLayerProjection != nil && l.PostPerLayerInputNormScaled != nil && perLayerInput != nil {
		gate := l.PerLayerInputGate.Forward(hNext)
		activated := getCompiledGELU().Call(gate)[0]
		Free(gate)
		multiplied := Mul(activated, perLayerInput)
		Free(activated)
		projected := l.PerLayerProjection.Forward(multiplied)
		Free(multiplied)
		projectedNormed := RMSNorm(projected, l.PostPerLayerInputNormScaled, cfg.RMSNormEps)
		Free(projected)
		gated := Add(hNext, projectedNormed)
		Free(hNext, projectedNormed)
		hNext = gated
	}

	if l.LayerScalar != nil && l.LayerScalar.Valid() {
		scaled := Mul(hNext, l.LayerScalar)
		Free(hNext)
		hNext = scaled
	}

	return hNext, kv
}

func (a *Gemma4Attention) applyRoPE(x *Array, offset int) *Array {
	if a.RopeFreqs != nil {
		return RoPEWithFreqs(x, int(a.HeadDim), false, 0, 1.0, offset, a.RopeFreqs)
	}
	return RoPE(x, int(a.RopeRotatedDim), false, a.RopeBase, 1.0, offset)
}

func (a *Gemma4Attention) forward(x *Array, c Cache, B, L int32, mask *Array, prev sharedKV, cfg *Gemma4TextConfig) (*Array, sharedKV) {
	qProj := a.QProj.Forward(x)
	q := AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, a.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * a.HeadDim), int64(a.HeadDim), int64(cfg.NumAttentionHeads * a.HeadDim), 1}, 0)
	Free(qProj)
	oldQ := q
	q = RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	Free(oldQ)

	kv := prev
	offset := 0
	if kv.Keys == nil || kv.Values == nil {
		kProj := a.KProj.Forward(x)
		k := AsStrided(kProj, []int32{B, a.NKVHeads, L, a.HeadDim},
			[]int64{int64(L * a.NKVHeads * a.HeadDim), int64(a.HeadDim), int64(a.NKVHeads * a.HeadDim), 1}, 0)
		Free(kProj)

		var v *Array
		if a.UseKEqV {
			v = k.Clone()
		} else {
			vProj := a.VProj.Forward(x)
			v = AsStrided(vProj, []int32{B, a.NKVHeads, L, a.HeadDim},
				[]int64{int64(L * a.NKVHeads * a.HeadDim), int64(a.HeadDim), int64(a.NKVHeads * a.HeadDim), 1}, 0)
			Free(vProj)
		}

		if c != nil {
			offset = c.Offset()
		}

		oldK := k
		k = RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)
		Free(oldK)
		kRoPE := a.applyRoPE(k, offset)
		Free(k)
		k = kRoPE

		vNormed := RMSNormNoScale(v, cfg.RMSNormEps)
		Free(v)
		v = vNormed

		if c != nil {
			oldK, oldV := k, v
			k, v = c.Update(k, v, int(L))
			Free(oldK, oldV)
		}
		kv = sharedKV{Keys: k, Values: v, Offset: offset}
	} else {
		offset = kv.Offset
	}

	qRoPE := a.applyRoPE(q, offset)
	Free(q)
	q = qRoPE

	repeatFactor := cfg.NumAttentionHeads / a.NKVHeads
	kAttn, vAttn := kv.Keys, kv.Values
	repeated := false
	if repeatFactor > 1 {
		kAttn = RepeatKV(kv.Keys, repeatFactor)
		vAttn = RepeatKV(kv.Values, repeatFactor)
		repeated = true
	}

	var out *Array
	if mask != nil {
		out = ScaledDotProductAttentionWithMask(q, kAttn, vAttn, mask, a.Scale)
	} else {
		out = ScaledDotProductAttention(q, kAttn, vAttn, a.Scale, L > 1)
	}
	Free(q)
	if repeated {
		Free(kAttn, vAttn)
	}

	transposed := Transpose(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*a.HeadDim)
	Free(transposed)
	result := a.OProj.Forward(reshaped)
	Free(reshaped)
	return result, kv
}

func (r *Gemma4Router) forward(x *Array) (*Array, *Array) {
	scaled := r.ScaleScaled
	if scaled == nil {
		scaled = MulScalar(r.Scale, r.RootSize)
		defer Free(scaled)
	}
	normed := RMSNorm(x, scaled, r.Eps)
	expertScores := r.Proj.Forward(normed)
	Free(normed)

	numExperts := expertScores.Dim(expertScores.NumDims() - 1)
	topK := int(r.TopK)
	if topK <= 0 || topK > numExperts {
		topK = numExperts
	}
	kth := numExperts - topK
	topKIndices := Argpartition(expertScores, kth, -1)
	sliced := SliceAxis(topKIndices, -1, int32(kth), int32(numExperts))
	Free(topKIndices)
	topKIndices = sliced

	topKWeights := TakeAlongAxis(expertScores, topKIndices, -1)
	Free(expertScores)
	topKWeightsSoftmax := Softmax(topKWeights)
	Free(topKWeights)
	if r.PerExpertScale == nil || !r.PerExpertScale.Valid() {
		return topKIndices, topKWeightsSoftmax
	}
	perExpertScale := Take(r.PerExpertScale, topKIndices, 0)
	weighted := Mul(topKWeightsSoftmax, perExpertScale)
	Free(topKWeightsSoftmax, perExpertScale)
	return topKIndices, weighted
}

func (e *Gemma4Experts) forward(x, topKIndices, topKWeights *Array) *Array {
	expanded1 := ExpandDims(x, 2)
	expanded := ExpandDims(expanded1, 2)
	Free(expanded1)

	up := e.UpProj.Forward(expanded, topKIndices)
	gate := e.GateProj.Forward(expanded, topKIndices)
	activatedGate := getCompiledGELU().Call(gate)[0]
	Free(gate)
	activated := Mul(activatedGate, up)
	Free(activatedGate, up)
	down := e.DownProj.Forward(activated, topKIndices)
	Free(activated)
	downSqueezed := Squeeze(down, 3)
	Free(down)

	weightsExpanded := ExpandDims(topKWeights, 3)
	weighted := Mul(weightsExpanded, downSqueezed)
	Free(weightsExpanded, downSqueezed)
	result := Sum(weighted, -2, false)
	Free(weighted)
	return result
}

// NewCache creates per-layer KV caches for Gemma 4.
func (m *Gemma4Model) NewCache() []Cache {
	m.ensureCacheLayout()

	numCaches := 0
	for _, cacheIdx := range m.CacheIndexByLayer {
		if cacheIdx >= 0 {
			numCaches++
		}
	}
	caches := make([]Cache, numCaches)
	for layerIdx, cacheIdx := range m.CacheIndexByLayer {
		if cacheIdx < 0 {
			continue
		}
		if m.Layers[layerIdx].LayerType == "full_attention" {
			caches[cacheIdx] = NewKVCache()
		} else {
			caches[cacheIdx] = NewRotatingKVCache(int(m.Cfg.SlidingWindow))
		}
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *Gemma4Model) NumLayers() int { return len(m.Layers) }

// Tokenizer returns the model's tokenizer.
func (m *Gemma4Model) Tokenizer() *Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *Gemma4Model) ModelType() string { return m.modelType }

// ApplyLoRA wraps target projection layers with LoRA adapters for training.
func (m *Gemma4Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	cfg = normalizeLoRAConfig(cfg)
	adapter := &LoRAAdapter{
		Layers: make(map[string]*LoRALinear),
		Config: cfg,
		Model:  m,
	}

	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *Linear
			var prefix string
			switch target {
			case "q_proj":
				prefix = core.Sprintf("model.layers.%d.self_attn", i)
				proj = layer.Attention.QProj
			case "k_proj":
				prefix = core.Sprintf("model.layers.%d.self_attn", i)
				proj = layer.Attention.KProj
			case "v_proj":
				prefix = core.Sprintf("model.layers.%d.self_attn", i)
				proj = layer.Attention.VProj
			case "o_proj":
				prefix = core.Sprintf("model.layers.%d.self_attn", i)
				proj = layer.Attention.OProj
			case "gate_proj":
				prefix = core.Sprintf("model.layers.%d.mlp", i)
				proj = layer.MLP.GateProj
			case "up_proj":
				prefix = core.Sprintf("model.layers.%d.mlp", i)
				proj = layer.MLP.UpProj
			case "down_proj":
				prefix = core.Sprintf("model.layers.%d.mlp", i)
				proj = layer.MLP.DownProj
			case "router.proj":
				prefix = core.Sprintf("model.layers.%d", i)
				if layer.Router != nil {
					proj = layer.Router.Proj
				}
			case "per_layer_input_gate":
				prefix = core.Sprintf("model.layers.%d", i)
				proj = layer.PerLayerInputGate
			case "per_layer_projection":
				prefix = core.Sprintf("model.layers.%d", i)
				proj = layer.PerLayerProjection
			}
			if proj != nil {
				lora := NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
				proj.LoRA = lora
				adapter.Layers[prefix+"."+target] = lora
			}
		}
	}

	return adapter
}
