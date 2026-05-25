// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"time"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

var enableCompiledGemma4PerLayerInputs = core.Env("GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS") == "1"

// GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS is a correctness-breaking diagnostic.
// It exists only to isolate the Gemma 4 per-layer input cost.
var disableGemma4PerLayerInputs = core.Env("GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS") == "1"

// gemma4PerLayerCombineScale is the constant 2**-0.5 (i.e. 1/sqrt(2))
// applied as the final scaling factor when combining the per-layer
// projected hidden with the per-layer input embedding inside
// perLayerInputTensor. Lifting the float32 narrowing here keeps the
// per-token forward pass free of math.Pow.
const gemma4PerLayerCombineScale float32 = 0.70710678118654752440

// Gemma4TextConfig holds Gemma 4 text model configuration.
type Gemma4TextConfig struct {
	ModelType                 string                `json:"model_type"`
	PadTokenID                int32                 `json:"pad_token_id"`
	ImageTokenID              int32                 `json:"image_token_id"`
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

	Quantization                *QuantizationConfig `json:"-"`
	VisionConfig                *Gemma4VisionConfig `json:"-"`
	LayerTypes                  []string            `json:"-"`
	EmbeddingScale              float32             `json:"-"` // Computed: sqrt(hidden_size); cached to skip per-token math.Sqrt
	PerLayerInputEmbeddingScale float32             `json:"-"` // Computed: sqrt(hidden_size_per_layer_input); cached to skip per-token math.Sqrt
	PerLayerProjectionScale     float32             `json:"-"` // Computed: 1/sqrt(hidden_size); cached to skip per-token math.Pow in perLayerInputTensor
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
	VisionTower         *Gemma4VisionModel
	MultiModalProjector *Gemma4MultiModalProjector
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

	compiledPerLayerInputs       *CompiledFunc
	compiledPerLayerInputsFailed bool
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

	compiledNativeOwnerDecode             *CompiledFunc
	compiledNativeSharedDecode            *CompiledFunc
	compiledNativeFixedOwnerDecode        *CompiledFunc
	compiledNativeFixedSharedDecode       *CompiledFunc
	compiledNativeFixedMaskedOwnerDecode  *CompiledFunc
	compiledNativeFixedMaskedSharedDecode *CompiledFunc
	compiledNativeOwnerFailed             bool
	compiledNativeSharedFailed            bool
	compiledNativeFixedOwnerFailed        bool
	compiledNativeFixedSharedFailed       bool
	compiledNativeFixedMaskedOwnerFailed  bool
	compiledNativeFixedMaskedSharedFailed bool
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
	GateUpProj *SwitchLinear
	GateProj   *SwitchLinear
	UpProj     *SwitchLinear
	DownProj   *SwitchLinear
}

type sharedKV struct {
	Keys     *Array
	Values   *Array
	Pages    PagedKVState
	Offset   int
	Fixed    bool
	Borrowed bool
}

func (kv sharedKV) hasState() bool {
	return (kv.Keys != nil && kv.Keys.Valid() && kv.Values != nil && kv.Values.Valid()) || kv.hasPages()
}

func (kv sharedKV) hasPages() bool {
	if len(kv.Pages.Keys) == 0 || len(kv.Pages.Keys) != len(kv.Pages.Values) {
		return false
	}
	for i := range kv.Pages.Keys {
		if kv.Pages.Keys[i] == nil || !kv.Pages.Keys[i].Valid() || kv.Pages.Values[i] == nil || !kv.Pages.Values[i].Valid() {
			return false
		}
	}
	return true
}

func (kv sharedKV) free() {
	if !kv.Borrowed {
		Free(kv.Keys, kv.Values)
	}
	kv.Pages.Free()
}

func (kv sharedKV) clone() sharedKV {
	out := sharedKV{
		Offset: kv.Offset,
		Fixed:  kv.Fixed,
	}
	if kv.Keys != nil && kv.Keys.Valid() {
		out.Keys = kv.Keys.Clone()
	}
	if kv.Values != nil && kv.Values.Valid() {
		out.Values = kv.Values.Clone()
	}
	out.Pages = clonePagedKVState(kv.Pages)
	return out
}

func moveSharedKV(kv *sharedKV) sharedKV {
	if kv == nil {
		return sharedKV{}
	}
	out := *kv
	*kv = sharedKV{}
	return out
}

func clonePagedKVState(state PagedKVState) PagedKVState {
	out := PagedKVState{Length: state.Length}
	if len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return out
	}
	out.Keys = make([]*Array, len(state.Keys))
	out.Values = make([]*Array, len(state.Values))
	out.Owned = make([]*Array, 0, len(state.Keys)+len(state.Values))
	for i := range state.Keys {
		if state.Keys[i] != nil && state.Keys[i].Valid() {
			out.Keys[i] = state.Keys[i].Clone()
			out.Owned = append(out.Owned, out.Keys[i])
		}
		if state.Values[i] != nil && state.Values[i].Valid() {
			out.Values[i] = state.Values[i].Clone()
			out.Owned = append(out.Owned, out.Values[i])
		}
	}
	return out
}

func gemma4ValidKV(k, v *Array) bool {
	return k != nil && k.Valid() && v != nil && v.Valid()
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
	if dst.PadTokenID == 0 && src.PadTokenID != 0 {
		dst.PadTokenID = src.PadTokenID
	}
	if dst.ImageTokenID == 0 && src.ImageTokenID != 0 {
		dst.ImageTokenID = src.ImageTokenID
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
		PadTokenID                *int32                `json:"pad_token_id"`
		ImageTokenID              *int32                `json:"image_token_id"`
		NumExperts                *int32                `json:"num_experts"`
		TopKExperts               *int32                `json:"top_k_experts"`
		MoEIntermediateSize       *int32                `json:"moe_intermediate_size"`
		SlidingWindow             *int32                `json:"sliding_window"`
		TieWordEmbeddings         *bool                 `json:"tie_word_embeddings"`
		RopeParameters            map[string]RopeParams `json:"rope_parameters"`
		VisionConfig              *Gemma4VisionConfig   `json:"vision_config"`
		TextConfig                struct {
			Gemma4TextConfig
			Quantization              *QuantizationConfig   `json:"quantization"`
			LayerTypes                []string              `json:"layer_types"`
			NumGlobalKeyValueHeads    *int32                `json:"num_global_key_value_heads"`
			NumKVSharedLayers         *int32                `json:"num_kv_shared_layers"`
			GlobalHeadDim             *int32                `json:"global_head_dim"`
			GlobalPartialRotaryFactor *float32              `json:"global_partial_rotary_factor"`
			HiddenSizePerLayerInput   *int32                `json:"hidden_size_per_layer_input"`
			PadTokenID                *int32                `json:"pad_token_id"`
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
	cfg.VisionConfig = normalizeGemma4VisionConfig(wrapper.VisionConfig)
	cfg.Quantization = wrapper.Quantization
	if cfg.Quantization == nil {
		cfg.Quantization = wrapper.TextConfig.Quantization
	}
	switch {
	case wrapper.PadTokenID != nil:
		cfg.PadTokenID = *wrapper.PadTokenID
	case wrapper.TextConfig.PadTokenID != nil:
		cfg.PadTokenID = *wrapper.TextConfig.PadTokenID
	}
	switch {
	case wrapper.ImageTokenID != nil:
		cfg.ImageTokenID = *wrapper.ImageTokenID
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
	if cfg.ImageTokenID == 0 {
		cfg.ImageTokenID = 258880
	}
	if cfg.VocabSizePerLayerInput == 0 {
		cfg.VocabSizePerLayerInput = cfg.VocabSize
	}
	if cfg.SlidingWindow == 0 {
		cfg.SlidingWindow = 512
	}
	if cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 6
	}
	if cfg.MaxPositionEmbeddings == 0 {
		cfg.MaxPositionEmbeddings = 131072
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
	if !cfg.UseDoubleWideMLP && wrapper.UseDoubleWideMLP == nil && wrapper.TextConfig.UseDoubleWideMLP == nil {
		cfg.UseDoubleWideMLP = true
	}
	if !cfg.TieWordEmbeddings && wrapper.TieWordEmbeddings == nil && wrapper.TextConfig.TieWordEmbeddings == nil {
		cfg.TieWordEmbeddings = true
	}
	if field := gemma4NegativeConfigField(&cfg); field != "" {
		return nil, core.E("gemma4.parseConfig", "negative "+field+" is invalid", nil)
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
	if len(cfg.LayerTypes) > 0 {
		cfg.LayerTypes[len(cfg.LayerTypes)-1] = "full_attention"
	}
	if len(cfg.LayerTypes) < int(cfg.NumHiddenLayers) {
		return nil, core.E("gemma4.parseConfig", "layer_types shorter than num_hidden_layers", nil)
	}
	cfg.LayerTypes = cfg.LayerTypes[:cfg.NumHiddenLayers]
	gemma4FinaliseEmbeddingScales(&cfg)
	return &cfg, nil
}

// gemma4FinaliseEmbeddingScales caches sqrt(HiddenSize),
// sqrt(HiddenSizePerLayerInput), and 1/sqrt(HiddenSize) on the config
// so per-token forward passes can skip the math.Sqrt/math.Pow + float32
// narrowing entirely. Safe to call multiple times — the loader
// re-invokes after inferring or resetting HiddenSizePerLayerInput from
// weights.
func gemma4FinaliseEmbeddingScales(cfg *Gemma4TextConfig) {
	if cfg == nil {
		return
	}
	if cfg.HiddenSize > 0 {
		cfg.EmbeddingScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
		cfg.PerLayerProjectionScale = float32(math.Pow(float64(cfg.HiddenSize), -0.5))
	} else {
		cfg.EmbeddingScale = 0
		cfg.PerLayerProjectionScale = 0
	}
	if cfg.HiddenSizePerLayerInput > 0 {
		cfg.PerLayerInputEmbeddingScale = float32(math.Sqrt(float64(cfg.HiddenSizePerLayerInput)))
	} else {
		cfg.PerLayerInputEmbeddingScale = 0
	}
}

func validateGemma4QuantizationConfig(q *QuantizationConfig) error {
	if q == nil {
		return nil
	}
	if q.GroupSize < 0 {
		return core.NewError("gemma4: quantization group_size must be >= 0")
	}
	if q.Bits < 0 {
		return core.NewError("gemma4: quantization bits must be >= 0")
	}
	mode := normalizeQuantizationMode(q.Mode)
	switch mode {
	case "affine":
		if q.Bits != 0 && q.Bits != 2 && q.Bits != 3 && q.Bits != 4 && q.Bits != 5 && q.Bits != 6 && q.Bits != 8 {
			return core.NewError(core.Sprintf("gemma4: affine quantization bits %d are unsupported", q.Bits))
		}
	case "mxfp4":
		if q.GroupSize != 0 && q.GroupSize != 32 {
			return core.NewError(core.Sprintf("gemma4: mxfp4 quantization requires group_size=32, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 4 {
			return core.NewError(core.Sprintf("gemma4: mxfp4 quantization requires bits=4, got %d", q.Bits))
		}
	case "mxfp8":
		if q.GroupSize != 0 && q.GroupSize != 32 {
			return core.NewError(core.Sprintf("gemma4: mxfp8 quantization requires group_size=32, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 8 {
			return core.NewError(core.Sprintf("gemma4: mxfp8 quantization requires bits=8, got %d", q.Bits))
		}
	case "nvfp4":
		if q.GroupSize != 0 && q.GroupSize != 16 {
			return core.NewError(core.Sprintf("gemma4: nvfp4 quantization requires group_size=16, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 4 {
			return core.NewError(core.Sprintf("gemma4: nvfp4 quantization requires bits=4, got %d", q.Bits))
		}
	default:
		return core.NewError(core.Sprintf("gemma4: unsupported quantization mode %q", q.Mode))
	}
	return nil
}

func gemma4NegativeConfigField(cfg *Gemma4TextConfig) string {
	checks := []struct {
		name  string
		value int32
	}{
		{"pad_token_id", cfg.PadTokenID},
		{"image_token_id", cfg.ImageTokenID},
		{"hidden_size", cfg.HiddenSize},
		{"num_hidden_layers", cfg.NumHiddenLayers},
		{"intermediate_size", cfg.IntermediateSize},
		{"num_attention_heads", cfg.NumAttentionHeads},
		{"num_key_value_heads", cfg.NumKeyValueHeads},
		{"head_dim", cfg.HeadDim},
		{"global_head_dim", cfg.GlobalHeadDim},
		{"vocab_size", cfg.VocabSize},
		{"vocab_size_per_layer_input", cfg.VocabSizePerLayerInput},
		{"sliding_window", cfg.SlidingWindow},
		{"sliding_window_pattern", cfg.SlidingWindowPattern},
		{"max_position_embeddings", cfg.MaxPositionEmbeddings},
		{"num_kv_shared_layers", cfg.NumKVSharedLayers},
		{"hidden_size_per_layer_input", cfg.HiddenSizePerLayerInput},
	}
	for _, check := range checks {
		if check.value < 0 {
			return check.name
		}
	}
	ptrChecks := []struct {
		name  string
		value *int32
	}{
		{"num_global_key_value_heads", cfg.NumGlobalKeyValueHeads},
		{"num_experts", cfg.NumExperts},
		{"top_k_experts", cfg.TopKExperts},
		{"moe_intermediate_size", cfg.MoEIntermediateSize},
	}
	for _, check := range ptrChecks {
		if check.value != nil && *check.value < 0 {
			return check.name
		}
	}
	return ""
}

func gemma4QuantPredicate(path string, defaultConfig *QuantizationConfig) *QuantizationConfig {
	if core.HasSuffix(path, "router.proj") {
		if defaultConfig != nil {
			q := *defaultConfig
			q.Mode = normalizeQuantizationMode(q.Mode)
			if isAffineQuantizationMode(q.Mode) {
				q.GroupSize = 64
				q.Bits = 8
			}
			return &q
		}
		return &QuantizationConfig{GroupSize: 64, Bits: 8}
	}
	if defaultConfig != nil {
		return defaultConfig
	}
	// When weights already carry quantization side tensors but config.json omits
	// the quantization block, let MLX use its affine defaults instead of
	// silently downgrading the layer to an incorrect dense projection.
	return &QuantizationConfig{}
}

func gemma4QuantForWeight(path string, defaultConfig *QuantizationConfig, weight, scales *Array) *QuantizationConfig {
	q := gemma4QuantPredicate(path, defaultConfig)
	if q == nil {
		return nil
	}
	resolved := *q
	resolved.Mode = normalizeQuantizationMode(resolved.Mode)
	if resolved.Mode == "mxfp4" && resolved.Bits == 0 {
		resolved.Bits = 4
	}
	if resolved.Mode == "mxfp8" && resolved.Bits == 0 {
		resolved.Bits = 8
	}
	if (resolved.Mode == "mxfp4" || resolved.Mode == "mxfp8") && resolved.GroupSize == 0 {
		resolved.GroupSize = 32
	}
	if resolved.Mode == "nvfp4" {
		if resolved.Bits == 0 {
			resolved.Bits = 4
		}
		if resolved.GroupSize == 0 {
			resolved.GroupSize = 16
		}
	}
	if !isAffineQuantizationMode(resolved.Mode) &&
		resolved.GroupSize > 0 &&
		inferGemma4QuantBits(weight, scales, resolved.GroupSize) == 0 {
		if inferred := inferGemma4QuantBits(weight, scales, 64); inferred > 0 {
			resolved.Mode = "affine"
			resolved.GroupSize = 64
			resolved.Bits = inferred
		}
	}
	if isAffineQuantizationMode(resolved.Mode) && resolved.GroupSize <= 0 && weight != nil && weight.Valid() && weight.Dtype() == DTypeUint32 {
		if inferred := inferGemma4QuantBits(weight, scales, 64); inferred > 0 {
			resolved.GroupSize = 64
			resolved.Bits = inferred
		}
	}
	if isAffineQuantizationMode(resolved.Mode) {
		if inferred := inferGemma4QuantBits(weight, scales, resolved.GroupSize); inferred > 0 {
			resolved.Bits = inferred
		}
	}
	return &resolved
}

func inferGemma4QuantBits(weight, scales *Array, groupSize int) int {
	if weight == nil || scales == nil || groupSize <= 0 || !weight.Valid() || !scales.Valid() {
		return 0
	}
	wShape := weight.Shape()
	sShape := scales.Shape()
	if len(wShape) == 0 || len(sShape) == 0 {
		return 0
	}
	weightCols := int(wShape[len(wShape)-1])
	scaleCols := int(sShape[len(sShape)-1])
	if weightCols <= 0 || scaleCols <= 0 {
		return 0
	}
	numerator := weightCols * 32
	denominator := scaleCols * groupSize
	if denominator <= 0 || numerator%denominator != 0 {
		return 0
	}
	bits := numerator / denominator
	switch bits {
	case 2, 3, 4, 5, 6, 8:
		return bits
	default:
		return 0
	}
}

func splitGemma4GateUpArray(a *Array) (*Array, *Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	var shapeBuf [maxTensorRank]int32
	shape := a.ShapeInto(shapeBuf[:0])
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
	var startsBuf, endsBuf [maxTensorRank]int32
	starts := startsBuf[:len(shape)]
	ends := endsBuf[:len(shape)]
	copy(ends, shape)
	ends[axis] = mid
	left := Slice(a, starts, ends)
	if !left.IsRowContiguous() {
		contiguous := Contiguous(left)
		Free(left)
		Materialize(contiguous)
		left = contiguous
	}
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := Slice(a, starts, ends)
	if !right.IsRowContiguous() {
		contiguous := Contiguous(right)
		Free(right)
		Materialize(contiguous)
		right = contiguous
	}
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
			if core.HasSuffix(canonical, ".experts.gate_up_proj"+suffix) {
				base := core.TrimSuffix(canonical, suffix)
				base = core.TrimSuffix(base, ".gate_up_proj")
				fused := base + ".switch_glu.gate_up_proj" + suffix
				if prev, ok := sanitized[fused]; ok && prev != arr {
					delete(retained, prev)
					discarded = append(discarded, prev)
				}
				sanitized[fused] = arr
				if arr != nil {
					retained[arr] = struct{}{}
				}
				gate, up, ok := splitGemma4GateUpArray(arr)
				if !ok {
					goto nextWeight
				}
				sanitized[base+".switch_glu.gate_proj"+suffix] = gate
				sanitized[base+".switch_glu.up_proj"+suffix] = up
				goto nextWeight
			}
			if core.HasSuffix(canonical, ".experts.down_proj"+suffix) {
				canonical = core.TrimSuffix(canonical, ".down_proj"+suffix) + ".switch_glu.down_proj" + suffix
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

func trimGemma4WrapperPrefix(name string) (string, bool) {
	for _, prefix := range []string{
		"model.language_model.model.",
		"model.language_model.",
		"language_model.model.",
		"language_model.",
		"model.model.",
		"model.",
	} {
		if core.HasPrefix(name, prefix) {
			return core.TrimPrefix(name, prefix), true
		}
	}
	return name, false
}

func canonicalGemma4WeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := trimGemma4WrapperPrefix(trimmed)
		if !changed {
			break
		}
		trimmed = next
	}

	if core.HasPrefix(trimmed, "vision_tower") ||
		core.HasPrefix(trimmed, "multi_modal_projector") ||
		core.HasPrefix(trimmed, "audio_tower") ||
		core.HasPrefix(trimmed, "embed_audio") ||
		core.HasPrefix(trimmed, "embed_vision") ||
		core.Contains(trimmed, "self_attn.rotary_emb") ||
		core.Contains(trimmed, "input_max") ||
		core.Contains(trimmed, "input_min") ||
		core.Contains(trimmed, "output_max") ||
		core.Contains(trimmed, "output_min") {
		return "", true
	}

	switch {
	case core.HasPrefix(trimmed, "layers."),
		core.HasPrefix(trimmed, "embed_tokens."),
		core.HasPrefix(trimmed, "embed_tokens_per_layer."),
		core.HasPrefix(trimmed, "norm."),
		core.HasPrefix(trimmed, "per_layer_model_projection."),
		core.HasPrefix(trimmed, "per_layer_projection_norm."):
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
		if q := gemma4QuantForWeight(prefix, defaultQ, weight, scales); q != nil {
			return newQuantizedLinearWithMode(weight, scales, biases, bias, q.GroupSize, q.Bits, q.Mode)
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
			if q := gemma4QuantForWeight(prefix, defaultQ, weight, scales); q != nil {
				return newQuantizedSwitchLinearWithMode(weight, scales, biases, bias, q.GroupSize, q.Bits, q.Mode)
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
		combined := concatenate2(freqs, inf, 0)
		Free(freqs, inf)
		freqs = combined
	}
	return freqs
}

func gemma4AttentionScale(headDim int32) float32 {
	return 1.0
}

func gemma4TrackArrays(retained map[*Array]struct{}, arrays ...*Array) {
	for _, arr := range arrays {
		if arr == nil || !arr.Valid() {
			continue
		}
		retained[arr] = struct{}{}
	}
}

func gemma4TrackEmbedding(retained map[*Array]struct{}, embedding *Embedding) {
	if embedding == nil {
		return
	}
	gemma4TrackArrays(retained, embedding.Weight, embedding.Scales, embedding.Biases)
}

func gemma4TrackLinear(retained map[*Array]struct{}, linear *Linear) {
	if linear == nil {
		return
	}
	gemma4TrackArrays(retained, linear.Weight, linear.Scales, linear.Biases, linear.Bias)
}

func gemma4TrackSwitchLinear(retained map[*Array]struct{}, linear *SwitchLinear) {
	if linear == nil {
		return
	}
	gemma4TrackArrays(retained, linear.Weight, linear.Scales, linear.Biases, linear.Bias)
}

func gemma4RetainedWeights(m *Gemma4Model) map[*Array]struct{} {
	retained := make(map[*Array]struct{})
	if m == nil {
		return retained
	}

	gemma4TrackEmbedding(retained, m.EmbedTokens)
	gemma4TrackEmbedding(retained, m.EmbedTokensPerLayer)
	gemma4TrackLinear(retained, m.PerLayerModelProj)
	gemma4TrackLinear(retained, m.Output)
	if m.Norm != nil {
		gemma4TrackArrays(retained, m.Norm.Weight)
	}
	if m.PerLayerProjNorm != nil {
		gemma4TrackArrays(retained, m.PerLayerProjNorm.Weight)
	}

	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		if layer.InputNorm != nil {
			gemma4TrackArrays(retained, layer.InputNorm.Weight)
		}
		if layer.PostAttnNorm != nil {
			gemma4TrackArrays(retained, layer.PostAttnNorm.Weight)
		}
		if layer.PreFFNorm != nil {
			gemma4TrackArrays(retained, layer.PreFFNorm.Weight)
		}
		if layer.PostFFNorm != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm.Weight)
		}
		if layer.PreFFNorm2 != nil {
			gemma4TrackArrays(retained, layer.PreFFNorm2.Weight)
		}
		if layer.PostFFNorm1 != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm1.Weight)
		}
		if layer.PostFFNorm2 != nil {
			gemma4TrackArrays(retained, layer.PostFFNorm2.Weight)
		}
		if layer.PostPerLayerInputNorm != nil {
			gemma4TrackArrays(retained, layer.PostPerLayerInputNorm.Weight)
		}
		gemma4TrackArrays(retained, layer.LayerScalar)
		gemma4TrackLinear(retained, layer.PerLayerInputGate)
		gemma4TrackLinear(retained, layer.PerLayerProjection)

		if attn := layer.Attention; attn != nil {
			gemma4TrackLinear(retained, attn.QProj)
			gemma4TrackLinear(retained, attn.KProj)
			gemma4TrackLinear(retained, attn.VProj)
			gemma4TrackLinear(retained, attn.OProj)
			if attn.QNorm != nil {
				gemma4TrackArrays(retained, attn.QNorm.Weight)
			}
			if attn.KNorm != nil {
				gemma4TrackArrays(retained, attn.KNorm.Weight)
			}
		}

		if mlp := layer.MLP; mlp != nil {
			gemma4TrackLinear(retained, mlp.GateProj)
			gemma4TrackLinear(retained, mlp.UpProj)
			gemma4TrackLinear(retained, mlp.DownProj)
		}

		if router := layer.Router; router != nil {
			gemma4TrackLinear(retained, router.Proj)
			gemma4TrackArrays(retained, router.Scale, router.PerExpertScale)
		}

		if experts := layer.Experts; experts != nil {
			gemma4TrackSwitchLinear(retained, experts.GateUpProj)
			gemma4TrackSwitchLinear(retained, experts.GateProj)
			gemma4TrackSwitchLinear(retained, experts.UpProj)
			gemma4TrackSwitchLinear(retained, experts.DownProj)
		}
	}

	return retained
}

func gemma4LazyRetainedWeights(m *Gemma4Model) map[*Array]struct{} {
	lazy := make(map[*Array]struct{})
	if m == nil {
		return lazy
	}
	gemma4TrackEmbedding(lazy, m.EmbedTokensPerLayer)
	return lazy
}

func gemma4FreeUnusedWeights(weights map[string]*Array, retained map[*Array]struct{}) {
	freed := make(map[*Array]struct{})
	for _, arr := range weights {
		if arr == nil || !arr.Valid() {
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
}

func gemma4MaterializableRetainedWeights(retained, lazy map[*Array]struct{}) []*Array {
	all := make([]*Array, 0, len(retained))
	for arr := range retained {
		if arr == nil || !arr.Valid() {
			continue
		}
		if _, ok := lazy[arr]; ok {
			continue
		}
		all = append(all, arr)
	}
	return all
}

func gemma4MaterializeRetainedWeights(retained, lazy map[*Array]struct{}) {
	all := gemma4MaterializableRetainedWeights(retained, lazy)
	Materialize(all...)
}

func precomputeGemma4ScaledWeights(m *Gemma4Model) {
	if m.Norm != nil {
		m.NormScaled = Copy(m.Norm.Weight)
	}
	if m.PerLayerProjNorm != nil && m.PerLayerProjNorm.Weight != nil {
		m.PerLayerProjNormScaled = Copy(m.PerLayerProjNorm.Weight)
	}

	var scaled []*Array
	scaled = append(scaled, m.NormScaled, m.PerLayerProjNormScaled)

	for _, layer := range m.Layers {
		if layer.InputNorm != nil && layer.InputNorm.Weight != nil {
			layer.InputNormScaled = Copy(layer.InputNorm.Weight)
		}
		if layer.PostAttnNorm != nil && layer.PostAttnNorm.Weight != nil {
			layer.PostAttnNormScaled = Copy(layer.PostAttnNorm.Weight)
		}
		if layer.PreFFNorm != nil && layer.PreFFNorm.Weight != nil {
			layer.PreFFNormScaled = Copy(layer.PreFFNorm.Weight)
		}
		if layer.PostFFNorm != nil && layer.PostFFNorm.Weight != nil {
			layer.PostFFNormScaled = Copy(layer.PostFFNorm.Weight)
		}
		if layer.PreFFNorm2 != nil && layer.PreFFNorm2.Weight != nil {
			layer.PreFFNorm2Scaled = Copy(layer.PreFFNorm2.Weight)
		}
		if layer.PostFFNorm1 != nil && layer.PostFFNorm1.Weight != nil {
			layer.PostFFNorm1Scaled = Copy(layer.PostFFNorm1.Weight)
		}
		if layer.PostFFNorm2 != nil && layer.PostFFNorm2.Weight != nil {
			layer.PostFFNorm2Scaled = Copy(layer.PostFFNorm2.Weight)
		}
		if layer.PostPerLayerInputNorm != nil && layer.PostPerLayerInputNorm.Weight != nil {
			layer.PostPerLayerInputNormScaled = Copy(layer.PostPerLayerInputNorm.Weight)
		}
		if layer.Attention != nil {
			if layer.Attention.QNorm != nil && layer.Attention.QNorm.Weight != nil {
				layer.Attention.QNormScaled = Copy(layer.Attention.QNorm.Weight)
			}
			if layer.Attention.KNorm != nil && layer.Attention.KNorm.Weight != nil {
				layer.Attention.KNormScaled = Copy(layer.Attention.KNorm.Weight)
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
	if err := validateGemma4QuantizationConfig(cfg.Quantization); err != nil {
		return nil, core.E("gemma4.LoadGemma4", "validate quantization", err)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load tokenizer", err)
	}

	rawWeights, err := loadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load weights", err)
	}
	visionWeights := sanitizeGemma4VisionWeights(rawWeights)
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
	// Re-cache once HiddenSizePerLayerInput is finalised against the
	// loaded weights — keeps cfg.PerLayerInputEmbeddingScale in sync.
	gemma4FinaliseEmbeddingScales(cfg)

	modelType := cfg.ModelType
	if modelType == "" {
		modelType = "gemma4_text"
	}

	embed := &Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens.weight")}
	if embedScales := gemma4WeightAny(weights, "model.embed_tokens.scales"); embedScales != nil {
		embed.Scales = embedScales
		embed.Biases = gemma4WeightAny(weights, "model.embed_tokens.biases")
		if q := gemma4QuantForWeight("model.embed_tokens", cfg.Quantization, embed.Weight, embedScales); q != nil {
			embed.GroupSize = q.GroupSize
			embed.Bits = q.Bits
			embed.QuantizationMode = q.Mode
		}
	}

	var embedPerLayer *Embedding
	if cfg.HiddenSizePerLayerInput > 0 {
		embedPerLayer = &Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight")}
		if scales := gemma4WeightAny(weights, "model.embed_tokens_per_layer.scales"); scales != nil {
			embedPerLayer.Scales = scales
			embedPerLayer.Biases = gemma4WeightAny(weights, "model.embed_tokens_per_layer.biases")
			if q := gemma4QuantForWeight("model.embed_tokens_per_layer", cfg.Quantization, embedPerLayer.Weight, scales); q != nil {
				embedPerLayer.GroupSize = q.GroupSize
				embedPerLayer.Bits = q.Bits
				embedPerLayer.QuantizationMode = q.Mode
			}
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
	loadSucceeded := false
	defer func() {
		if loadSucceeded {
			return
		}
		retained := gemma4RetainedWeights(m)
		gemma4FreeUnusedWeights(weights, retained)
		gemma4FreeUnusedWeights(visionWeights, retained)
		closeGemma4(m)
		ClearCache()
	}()

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
				GateUpProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.gate_up_proj",
					prefix+".experts.gate_up_proj",
				),
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

	if len(visionWeights) > 0 {
		m.VisionTower, m.MultiModalProjector, err = buildGemma4VisionComponents(cfg, visionWeights)
		if err != nil {
			return nil, core.E("gemma4.LoadGemma4", "build vision tower", err)
		}
	}

	m.PreviousKVs, m.CacheIndexByLayer = buildGemma4CacheLayout(m.Layers, cfg.NumKVSharedLayers)
	retainedWeights := gemma4RetainedWeights(m)
	lazyWeights := gemma4LazyRetainedWeights(m)
	gemma4FreeUnusedWeights(weights, retainedWeights)
	gemma4MaterializeRetainedWeights(retainedWeights, lazyWeights)
	precomputeGemma4ScaledWeights(m)

	loadSucceeded = true
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

	// Stack-allocated shape scratch — per-layer tensor reshape is in the
	// per-token decode path. Avoids the per-call []int32 heap alloc from
	// x.Shape() (24 B/op × NumHiddenLayers × tokens).
	var shapeBuf [maxTensorRank]int32
	shape := x.ShapeInto(shapeBuf[:0])
	switch len(shape) {
	case 4:
		if shape[2] == numLayers && shape[3] == hiddenSize {
			return x
		}
		if shape[2] == hiddenSize && shape[3] == numLayers {
			return Transpose4(x, 0, 1, 3, 2)
		}
	case 3:
		if shape[2] == numLayers*hiddenSize {
			return Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
		}
	}

	return Reshape(x, batchSize, seqLen, numLayers, hiddenSize)
}

func (m *Gemma4Model) computePerLayerInputs(tokens, hidden *Array) []*Array {
	// Stack-allocated shape scratch — per-token decode hot path. Calling
	// tokens.Shape() twice paid two []int32 heap allocs (24 B/op each).
	var tokShapeBuf [maxTensorRank]int32
	tokShape := tokens.ShapeInto(tokShapeBuf[:0])
	B, L := tokShape[0], tokShape[1]
	combined := m.computePerLayerInputTensor(tokens, hidden, B, L)
	return m.splitPerLayerInputTensor(combined)
}

func (m *Gemma4Model) computePerLayerInputTensor(tokens, hidden *Array, B, L int32) *Array {
	if disableGemma4PerLayerInputs {
		return nil
	}
	if m.EmbedTokensPerLayer == nil || m.PerLayerModelProj == nil || m.PerLayerProjNorm == nil || m.PerLayerProjNormScaled == nil {
		return nil
	}
	if combined, ok := m.compiledPerLayerInputTensor(tokens, hidden); ok {
		return combined
	}
	return m.perLayerInputTensor(tokens, hidden, B, L)
}

func (m *Gemma4Model) perLayerInputTensor(tokens, hidden *Array, B, L int32) *Array {
	perLayer := m.EmbedTokensPerLayer.Forward(tokens)
	scaled := MulScalar(perLayer, m.Cfg.PerLayerInputEmbeddingScale)
	Free(perLayer)
	perLayer = gemma4NormalizePerLayerTensor(scaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if perLayer != scaled {
		Free(scaled)
	}

	projected := m.PerLayerModelProj.Forward(hidden)
	projectedScaled := MulScalar(projected, m.Cfg.PerLayerProjectionScale)
	Free(projected)
	projected = gemma4NormalizePerLayerTensor(projectedScaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	if projected != projectedScaled {
		Free(projectedScaled)
	}
	projectedNormed := RMSNorm(projected, m.PerLayerProjNormScaled, m.Cfg.RMSNormEps)
	Free(projected)

	combined := Add(projectedNormed, perLayer)
	Free(projectedNormed, perLayer)
	combinedScaled := MulScalar(combined, gemma4PerLayerCombineScale)
	Free(combined)
	combined = combinedScaled
	return combined
}

func (m *Gemma4Model) splitPerLayerInputTensor(combined *Array) []*Array {
	if combined == nil || !combined.Valid() {
		return nil
	}
	defer Free(combined)

	perLayerInputs := make([]*Array, m.Cfg.NumHiddenLayers)
	var shapeBuf [maxTensorRank]int32
	shape := combined.ShapeInto(shapeBuf[:0])
	if len(shape) == 4 {
		for i := range m.Cfg.NumHiddenLayers {
			perLayerInputs[i] = m.perLayerInputForLayer(combined, shape[0], shape[1], i)
		}
		return perLayerInputs
	}

	// Generic fallback for malformed or legacy shapes. The normal Gemma 4 path
	// is rank-4 and should use the allocation-free Slice4/Reshape3 helper above.
	squeezeAxis2 := []int{2}
	for i := range m.Cfg.NumHiddenLayers {
		sliced := SliceAxis(combined, 2, i, i+1)
		perLayerInputs[i] = Squeeze(sliced, squeezeAxis2...)
		Free(sliced)
	}
	return perLayerInputs
}

func (m *Gemma4Model) perLayerInputForLayer(combined *Array, B, L, layer int32) *Array {
	if combined == nil || !combined.Valid() || layer < 0 || layer >= m.Cfg.NumHiddenLayers {
		return nil
	}
	if combined.NumDims() != 4 {
		sliced := SliceAxis(combined, 2, layer, layer+1)
		out := Reshape3(sliced, B, L, m.Cfg.HiddenSizePerLayerInput)
		Free(sliced)
		return out
	}
	sliced := Slice4(combined, 0, 0, layer, 0, B, L, layer+1, m.Cfg.HiddenSizePerLayerInput)
	out := Reshape3(sliced, B, L, m.Cfg.HiddenSizePerLayerInput)
	Free(sliced)
	return out
}

func (m *Gemma4Model) compiledPerLayerInputTensor(tokens, hidden *Array) (_ *Array, ok bool) {
	if !enableCompiledGemma4PerLayerInputs || m.compiledPerLayerInputsFailed {
		return nil, false
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			core.Error("mlx: compiled Gemma 4 per-layer inputs failed; falling back to Go graph", "error", recovered)
			m.compiledPerLayerInputsFailed = true
			if m.compiledPerLayerInputs != nil {
				m.compiledPerLayerInputs.Free()
				m.compiledPerLayerInputs = nil
			}
			ok = false
		}
	}()
	if m.compiledPerLayerInputs == nil || !m.compiledPerLayerInputs.Valid() {
		m.compiledPerLayerInputs = CompileShapeless(func(inputs []*Array) []*Array {
			if len(inputs) < 2 {
				return nil
			}
			shape := inputs[0].Shape()
			if len(shape) < 2 {
				return nil
			}
			out := m.perLayerInputTensor(inputs[0], inputs[1], shape[0], shape[1])
			return []*Array{out}
		}, true)
	}
	outs := m.compiledPerLayerInputs.Call(tokens, hidden)
	if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
		Free(outs...)
		m.compiledPerLayerInputsFailed = true
		return nil, false
	}
	return outs[0], true
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

func buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window int32) *Array {
	negInf := float32(math.Inf(-1))
	data := make([]float32, int(batchSize)*int(queryLen)*int(keyLen))
	for b := range batchSize {
		base := int(b) * int(queryLen) * int(keyLen)
		for i := range queryLen {
			queryPos := offset + i
			for j := range keyLen {
				keyPos := keyStart + j
				allowed := keyPos <= queryPos
				if window > 0 && allowed {
					allowed = queryPos-keyPos < window
				}
				if allowed {
					data[base+int(i)*int(keyLen)+int(j)] = 0
				} else {
					data[base+int(i)*int(keyLen)+int(j)] = negInf
				}
			}
		}
	}
	return FromValues(data, int(batchSize), 1, int(queryLen), int(keyLen))
}

type gemma4CachedAttentionMaskKey struct {
	batchSize int32
	queryLen  int32
	keyLen    int32
	offset    int32
	keyStart  int32
	window    int32
}

type gemma4RuntimeMaskCache struct {
	masks map[gemma4CachedAttentionMaskKey]*Array
	owned []*Array
}

func newGemma4RuntimeMaskCache() *gemma4RuntimeMaskCache {
	return &gemma4RuntimeMaskCache{}
}

func (c *gemma4RuntimeMaskCache) CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window int32) *Array {
	if c == nil {
		return buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window)
	}
	key := gemma4CachedAttentionMaskKey{
		batchSize: batchSize,
		queryLen:  queryLen,
		keyLen:    keyLen,
		offset:    offset,
		keyStart:  keyStart,
		window:    window,
	}
	if c.masks == nil {
		c.masks = make(map[gemma4CachedAttentionMaskKey]*Array)
	}
	if mask := c.masks[key]; mask != nil && mask.Valid() {
		return mask
	}
	mask := buildGemma4CachedAttentionMask(batchSize, queryLen, keyLen, offset, keyStart, window)
	if mask == nil || !mask.Valid() {
		Free(mask)
		return nil
	}
	c.masks[key] = mask
	c.owned = append(c.owned, mask)
	return mask
}

func (c *gemma4RuntimeMaskCache) Free() {
	if c == nil {
		return
	}
	Free(c.owned...)
	c.owned = nil
	c.masks = nil
}

func gemma4CanUseOffsetCausalAttention(queryLen, keyLen, window int32) bool {
	if queryLen <= 1 || keyLen <= 0 {
		return false
	}
	if window <= 0 {
		return true
	}
	return queryLen <= window && keyLen <= window+queryLen-1
}

func gemma4SlidingCausalContextLen(queryLen, keyLen, window int32) int {
	if queryLen <= 1 || keyLen <= 0 || window <= 0 || queryLen > window {
		return int(keyLen)
	}
	needed := window + queryLen - 1
	if needed >= keyLen {
		return int(keyLen)
	}
	return int(needed)
}

func fixedSingleTokenCausalMaskFromHost(batchSize int32, capacity, offset int) *Array {
	if batchSize <= 0 || capacity <= 0 {
		return nil
	}
	data := make([]float32, int(batchSize)*capacity)
	for b := range int(batchSize) {
		base := b * capacity
		for i := range capacity {
			if i > offset {
				data[base+i] = -1e9
			}
		}
	}
	return FromValues(data, int(batchSize), 1, 1, capacity)
}

type fixedGemma4AttentionMaskSet struct {
	batchSize int32
	seqLen    int32
	disabled  bool
	masks     map[fixedGemma4AttentionMaskKey]*Array
	owned     []*Array
}

type fixedGemma4AttentionMaskKey struct {
	capacity int
	offset   int
}

func newFixedGemma4AttentionMaskSet(batchSize, seqLen int32, mask *Array) *fixedGemma4AttentionMaskSet {
	return &fixedGemma4AttentionMaskSet{
		batchSize: batchSize,
		seqLen:    seqLen,
		disabled:  !fixedGemma4SharedMaskEnabled() || mask != nil || seqLen != 1,
	}
}

func (s *fixedGemma4AttentionMaskSet) ForLayer(cache Cache, prev sharedKV) *Array {
	if s == nil || s.disabled {
		return nil
	}
	capacity, offset, ok := fixedGemma4AttentionMaskCapacityOffset(cache, prev, s.seqLen)
	if !ok {
		return nil
	}
	key := fixedGemma4AttentionMaskKey{capacity: capacity, offset: offset}
	if s.masks == nil {
		s.masks = make(map[fixedGemma4AttentionMaskKey]*Array)
	}
	if mask := s.masks[key]; mask != nil && mask.Valid() {
		return mask
	}
	mask := fixedSingleTokenCausalMaskFromHost(s.batchSize, capacity, offset)
	if mask == nil || !mask.Valid() {
		Free(mask)
		return nil
	}
	s.masks[key] = mask
	s.owned = append(s.owned, mask)
	return mask
}

func (s *fixedGemma4AttentionMaskSet) Free() {
	if s == nil {
		return
	}
	Free(s.owned...)
	s.owned = nil
	s.masks = nil
}

func fixedGemma4AttentionMaskCapacityOffset(cache Cache, prev sharedKV, seqLen int32) (int, int, bool) {
	if seqLen != 1 {
		return 0, 0, false
	}
	if fixed, ok := cache.(*FixedKVCache); ok && fixed != nil && fixed.maxSize > 0 {
		offset := fixed.Offset()
		if offset >= 0 && offset+int(seqLen) <= fixed.maxSize {
			return fixed.maxSize, offset, true
		}
		return 0, 0, false
	}
	if prev.Fixed && prev.Keys != nil && prev.Keys.Valid() && prev.Keys.NumDims() == 4 {
		capacity := int(prev.Keys.Dim(2))
		offset := prev.Offset
		if capacity > 0 && offset >= 0 && offset+int(seqLen) <= capacity {
			return capacity, offset, true
		}
	}
	return 0, 0, false
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
	h, _, _ := m.forwardHidden(tokens, mask, caches)
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

// ForwardLastTokenLogits runs prefill while projecting only the final sequence
// position. Long local-context warmup needs KV cache updates for every token,
// but generation only consumes logits from the last token; avoiding full
// [sequence, vocab] logits keeps Gemma 4 prefill inside Apple memory limits.
func (m *Gemma4Model) ForwardLastTokenLogits(tokens *Array, mask *Array, caches []Cache) *Array {
	out, hidden := m.ForwardLastTokenLogitsAndHidden(tokens, mask, caches)
	Free(hidden)
	return out
}

// ForwardLastTokenLogitsAndHidden runs prefill while returning both final
// position logits and the corresponding target hidden state before output
// normalisation. The hidden state is the seed consumed by attached MTP
// assistants.
func (m *Gemma4Model) ForwardLastTokenLogitsAndHidden(tokens *Array, mask *Array, caches []Cache) (*Array, *Array) {
	h, _, L := m.forwardHidden(tokens, mask, caches)
	h = gemma4LastSequenceHidden(h, L)
	h = gemma4ProjectionHidden(h)
	h = gemma4ContiguousHidden(h)
	if gemma4PreferNativeLastTokenOutputLogits(m.Output) {
		if out, ok, err := nativeLastTokenOutputLogits(h, m.NormScaled, m.Output, m.Cfg.RMSNormEps, m.Cfg.FinalLogitSoftcapping); ok {
			if err == nil {
				return out, h
			}
			core.Error("mlx: native Gemma 4 last-token output failed; falling back to Go graph", "error", err)
		}
	}
	return m.forwardLastTokenOutputGraph(h), h
}

func gemma4PreferNativeLastTokenOutputLogits(output *Linear) bool {
	if output == nil {
		return false
	}
	if output.Scales != nil {
		return false
	}
	return true
}

func (m *Gemma4Model) forwardLastTokenOutputGraph(h *Array) *Array {
	if m == nil || m.Cfg == nil {
		return nil
	}
	normed := RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(normed)
	if m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, m.Cfg.FinalLogitSoftcapping)
		Free(out)
		out = softcapped
	}
	return out
}

// ForwardGreedyToken runs a forward pass and returns the greedy next token
// directly. Final logit softcapping is monotonic, so greedy selection can skip
// materialising a softcapped logits tensor.
func (m *Gemma4Model) ForwardGreedyToken(tokens *Array, mask *Array, caches []Cache) *Array {
	return m.forwardGreedyToken(tokens, mask, caches, nil)
}

// ForwardGreedyTokenWithSuppression runs the same greedy decode path while
// masking chat-template and modality token IDs before argmax.
func (m *Gemma4Model) ForwardGreedyTokenWithSuppression(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32) *Array {
	return m.forwardGreedyTokenWithSuppressionArray(tokens, mask, caches, suppressTokens, nil)
}

func (m *Gemma4Model) forwardGreedyToken(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32) *Array {
	return m.forwardGreedyTokenWithSuppressionArray(tokens, mask, caches, suppressTokens, nil)
}

func (m *Gemma4Model) forwardGreedyTokenWithSuppressionArray(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32, suppress *Array) *Array {
	if out, ok, err := m.forwardNativeFixedGreedyToken(tokens, mask, caches, suppress, suppressTokens); ok {
		if err == nil {
			traceNativeMaterialize("gemma4.model.greedy_token", out)
			return out
		}
		core.Error("mlx: native Gemma 4 model greedy token failed; falling back to Go graph", "error", err)
	}
	h, _, L := m.forwardHidden(tokens, mask, caches)
	h = gemma4LastSequenceHidden(h, L)
	h = gemma4ProjectionHidden(h)
	h = gemma4ContiguousHidden(h)
	if out, ok, err := nativeLastTokenGreedyTokenWithArray(h, m.NormScaled, m.Output, m.Cfg.RMSNormEps, suppress, suppressTokens...); ok {
		if err == nil {
			Free(h)
			return out
		}
		core.Error("mlx: native Gemma 4 greedy token failed; falling back to Go graph", "error", err)
	}
	normed := RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	logits := m.Output.Forward(normed)
	var out *Array
	if len(suppressTokens) > 0 {
		var err error
		sampler := newSamplerWithSuppression(0, 0, 0, 0, suppressTokens)
		out, err = sampleTokenWithSuppressionGuard(logits, sampler, suppressTokens)
		closeSampler(sampler)
		if err != nil {
			core.Error("mlx: Gemma 4 suppressed greedy fallback failed; falling back to unsuppressed argmax", "error", err)
			Free(out)
			out = Argmax(logits, -1, false)
		}
	} else {
		out = Argmax(logits, -1, false)
	}
	Free(h, normed, logits)
	return out
}

func (m *Gemma4Model) forwardNativeFixedGreedyToken(tokens *Array, mask *Array, caches []Cache, suppress *Array, suppressTokens []int32) (*Array, bool, error) {
	if !nativeGemma4ModelGreedyEnabled() || mask != nil || tokens == nil || !tokens.Valid() {
		return nil, false, nil
	}
	m.ensureCacheLayout()
	// Stack-allocated shape scratch — native fixed greedy single-token decode
	// hot path. Avoids the per-call []int32 heap alloc.
	var shapeBuf [maxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	if len(shape) != 2 || shape[0] <= 0 || shape[1] != 1 {
		return nil, false, nil
	}

	h := m.EmbedTokens.Forward(tokens)
	scaledH := MulScalar(h, m.Cfg.EmbeddingScale)
	Free(h)
	h = scaledH
	defer Free(h)

	perLayerInputs := m.computePerLayerInputs(tokens, h)
	defer Free(perLayerInputs...)
	fixedMasks := newFixedGemma4AttentionMaskSet(shape[0], shape[1], nil)
	defer fixedMasks.Free()

	return nativeGemma4FixedGreedyTokenWithArray(h, perLayerInputs, caches, m, fixedMasks, suppress, suppressTokens...)
}

func gemma4LastSequenceHidden(h *Array, seqLen int32) *Array {
	if h == nil || !h.Valid() || seqLen <= 1 {
		return h
	}
	ndim := h.NumDims()
	var axis int
	switch {
	case ndim >= 3:
		axis = ndim - 2
	case ndim == 2:
		axis = 0
	default:
		return h
	}
	dim := h.Dim(axis)
	if dim <= 1 {
		return h
	}
	start := int32(dim - 1)
	if seqLen > 0 && seqLen <= int32(dim) {
		start = seqLen - 1
	}
	last := SliceAxis(h, axis, start, start+1)
	Free(h)
	return last
}

func gemma4ProjectionHidden(h *Array) *Array {
	if h == nil || !h.Valid() {
		return h
	}
	switch h.NumDims() {
	case 1:
		out := Reshape(h, 1, 1, int32(h.Dim(0)))
		Free(h)
		return out
	case 2:
		out := Reshape(h, 1, int32(h.Dim(0)), int32(h.Dim(1)))
		Free(h)
		return out
	default:
		return h
	}
}

func gemma4ContiguousHidden(h *Array) *Array {
	if h == nil || !h.Valid() || h.IsRowContiguous() {
		return h
	}
	out := Contiguous(h)
	Free(h)
	return out
}

func (m *Gemma4Model) forwardHidden(tokens *Array, mask *Array, caches []Cache) (*Array, int32, int32) {
	m.ensureCacheLayout()

	// Stack-allocated shape scratch — per-forward-pass hot path. Avoids
	// the per-call []int32 heap alloc from tokens.Shape().
	var shapeBuf [maxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)
	scaledH := MulScalar(h, m.Cfg.EmbeddingScale)
	Free(h)
	h = scaledH

	perLayerInputTensor := m.computePerLayerInputTensor(tokens, h, B, L)
	defer Free(perLayerInputTensor)

	var ownedMasks []*Array
	var runtimeMasks *gemma4RuntimeMaskCache
	if L > 1 {
		runtimeMasks = newGemma4RuntimeMaskCache()
		defer runtimeMasks.Free()
	}
	fixedMasks := newFixedGemma4AttentionMaskSet(B, L, mask)
	defer fixedMasks.Free()
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

	var stackIntermediates [64]sharedKV
	var intermediates []sharedKV
	var stackSharedSources [64]bool
	var sharedSources []bool
	if len(m.Layers) <= len(stackIntermediates) {
		intermediates = stackIntermediates[:len(m.Layers)]
		sharedSources = stackSharedSources[:len(m.Layers)]
	} else {
		intermediates = make([]sharedKV, len(m.Layers))
		sharedSources = make([]bool, len(m.Layers))
	}
	for i, prevIdx := range m.PreviousKVs {
		if i >= len(sharedSources) {
			break
		}
		if prevIdx != int32(i) && prevIdx >= 0 && prevIdx < int32(len(sharedSources)) {
			sharedSources[prevIdx] = true
		}
	}
	defer func() {
		for _, kv := range intermediates {
			kv.free()
		}
	}()
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

		pli := m.perLayerInputForLayer(perLayerInputTensor, B, L, int32(i))

		fixedMask := fixedMasks.ForLayer(cache, prev)
		prevAvailable := prev.hasState()
		materializePagedKVForReuse := m.PreviousKVs[i] == int32(i) && sharedSources[i]
		nextH, kv := layer.forward(h, cache, B, L, layerMask, pli, prev, m.Cfg, fixedMask, runtimeMasks, materializePagedKVForReuse)
		Free(pli)
		Free(h)
		h = nextH
		if m.PreviousKVs[i] == int32(i) || !prevAvailable {
			if sharedSources[i] {
				intermediates[i] = moveSharedKV(&kv)
			}
			kv.free()
		}
	}
	return h, B, L
}

func logitSoftcap(x *Array, softcap float32) *Array {
	scaled := MulScalar(x, 1.0/softcap)
	capped := Tanh(scaled)
	Free(scaled)
	out := MulScalar(capped, softcap)
	Free(capped)
	return out
}

func (l *Gemma4DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, cfg *Gemma4TextConfig, fixedMask *Array, runtimeMasks *gemma4RuntimeMaskCache, materializePagedKVForReuse bool) (*Array, sharedKV) {
	defer func() {
		if recovered := recover(); recovered != nil {
			panic(core.Sprintf("Gemma 4 layer %d %s: %v", l.LayerIdx, l.LayerType, recovered))
		}
	}()
	traceEnabled := nativePhaseMaterializeTraceEnabled() && nativePhaseTraceArmed()
	if out, kv, ok, err := compiledGemma4DecodeLayer(x, c, B, L, mask, perLayerInput, prev, l, cfg, fixedMask); ok {
		if err == nil {
			l.traceNativeMaterialize(traceEnabled, "compiled_layer", out)
			return out, kv
		}
		core.Error("mlx: compiled Gemma 4 decode layer failed; falling back to Go graph", "layer", l.LayerIdx, "type", l.LayerType, "error", err)
	}
	if out, kv, ok, err := nativeGemma4DecodeLayer(x, c, B, L, mask, perLayerInput, prev, l, cfg, fixedMask); ok {
		if err == nil {
			l.traceNativeMaterialize(traceEnabled, "native_layer", out)
			return out, kv
		}
		core.Error("mlx: native Gemma 4 decode layer failed; falling back to Go graph", "layer", l.LayerIdx, "type", l.LayerType, "error", err)
	}

	residual := x

	normed := RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	window := int32(0)
	if l.IsSliding {
		window = cfg.SlidingWindow
	}
	var h *Array
	var kv sharedKV
	if nativeGemma4FixedOwnerAttentionResidualEnabled() && !l.IsSliding && !prev.hasState() && L == 1 && mask == nil {
		if fixed, ok := c.(*FixedKVCache); ok {
			if nativeH, nativeKV, ok, err := nativeGemma4FixedOwnerAttentionResidualBlock(residual, normed, fixed, fixedMask, l.Attention, l.PostAttnNormScaled, cfg); ok {
				h = nativeH
				kv = nativeKV
				l.traceNativeMaterialize(traceEnabled, "attention_residual", h)
			} else if err != nil {
				core.Error("mlx: native Gemma 4 fixed owner attention residual failed; falling back to Go graph", "error", err)
			}
		}
	}
	if h == nil {
		attnOut, nativeKV := l.Attention.forward(normed, c, B, L, mask, prev, cfg, window, fixedMask, runtimeMasks, materializePagedKVForReuse)
		kv = nativeKV
		l.traceNativeMaterialize(traceEnabled, "attention", attnOut)
		if nativeGemma4ResidualNormEnabled() {
			if nativeH, ok, err := nativeResidualNormAdd(residual, attnOut, l.PostAttnNormScaled, cfg.RMSNormEps); ok {
				h = nativeH
			} else if err != nil {
				core.Error("mlx: native Gemma 4 attention residual failed; falling back to Go graph", "error", err)
			}
		}
		if h == nil {
			attnNormed := RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
			h = Add(residual, attnNormed)
			Free(attnNormed)
		}
		Free(attnOut)
		l.traceNativeMaterialize(traceEnabled, "attention_residual", h)
	}
	Free(normed)

	residual = h
	var ffResidual *Array
	var hNext *Array
	if l.EnableMoE && l.Router != nil && l.Experts != nil {
		h1In := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		h1 := l.MLP.forward(h1In)
		l.traceNativeMaterialize(traceEnabled, "ffn_local_mlp", h1)
		Free(h1In)

		h2In := RMSNorm(h, l.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := l.Router.forward(h)
		l.traceNativeMaterialize(traceEnabled, "ffn_router", topKIndices, topKWeights)
		expertTracePrefix := ""
		if traceEnabled {
			expertTracePrefix = l.nativeTraceName("ffn_expert")
		}
		h2 := l.Experts.forward(h2In, topKIndices, topKWeights, expertTracePrefix)
		l.traceNativeMaterialize(traceEnabled, "ffn_experts", h2)
		Free(h2In, topKIndices, topKWeights)

		if nativeOut, ok, err := nativeGemma4FFNResidual(residual, h1, h2, l.PostFFNorm1Scaled, l.PostFFNorm2Scaled, l.PostFFNormScaled, cfg.RMSNormEps); ok {
			if err == nil {
				hNext = nativeOut
				l.traceNativeMaterialize(traceEnabled, "ffn_residual", hNext)
			} else {
				core.Error("mlx: native Gemma 4 FFN residual failed; falling back to Go graph", "error", err)
			}
		}
		if hNext == nil {
			h1Normed := RMSNorm(h1, l.PostFFNorm1Scaled, cfg.RMSNormEps)
			l.traceNativeMaterialize(traceEnabled, "ffn_local_norm", h1Normed)
			h2Normed := RMSNorm(h2, l.PostFFNorm2Scaled, cfg.RMSNormEps)
			l.traceNativeMaterialize(traceEnabled, "ffn_expert_norm", h2Normed)

			// Gemma 4 MoE layers normalise each branch independently, then apply
			// the standard post-feedforward norm to the combined branch output
			// before adding it back to the residual path.
			combined := Add(h1Normed, h2Normed)
			Free(h1Normed, h2Normed)
			ffResidual = RMSNorm(combined, l.PostFFNormScaled, cfg.RMSNormEps)
			Free(combined)
		}
		Free(h1, h2)
	} else {
		ffIn := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		ff := l.MLP.forward(ffIn)
		Free(ffIn)
		ffResidual = RMSNorm(ff, l.PostFFNormScaled, cfg.RMSNormEps)
		Free(ff)
	}
	if ffResidual != nil {
		l.traceNativeMaterialize(traceEnabled, "ffn", ffResidual)
	}

	if hNext == nil {
		hNext = Add(residual, ffResidual)
		Free(ffResidual)
	}
	Free(h)

	if l.PerLayerInputGate != nil && l.PerLayerProjection != nil && l.PostPerLayerInputNormScaled != nil && perLayerInput != nil {
		gate := l.PerLayerInputGate.Forward(hNext)
		multiplied := geluGateMul(gate, perLayerInput)
		Free(gate)
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
	l.traceNativeMaterialize(traceEnabled, "output", hNext)

	return hNext, kv
}

func (l *Gemma4DecoderLayer) traceNativeMaterialize(enabled bool, phase string, arrays ...*Array) {
	if !enabled {
		return
	}
	traceNativeMaterialize(l.nativeTraceName(phase), arrays...)
}

func gemma4AttentionWindowTraceName(window int32) string {
	if window > 0 {
		return "local"
	}
	return "global"
}

func tracePagedKVConcat(name string, start time.Time, state PagedKVState) {
	if !nativePhaseTraceArmed() || name == "" || start.IsZero() {
		return
	}
	duration := time.Since(start)
	if duration <= 0 {
		duration = time.Nanosecond
	}
	appendNativePhaseTraceEvent(NativePhaseTrace{
		Name:     name,
		Duration: duration,
		Pages:    len(state.Keys),
		Tokens:   state.Length,
	})
}

func (l *Gemma4DecoderLayer) nativeTraceName(phase string) string {
	return core.Sprintf("gemma4.layer.%02d.%s", l.LayerIdx, phase)
}

func (a *Gemma4Attention) applyRoPE(x *Array, offset int) *Array {
	if a.RopeFreqs != nil {
		return RoPEWithFreqs(x, int(a.HeadDim), false, 0, 1.0, offset, a.RopeFreqs)
	}
	return RoPE(x, int(a.RopeRotatedDim), false, a.RopeBase, 1.0, offset)
}

func attentionQueryForKV(query, key *Array) (*Array, *Array) {
	if query == nil || key == nil || !query.Valid() || !key.Valid() {
		return query, nil
	}
	dtype := key.Dtype()
	if query.Dtype() == dtype {
		return query, nil
	}
	switch dtype {
	case DTypeFloat16, DTypeBFloat16:
		cast := AsType(query, dtype)
		return cast, cast
	default:
		return query, nil
	}
}

func (a *Gemma4Attention) forward(x *Array, c Cache, B, L int32, mask *Array, prev sharedKV, cfg *Gemma4TextConfig, window int32, fixedMask *Array, runtimeMasks *gemma4RuntimeMaskCache, materializePagedKVForReuse bool) (*Array, sharedKV) {
	if nativeGemma4FixedOwnerAttentionEnabled() && window == 0 && !prev.hasState() && L == 1 && mask == nil {
		if fixed, ok := c.(*FixedKVCache); ok {
			if out, kv, ok, err := nativeGemma4FixedOwnerAttentionBlock(x, fixed, fixedMask, a, cfg); ok {
				return out, kv
			} else if err != nil {
				core.Error("mlx: native Gemma 4 fixed owner attention failed; falling back to Go graph", "error", err)
			}
		}
	}

	qProj := a.QProj.Forward(x)
	q := AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, a.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * a.HeadDim), int64(a.HeadDim), int64(cfg.NumAttentionHeads * a.HeadDim), 1}, 0)
	Free(qProj)
	oldQ := q
	q = RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	Free(oldQ)

	kv := prev
	offset := 0
	var out *Array
	qRoPEApplied := false
	if !kv.hasState() {
		kProj := a.KProj.Forward(x)
		k := AsStrided(kProj, []int32{B, a.NKVHeads, L, a.HeadDim},
			[]int64{int64(L * a.NKVHeads * a.HeadDim), int64(a.HeadDim), int64(a.NKVHeads * a.HeadDim), 1}, 0)
		Free(kProj)

		var v *Array
		if a.UseKEqV {
			// Gemma 4 K=V shares the projection source, not the final cache
			// tensors: K still takes KNorm+RoPE, while V takes value RMSNorm.
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
			if fixed, ok := c.(*FixedKVCache); ok && L == 1 && mask == nil && fixed.maxSize > 0 {
				// Stack-allocated shape scratch — per-token per-layer hot path.
				// K/V are always rank-4 ([B,H,L,D]); avoids 2 × []int32 heap
				// allocs per layer per token (× NumHiddenLayers).
				var kShapeBuf, vShapeBuf [maxTensorRank]int32
				kShape := k.ShapeInto(kShapeBuf[:0])
				vShape := v.ShapeInto(vShapeBuf[:0])
				fixed.ensureShape(kShape[0], kShape[1], kShape[3], vShape[3], k.Dtype(), v.Dtype())
				state := fixed.BorrowedFixedState()
				if state.Keys != nil && state.Values != nil {
					qRoPE := a.applyRoPE(q, offset)
					Free(q)
					q = qRoPE
					qRoPEApplied = true

					var nativeOut, nativeKeys, nativeValues *Array
					var ok bool
					var err error
					var offsetArray *Array
					if fixed.Offset()+int(L) <= fixed.maxSize {
						offsetArray = FromValue(offset)
						nativeOut, nativeKeys, nativeValues, ok, err = nativeFixedSingleTokenAttention(q, state.Keys, state.Values, k, v, offsetArray, nil, a.Scale)
					} else if nativeFixedSlidingAttentionEnabled() && fixed.length >= fixed.maxSize {
						shiftIndices, lastIndex := fixed.slidingUpdateInputs()
						nativeOut, nativeKeys, nativeValues, ok, err = nativeFixedSlidingSingleTokenAttention(q, state.Keys, state.Values, k, v, shiftIndices, lastIndex, a.Scale)
					}
					if err != nil {
						core.Error("mlx: native fixed owner attention failed; falling back to Go graph", "error", err)
						Free(nativeOut, nativeKeys, nativeValues)
						nativeOut, nativeKeys, nativeValues = nil, nil, nil
						ok = false
					}
					if ok {
						if err := validateGemma4LayerOutputShapes("mlx.nativeFixedSingleTokenAttention", q, nativeOut, nativeKeys, nativeValues, state.Keys, state.Values, true, true); err == nil {
							fixedState := fixed.ReplaceFixedFromNativeBorrowed(nativeKeys, nativeValues, int(L))
							if gemma4ValidKV(fixedState.Keys, fixedState.Values) {
								kv = sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}
								out = nativeOut
								fixed.RetireAfterNextEval(oldK, oldV, q, offsetArray)
								q = nil
								offsetArray = nil
							} else {
								core.Error("mlx: native fixed attention updated cache without valid K/V state; falling back to Go graph")
								Free(nativeOut)
							}
						} else {
							core.Error("mlx: native fixed owner attention returned invalid K/V state; falling back to Go graph", "error", err)
							Free(nativeOut, nativeKeys, nativeValues)
						}
					}
					Free(offsetArray)
				}
			}
			if out == nil {
				if paged, ok := c.(*PagedKVCache); ok && L == 1 && mask == nil {
					pages := paged.UpdateBorrowedPages(k, v, int(L))
					pagedKV := sharedKV{Pages: pages, Offset: offset}
					if pagedKV.hasPages() {
						Free(oldK, oldV)
						kv = pagedKV
					} else {
						pages.Free()
						kv = sharedKV{Keys: oldK, Values: oldV, Offset: offset}
					}
				} else {
					k, v = c.Update(k, v, int(L))
					if gemma4ValidKV(k, v) {
						Free(oldK, oldV)
						kv = sharedKV{Keys: k, Values: v, Offset: offset}
					} else {
						Free(k, v)
						kv = sharedKV{Keys: oldK, Values: oldV, Offset: offset}
					}
				}
			}
		} else {
			kv = sharedKV{Keys: k, Values: v, Offset: offset}
		}
	} else {
		offset = kv.Offset
	}

	if out == nil {
		repeatFactor := cfg.NumAttentionHeads / a.NKVHeads
		if kv.hasPages() && L == 1 && mask == nil {
			qRoPE := a.applyRoPE(q, offset)
			Free(q)
			q = qRoPE
			qRoPEApplied = true
			attentionQ := q
			var ownedAttentionQ *Array
			if len(kv.Pages.Keys) > 0 {
				attentionQ, ownedAttentionQ = attentionQueryForKV(q, kv.Pages.Keys[0])
			} else if kv.Keys != nil {
				attentionQ, ownedAttentionQ = attentionQueryForKV(q, kv.Keys)
			}
			if gemma4ValidKV(kv.Keys, kv.Values) {
				out = ScaledDotProductAttention(attentionQ, kv.Keys, kv.Values, a.Scale, false)
			}
			if out == nil && nativePagedAttentionEnabled() && !materializePagedKVForReuse && len(kv.Pages.Keys) > 1 {
				var ok bool
				var err error
				out, ok, err = nativePagedSingleTokenAttention(attentionQ, kv.Pages.Keys, kv.Pages.Values, a.Scale)
				if !ok || err != nil {
					if err != nil {
						core.Error("mlx: native paged attention failed; falling back to Go graph", "error", err)
					}
					out = nil
				}
			}
			if out == nil && pagedDecodeFastConcatEnabled() && len(kv.Pages.Keys) > 1 {
				traceStart := time.Time{}
				if nativePhaseTraceArmed() {
					traceStart = time.Now()
				}
				kBase, vBase := concatenatePagedState(kv.Pages.Keys, kv.Pages.Values)
				tracePagedKVConcat("paged_kv.fast_concat."+gemma4AttentionWindowTraceName(window), traceStart, kv.Pages)
				concatQ := attentionQ
				var ownedConcatQ *Array
				if ownedAttentionQ == nil {
					concatQ, ownedConcatQ = attentionQueryForKV(q, kBase)
				}
				out = ScaledDotProductAttention(concatQ, kBase, vBase, a.Scale, false)
				Free(ownedConcatQ)
				if window == 0 {
					kv.Keys = kBase
					kv.Values = vBase
				} else {
					Free(kBase, vBase)
				}
			}
			if out == nil {
				kPages, vPages := kv.Pages.Keys, kv.Pages.Values
				var repeatedPages []*Array
				if len(kPages) > 1 && pagedStateNeedsMaterializedRepeat(kv.Pages, repeatFactor) {
					kPages, vPages, repeatedPages = repeatPagedState(kv.Pages, repeatFactor)
				}
				out = ScaledDotProductAttentionPaged(attentionQ, kPages, vPages, a.Scale)
				Free(repeatedPages...)
			}
			Free(ownedAttentionQ)
		} else {
			kBase, vBase := kv.Keys, kv.Values
			var ownedContiguous []*Array
			if (kBase == nil || vBase == nil) && kv.hasPages() {
				traceStart := time.Time{}
				if nativePhaseTraceArmed() {
					traceStart = time.Now()
				}
				kBase, vBase = concatenatePagedState(kv.Pages.Keys, kv.Pages.Values)
				tracePagedKVConcat("paged_kv.contiguous."+gemma4AttentionWindowTraceName(window), traceStart, kv.Pages)
				ownedContiguous = append(ownedContiguous, kBase, vBase)
			}
			if !gemma4ValidKV(kBase, vBase) {
				Free(q)
				Free(ownedContiguous...)
				panic("mlx: Gemma 4 attention missing valid K/V state")
			}
			if mask == nil && offset > 0 && L > 1 && window > 0 {
				localContextLen := gemma4SlidingCausalContextLen(L, int32(kBase.Dim(2)), window)
				tailK, tailV := cacheTail(kBase, vBase, localContextLen)
				if tailK != kBase {
					ownedContiguous = append(ownedContiguous, tailK)
					kBase = tailK
				}
				if tailV != vBase {
					ownedContiguous = append(ownedContiguous, tailV)
					vBase = tailV
				}
			}
			var cachedMask *Array
			cachedMaskOwned := false
			useCausalAttention := false
			if mask == nil && offset > 0 && L > 1 {
				keyLen := int32(kBase.Dim(2))
				if gemma4CanUseOffsetCausalAttention(L, keyLen, window) {
					useCausalAttention = true
				} else {
					keyStart := int32(offset) + L - keyLen
					if keyStart < 0 {
						keyStart = 0
					}
					if runtimeMasks != nil {
						cachedMask = runtimeMasks.CachedAttentionMask(B, L, keyLen, int32(offset), keyStart, window)
					} else {
						cachedMask = buildGemma4CachedAttentionMask(B, L, keyLen, int32(offset), keyStart, window)
						cachedMaskOwned = true
					}
					mask = cachedMask
				}
			} else if kv.Fixed && L == 1 && mask == nil {
				offsetArray := FromValue(offset)
				cachedMask = singleTokenCausalMask(int(kBase.Dim(2)), offsetArray)
				Free(offsetArray)
				cachedMaskOwned = true
				mask = cachedMask
			}
			if !qRoPEApplied {
				qRoPE := a.applyRoPE(q, offset)
				Free(q)
				q = qRoPE
				qRoPEApplied = true
			}
			attentionQ, ownedAttentionQ := attentionQueryForKV(q, kBase)
			if mask != nil {
				out = ScaledDotProductAttentionWithMask(attentionQ, kBase, vBase, mask, a.Scale)
			} else if useCausalAttention {
				out = ScaledDotProductAttention(attentionQ, kBase, vBase, a.Scale, true)
			} else {
				out = ScaledDotProductAttention(attentionQ, kBase, vBase, a.Scale, L > 1)
			}
			Free(ownedAttentionQ)
			if cachedMaskOwned {
				Free(cachedMask)
			}
			Free(ownedContiguous...)
		}
	}
	if !qRoPEApplied {
		qRoPE := a.applyRoPE(q, offset)
		Free(q)
		q = qRoPE
		qRoPEApplied = true
	}
	Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := Transpose4(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*a.HeadDim)
	Free(transposed)
	result := a.forwardOProjection(reshaped)
	Free(reshaped)
	return result, kv
}

func (a *Gemma4Attention) forwardOProjection(x *Array) *Array {
	if nativeGemma4AttentionOMatVecEnabled() {
		out, ok, err := quantizedDenseMatVec(x, a.OProj)
		if err != nil {
			core.Error("mlx: native Gemma 4 attention output matvec failed; falling back to Go graph", "error", err)
			Free(out)
		} else if ok {
			return out
		}
	}
	return a.OProj.Forward(x)
}

func (r *Gemma4Router) forward(x *Array) (*Array, *Array) {
	scaled := r.ScaleScaled
	if scaled == nil {
		scaled = MulScalar(r.Scale, r.RootSize)
		defer Free(scaled)
	}
	normed := RMSNorm(x, scaled, r.Eps)
	expertScores, ok, err := nativeGemma4RouterMatVecScores(normed, r.Proj)
	if !ok {
		expertScores = r.Proj.Forward(normed)
	} else if err != nil {
		core.Error("mlx: native Gemma 4 router matvec failed; falling back to Go graph", "error", err)
		Free(expertScores)
		expertScores = r.Proj.Forward(normed)
	}
	Free(normed)

	numExperts := expertScores.Dim(expertScores.NumDims() - 1)
	topK := int(r.TopK)
	if topK <= 0 || topK > numExperts {
		topK = numExperts
	}
	if topKIndices, topKWeights, ok, err := nativeGemma4RouterTopK(expertScores, r.PerExpertScale, topK); ok {
		if err == nil {
			Free(expertScores)
			return topKIndices, topKWeights
		}
		core.Error("mlx: native Gemma 4 router top-k failed; falling back to Go graph", "error", err)
		Free(topKIndices, topKWeights)
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

func (e *Gemma4Experts) forward(x, topKIndices, topKWeights *Array, tracePrefix string) *Array {
	trace := func(phase string, arrays ...*Array) {
		if tracePrefix == "" {
			return
		}
		traceNativeMaterialize(tracePrefix+"."+phase, arrays...)
	}
	if result, ok := e.forwardExpertIDMatVec(x, topKIndices, topKWeights, trace); ok {
		return result
	}
	if result, ok := e.forwardSortedExpertPrefill(x, topKIndices, topKWeights, trace); ok {
		return result
	}
	expanded1 := ExpandDims(x, 2)
	expanded := ExpandDims(expanded1, 2)
	Free(expanded1)

	var gate, up *Array
	if e.GateUpProj != nil && gemma4UseFusedExpertGateUp(x) {
		gateUp := e.GateUpProj.Forward(expanded, topKIndices)
		trace("gate_up", gateUp)
		var ok bool
		gate, up, ok = splitLastDimArray(gateUp)
		Free(gateUp)
		if !ok {
			gate, up = nil, nil
		}
	}
	if gate == nil || up == nil {
		Free(gate, up)
		up = e.UpProj.Forward(expanded, topKIndices)
		trace("up", up)
		gate = e.GateProj.Forward(expanded, topKIndices)
		trace("gate", gate)
	}
	Free(expanded)
	activated := geluGateMul(gate, up)
	trace("activation", activated)
	Free(gate, up)
	down := e.DownProj.Forward(activated, topKIndices)
	trace("down", down)
	Free(activated)
	downSqueezed := Squeeze(down, 3)
	Free(down)

	weightsExpanded := ExpandDims(topKWeights, 3)
	weighted := Mul(weightsExpanded, downSqueezed)
	trace("weighted", weighted)
	Free(weightsExpanded, downSqueezed)
	result := Sum(weighted, -2, false)
	trace("sum", result)
	Free(weighted)
	return result
}

func (e *Gemma4Experts) forwardSortedExpertPrefill(x, topKIndices, topKWeights *Array, trace func(string, ...*Array)) (*Array, bool) {
	if !sortedExpertPrefillEnabled() {
		return nil, false
	}
	if !gemma4SortedExpertPrefillCompatible(e) {
		return nil, false
	}
	if x == nil || topKIndices == nil || topKWeights == nil || !x.Valid() || !topKIndices.Valid() || !topKWeights.Valid() {
		return nil, false
	}
	// Stack-allocated shape scratch — sorted-expert prefill is called
	// per MoE block (× NumHiddenLayers) per prefill batch. Avoids 2-3
	// per-call []int32 heap allocs from x/topKIndices/DownProj.Weight Shape().
	var xShapeBuf, indicesShapeBuf, weightShapeBuf [maxTensorRank]int32
	xShape := x.ShapeInto(xShapeBuf[:0])
	indicesShape := topKIndices.ShapeInto(indicesShapeBuf[:0])
	if len(xShape) != 3 || len(indicesShape) != 3 || indicesShape[0] != xShape[0] || indicesShape[1] != xShape[1] {
		return nil, false
	}
	if xShape[1] <= 1 {
		return nil, false
	}
	batch := int(xShape[0])
	seqLen := int(xShape[1])
	hidden := int(xShape[2])
	topK := int(indicesShape[2])
	routes := topKIndices.Size()
	if batch <= 0 || seqLen <= 1 || hidden <= 0 || topK <= 0 || routes != batch*seqLen*topK || topKWeights.Size() != routes {
		return nil, false
	}
	numExperts := int(e.DownProj.Weight.ShapeInto(weightShapeBuf[:0])[0])
	if routes < 16 || numExperts <= 0 || routes/numExperts < 4 {
		return nil, false
	}

	flatIndices := Reshape(topKIndices, int32(routes))
	sortOrder := Argsort(flatIndices, -1)
	sortedIndices := Take(flatIndices, sortOrder, 0)
	routePositions := Arange(0, float64(routes), 1, DTypeInt32)
	sortedRoutePositions := Take(routePositions, sortOrder, 0)
	topKDivisor := FromValue(topK)
	sortedTokenPositions := floorDivide(sortedRoutePositions, topKDivisor)
	flatX := Reshape(x, int32(batch*seqLen), int32(hidden))
	sortedInputFlat := Take(flatX, sortedTokenPositions, 0)
	sortedInput := Reshape(sortedInputFlat, int32(routes), 1, int32(hidden))
	Free(routePositions, sortedRoutePositions, topKDivisor, sortedTokenPositions, flatX, sortedInputFlat)
	defer Free(flatIndices, sortOrder, sortedIndices, sortedInput)

	gate := gemma4SwitchLinearForwardSortedRoutes(e.GateProj, sortedInput, sortedIndices)
	trace("sorted_gate", gate)
	up := gemma4SwitchLinearForwardSortedRoutes(e.UpProj, sortedInput, sortedIndices)
	trace("sorted_up", up)
	activated := geluGateMul(gate, up)
	trace("sorted_activation", activated)
	Free(gate, up)
	down := gemma4SwitchLinearForwardSortedRoutes(e.DownProj, activated, sortedIndices)
	trace("sorted_down", down)
	Free(activated)

	flatWeights := Reshape(topKWeights, int32(routes))
	sortedWeights := Take(flatWeights, sortOrder, 0)
	weightsExpanded1 := ExpandDims(sortedWeights, 1)
	weightsExpanded := ExpandDims(weightsExpanded1, 2)
	weightedSorted := Mul(weightsExpanded, down)
	trace("sorted_weighted", weightedSorted)
	Free(flatWeights, sortedWeights, weightsExpanded1, weightsExpanded, down)

	inverseOrder := Argsort(sortOrder, -1)
	weightedOriginal := Take(weightedSorted, inverseOrder, 0)
	weightedSqueezed := Squeeze(weightedOriginal, 1)
	grouped := Reshape(weightedSqueezed, int32(batch), int32(seqLen), int32(topK), int32(hidden))
	result := Sum(grouped, -2, false)
	trace("sorted_sum", result)
	Free(weightedSorted, inverseOrder, weightedOriginal, weightedSqueezed, grouped)
	return result, true
}

func gemma4SortedExpertPrefillCompatible(e *Gemma4Experts) bool {
	return e != nil &&
		gemma4ExpertIDMatVecSwitchCompatible(e.GateProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.UpProj) &&
		gemma4ExpertIDMatVecSwitchCompatible(e.DownProj)
}

func gemma4SwitchLinearForwardSortedRoutes(linear *SwitchLinear, input, expertIndices *Array) *Array {
	var out *Array
	if requiresDenseQuantizedMatmulFallback(linear.QuantizationMode) {
		denseWeight := dequantizeMode(linear.Weight, linear.Scales, linear.Biases, linear.GroupSize, linear.Bits, linear.QuantizationMode)
		weightTranspose := Transpose(denseWeight, 0, 2, 1)
		out = GatherMM(input, weightTranspose, nil, expertIndices, true)
		Free(denseWeight, weightTranspose)
	} else {
		out = GatherQMM(input, linear.Weight, linear.Scales, linear.Biases, nil, expertIndices, true, linear.GroupSize, linear.Bits, linear.QuantizationMode, true)
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		bias := Take(linear.Bias, expertIndices, 0)
		biasExpanded := ExpandDims(bias, bias.NumDims()-1)
		oldOut := out
		out = Add(out, biasExpanded)
		Free(oldOut, bias, biasExpanded)
	}
	return out
}

func (e *Gemma4Experts) forwardExpertIDMatVec(x, topKIndices, topKWeights *Array, trace func(string, ...*Array)) (*Array, bool) {
	if !expertIDMatVecEnabled() {
		return nil, false
	}
	if e == nil || e.DownProj == nil {
		return nil, false
	}
	hasFusedGateUp := gemma4ExpertIDMatVecSwitchCompatible(e.GateUpProj)
	hasSplitGateUp := gemma4ExpertIDMatVecSwitchCompatible(e.GateProj) && gemma4ExpertIDMatVecSwitchCompatible(e.UpProj)
	if (!hasFusedGateUp && !hasSplitGateUp) || !gemma4ExpertIDMatVecSwitchCompatible(e.DownProj) {
		return nil, false
	}
	if x == nil || topKIndices == nil || topKWeights == nil || !x.Valid() || !topKIndices.Valid() || !topKWeights.Valid() {
		return nil, false
	}
	// Stack-allocated shape scratch — per-token decode MoE hot path.
	// Called once per MoE block × NumHiddenLayers per generated token.
	var xShapeBuf, indicesShapeBuf [maxTensorRank]int32
	xShape := x.ShapeInto(xShapeBuf[:0])
	indicesShape := topKIndices.ShapeInto(indicesShapeBuf[:0])
	if len(xShape) != 3 || xShape[0] != 1 || xShape[1] != 1 || len(indicesShape) != 3 || indicesShape[0] != 1 || indicesShape[1] != 1 {
		return nil, false
	}
	hidden := int(xShape[2])
	routes := int(indicesShape[2])
	if hidden <= 0 || routes <= 0 || topKWeights.Size() != routes {
		return nil, false
	}

	xFlat := Reshape(x, 1, int32(hidden))
	idsFlat := Reshape(topKIndices, int32(routes))
	defer Free(xFlat, idsFlat)

	var activated *Array
	if hasFusedGateUp && expertIDFusedActivationEnabled() {
		var err error
		activated, err = quantizedExpertIDGELUGateUpMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id fused activation matvec failed; falling back", "error", err)
			return nil, false
		}
		trace("activation_id_matvec", activated)
	} else if hasFusedGateUp {
		gateUp, err := quantizedExpertIDMatVec(xFlat, e.GateUpProj.Weight, e.GateUpProj.Scales, e.GateUpProj.Biases, idsFlat, e.GateUpProj.GroupSize, e.GateUpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec gate/up failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_up_id_matvec", gateUp)
		gate, up, ok := splitLastDimArray(gateUp)
		Free(gateUp)
		if !ok {
			Free(gate, up)
			return nil, false
		}
		activated = geluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		Free(gate, up)
	} else if expertIDFusedActivationEnabled() {
		var err error
		activated, err = quantizedExpertIDGELUSplitGateUpMatVec(
			xFlat,
			e.GateProj.Weight, e.GateProj.Scales, e.GateProj.Biases,
			e.UpProj.Weight, e.UpProj.Scales, e.UpProj.Biases,
			idsFlat,
			e.GateProj.GroupSize,
			e.GateProj.Bits,
		)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id split gate/up fused activation matvec failed; falling back", "error", err)
			return nil, false
		}
		trace("activation_split_id_matvec", activated)
	} else {
		up, err := quantizedExpertIDMatVec(xFlat, e.UpProj.Weight, e.UpProj.Scales, e.UpProj.Biases, idsFlat, e.UpProj.GroupSize, e.UpProj.Bits)
		if err != nil {
			core.Error("mlx: Gemma 4 expert id matvec up failed; falling back", "error", err)
			return nil, false
		}
		trace("up_id_matvec", up)
		gate, err := quantizedExpertIDMatVec(xFlat, e.GateProj.Weight, e.GateProj.Scales, e.GateProj.Biases, idsFlat, e.GateProj.GroupSize, e.GateProj.Bits)
		if err != nil {
			Free(up)
			core.Error("mlx: Gemma 4 expert id matvec gate failed; falling back", "error", err)
			return nil, false
		}
		trace("gate_id_matvec", gate)
		activated = geluGateMul(gate, up)
		trace("activation_id_matvec", activated)
		Free(gate, up)
	}

	weightsFlat := Reshape(topKWeights, int32(routes))
	down, err := quantizedExpertIDWeightedMatVecSum(activated, weightsFlat, e.DownProj.Weight, e.DownProj.Scales, e.DownProj.Biases, idsFlat, e.DownProj.GroupSize, e.DownProj.Bits)
	Free(weightsFlat)
	Free(activated)
	if err != nil {
		core.Error("mlx: Gemma 4 expert id weighted matvec down failed; falling back", "error", err)
		return nil, false
	}
	trace("down_weighted_sum_id_matvec", down)
	result := Reshape(down, 1, 1, int32(hidden))
	Free(down)
	return result, true
}

func gemma4ExpertIDMatVecSwitchCompatible(linear *SwitchLinear) bool {
	return linear != nil &&
		linear.Weight != nil && linear.Weight.Valid() &&
		linear.Scales != nil && linear.Scales.Valid() &&
		linear.Biases != nil && linear.Biases.Valid() &&
		linear.GroupSize > 0 &&
		isAffineQuantizationMode(linear.QuantizationMode) &&
		(linear.Bits == 2 || linear.Bits == 4 || linear.Bits == 8)
}

func gemma4UseFusedExpertGateUp(x *Array) bool {
	if x == nil || !x.Valid() {
		return false
	}
	// Branch on the row dim only — Shape() would heap-allocate a fresh
	// []int32 per MoE block per layer per token. Dim() is one C call.
	return x.NumDims() >= 2 && x.Dim(1) == 1
}

func splitLastDimArray(a *Array) (*Array, *Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	// Stack-allocated shape scratch — called per MoE block on the
	// fused-gate-up split path. Avoids per-call []int32 heap alloc.
	var shapeBuf [maxTensorRank]int32
	shape := a.ShapeInto(shapeBuf[:0])
	if len(shape) == 0 {
		return nil, nil, false
	}
	axis := len(shape) - 1
	mid := shape[axis] / 2
	if mid <= 0 || shape[axis]%2 != 0 {
		return nil, nil, false
	}
	var startsBuf, endsBuf [maxTensorRank]int32
	starts := startsBuf[:len(shape)]
	ends := endsBuf[:len(shape)]
	copy(ends, shape)
	ends[axis] = mid
	left := Slice(a, starts, ends)
	starts[axis] = mid
	ends[axis] = shape[axis]
	right := Slice(a, starts, ends)
	return left, right, true
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
	cfg = normalizeGemma4LoRAConfig(cfg)
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
