// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
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

	Quantization                *metal.QuantizationConfig `json:"-"`
	VisionConfig                *Gemma4VisionConfig       `json:"-"`
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

// Gemma4Model is the Gemma 4 text model.
type Gemma4Model struct {
	EmbedTokens         *metal.Embedding
	EmbedTokensPerLayer *metal.Embedding
	VisionTower         *Gemma4VisionModel
	MultiModalProjector *Gemma4MultiModalProjector
	Layers              []*Gemma4DecoderLayer
	Norm                *metal.RMSNormModule
	Output              *metal.Linear
	PerLayerModelProj   *metal.Linear
	PerLayerProjNorm    *metal.RMSNormModule

	NormScaled             *metal.Array
	PerLayerProjNormScaled *metal.Array

	Tok *metal.Tokenizer
	Cfg *Gemma4TextConfig

	PreviousKVs       []int32
	CacheIndexByLayer []int32
	modelType         string

	compiledPerLayerInputs       *metal.CompiledFunc
	compiledPerLayerInputsFailed bool
}

// Gemma4DecoderLayer is a single transformer block.
type Gemma4DecoderLayer struct {
	InputNorm    *metal.RMSNormModule
	Attention    *Gemma4Attention
	PostAttnNorm *metal.RMSNormModule
	PreFFNorm    *metal.RMSNormModule
	MLP          *metal.MLP
	PostFFNorm   *metal.RMSNormModule

	EnableMoE   bool
	Router      *Gemma4Router
	Experts     *Gemma4Experts
	PreFFNorm2  *metal.RMSNormModule
	PostFFNorm1 *metal.RMSNormModule
	PostFFNorm2 *metal.RMSNormModule

	PerLayerInputGate     *metal.Linear
	PerLayerProjection    *metal.Linear
	PostPerLayerInputNorm *metal.RMSNormModule

	LayerScalar *metal.Array

	InputNormScaled             *metal.Array
	PostAttnNormScaled          *metal.Array
	PreFFNormScaled             *metal.Array
	PostFFNormScaled            *metal.Array
	PreFFNorm2Scaled            *metal.Array
	PostFFNorm1Scaled           *metal.Array
	PostFFNorm2Scaled           *metal.Array
	PostPerLayerInputNormScaled *metal.Array

	LayerType     string
	IsSliding     bool
	DoubleWideMLP bool
	LayerIdx      int32

	compiledNativeOwnerDecode             *metal.CompiledFunc
	compiledNativeSharedDecode            *metal.CompiledFunc
	compiledNativeFixedOwnerDecode        *metal.CompiledFunc
	compiledNativeFixedSharedDecode       *metal.CompiledFunc
	compiledNativeFixedMaskedOwnerDecode  *metal.CompiledFunc
	compiledNativeFixedMaskedSharedDecode *metal.CompiledFunc
	compiledNativeOwnerFailed             bool
	compiledNativeSharedFailed            bool
	compiledNativeFixedOwnerFailed        bool
	compiledNativeFixedSharedFailed       bool
	compiledNativeFixedMaskedOwnerFailed  bool
	compiledNativeFixedMaskedSharedFailed bool
}

// Gemma4Attention implements Gemma 4 attention with per-layer RoPE and K-eq-V.
type Gemma4Attention struct {
	QProj *metal.Linear
	KProj *metal.Linear
	VProj *metal.Linear
	OProj *metal.Linear
	QNorm *metal.RMSNormModule
	KNorm *metal.RMSNormModule
	VNorm *metal.RMSNormModule

	QNormScaled *metal.Array
	KNormScaled *metal.Array

	HeadDim        int32
	NKVHeads       int32
	UseKEqV        bool
	Scale          float32
	RopeBase       float32
	RopeRotatedDim int32
	RopeFreqs      *metal.Array
}

// Gemma4Router routes tokens to top-k experts.
type Gemma4Router struct {
	Proj           *metal.Linear
	Scale          *metal.Array
	PerExpertScale *metal.Array
	ScaleScaled    *metal.Array
	RootSize       float32
	TopK           int32
	Eps            float32
}

// Gemma4Experts holds the SwitchGLU sparse MoE block.
type Gemma4Experts struct {
	GateUpProj *metal.SwitchLinear
	GateProj   *metal.SwitchLinear
	UpProj     *metal.SwitchLinear
	DownProj   *metal.SwitchLinear
}

type sharedKV struct {
	Keys     *metal.Array
	Values   *metal.Array
	Pages    metal.PagedKVState
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
		metal.Free(kv.Keys, kv.Values)
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

func clonePagedKVState(state metal.PagedKVState) metal.PagedKVState {
	out := metal.PagedKVState{Length: state.Length}
	if len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return out
	}
	out.Keys = make([]*metal.Array, len(state.Keys))
	out.Values = make([]*metal.Array, len(state.Values))
	out.Owned = make([]*metal.Array, 0, len(state.Keys)+len(state.Values))
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

func gemma4ValidKV(k, v *metal.Array) bool {
	return k != nil && k.Valid() && v != nil && v.Valid()
}
