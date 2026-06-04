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
	AudioTokenID              int32                 `json:"audio_token_id"`
	VideoTokenID              int32                 `json:"video_token_id"`
	BOITokenID                int32                 `json:"boi_token_id"`
	BOATokenID                int32                 `json:"boa_token_id"`
	EOITokenID                int32                 `json:"eoi_token_id"`
	EOATokenIndex             int32                 `json:"eoa_token_index"`
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
	FFNMemory     metal.FFNMemoryAugmenter

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

// sharedKV is the per-layer K/V hand-off type. It now lives in package metal
// (metal.SharedKV) because the fused cgo kernels both produce and consume it
// (RFC.model-sdk Cat 3); this alias keeps the architecture's references stable.
// Methods are the exported metal ones: HasState/HasPages/Free/Clone.
type sharedKV = metal.SharedKV

// moveSharedKV forwards to metal.MoveSharedKV so architecture callers keep a
// short local name.
func moveSharedKV(kv *sharedKV) sharedKV {
	return metal.MoveSharedKV(kv)
}

func gemma4ValidKV(k, v *metal.Array) bool {
	return k != nil && k.Valid() && v != nil && v.Valid()
}
