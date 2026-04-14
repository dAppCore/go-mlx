//go:build darwin && arm64

package metal

import (
	"maps"
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

	PreviousKVs []int32
	modelType   string
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

	LayerType string
	IsSliding bool
	LayerIdx  int32
}

// Gemma4Attention implements Gemma 4 attention with per-layer RoPE and K-eq-V.
type Gemma4Attention struct {
	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear
	QNorm *RMSNormModule
	KNorm *RMSNormModule

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

func parseGemma4Config(data []byte) (*Gemma4TextConfig, error) {
	var wrapper struct {
		ModelType    string              `json:"model_type"`
		TextConfig   Gemma4TextConfig    `json:"text_config"`
		Quantization *QuantizationConfig `json:"quantization"`
		LayerTypes   []string            `json:"layer_types"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gemma4.parseConfig", "parse config", nil)
	}

	cfg := wrapper.TextConfig
	if cfg.NumHiddenLayers == 0 {
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("gemma4.parseConfig", "parse top-level config", nil)
		}
	}

	if wrapper.ModelType != "" {
		cfg.ModelType = wrapper.ModelType
	}
	cfg.Quantization = wrapper.Quantization
	if len(cfg.LayerTypesInput) == 0 && len(wrapper.LayerTypes) > 0 {
		cfg.LayerTypesInput = wrapper.LayerTypes
	}

	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.GlobalHeadDim == 0 {
		cfg.GlobalHeadDim = cfg.HeadDim
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
	if cfg.FinalLogitSoftcapping == 0 {
		cfg.FinalLogitSoftcapping = 30
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
	if cfg.RopeParameters == nil {
		cfg.RopeParameters = map[string]RopeParams{
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

func splitArrayAlongSecondLast(a *Array) (*Array, *Array, bool) {
	if a == nil || !a.Valid() {
		return nil, nil, false
	}
	shape := a.Shape()
	if len(shape) < 2 {
		return nil, nil, false
	}
	axis := len(shape) - 2
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
	for name, arr := range raw {
		trimmed := strings.TrimPrefix(name, "model.")
		if strings.HasPrefix(trimmed, "vision_tower") ||
			strings.HasPrefix(trimmed, "multi_modal_projector") ||
			strings.HasPrefix(trimmed, "audio_tower") ||
			strings.HasPrefix(trimmed, "embed_audio") ||
			strings.HasPrefix(trimmed, "embed_vision") {
			continue
		}
		if strings.Contains(name, "self_attn.rotary_emb") ||
			strings.Contains(name, "input_max") ||
			strings.Contains(name, "input_min") ||
			strings.Contains(name, "output_max") ||
			strings.Contains(name, "output_min") {
			continue
		}
		for _, suffix := range []string{".weight", ".scales", ".biases", ".bias"} {
			if strings.Contains(name, ".experts.gate_up_proj"+suffix) {
				base := strings.TrimSuffix(name, suffix)
				base = strings.TrimSuffix(base, ".gate_up_proj")
				gate, up, ok := splitArrayAlongSecondLast(arr)
				if !ok {
					break
				}
				sanitized[base+".gate_proj"+suffix] = gate
				sanitized[base+".up_proj"+suffix] = up
				goto nextWeight
			}
		}
		sanitized[name] = arr
	nextWeight:
	}
	return sanitized
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

func gemma4SwitchLinear(weights map[string]*Array, prefix string, defaultQ *QuantizationConfig) *SwitchLinear {
	weight := gemma4WeightAny(weights, prefix+".weight")
	if weight == nil {
		return nil
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

func buildGemma4PreviousKVs(layers []*Gemma4DecoderLayer, numShared int32) []int32 {
	previous := make([]int32, len(layers))
	for i := range previous {
		previous[i] = int32(i)
	}
	if numShared <= 0 {
		return previous
	}
	firstShared := int32(len(layers)) - numShared
	if firstShared < 0 {
		firstShared = 0
	}
	latestByType := make(map[string]int32)
	for i := int32(0); i < firstShared; i++ {
		latestByType[layers[i].LayerType] = i
	}
	for i := firstShared; i < int32(len(layers)); i++ {
		if prev, ok := latestByType[layers[i].LayerType]; ok {
			previous[i] = prev
		}
	}
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

// LoadGemma4 loads a Gemma 4 text model from a directory.
func LoadGemma4(modelPath string) (*Gemma4Model, error) {
	str, err := coreio.Local.Read(core.JoinPath(modelPath, "config.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseGemma4Config(data)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "parse config", err)
	}

	tok, err := LoadTokenizer(core.JoinPath(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load tokenizer", err)
	}

	rawWeights := make(map[string]*Array)
	matches := core.PathGlob(core.JoinPath(modelPath, "*.safetensors"))
	if len(matches) == 0 {
		return nil, core.E("gemma4.LoadGemma4", "no .safetensors files found in "+modelPath, nil)
	}
	for _, path := range matches {
		maps.Insert(rawWeights, LoadSafetensors(path))
		if err := lastError(); err != nil {
			return nil, core.E("gemma4.LoadGemma4", "load weights "+core.PathBase(path), err)
		}
	}
	weights := sanitizeGemma4Weights(rawWeights)

	if cfg.HeadDim == 0 {
		for i, layerType := range cfg.LayerTypes {
			if layerType != "sliding_attention" {
				continue
			}
			if qProj := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); qProj != nil {
				cfg.HeadDim = qProj.Shape()[0] / cfg.NumAttentionHeads
				break
			}
		}
	}
	if cfg.GlobalHeadDim == 0 {
		for i, layerType := range cfg.LayerTypes {
			if layerType != "full_attention" {
				continue
			}
			if qProj := gemma4WeightAny(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); qProj != nil {
				cfg.GlobalHeadDim = qProj.Shape()[0] / cfg.NumAttentionHeads
				break
			}
		}
		if cfg.GlobalHeadDim == 0 {
			cfg.GlobalHeadDim = cfg.HeadDim
		}
	}

	if cfg.HiddenSizePerLayerInput > 0 {
		if gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight") == nil ||
			gemma4WeightAny(weights, "model.per_layer_model_projection.weight") == nil {
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
			LayerScalar: gemma4WeightAny(weights, prefix+".layer_scalar", prefix+".layer_scalar.weight"),
			LayerType:   layerType,
			IsSliding:   isSliding,
			LayerIdx:    i,
			EnableMoE:   cfg.EnableMoEBlock,
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
				GateProj: gemma4SwitchLinear(weights, prefix+".experts.gate_proj", cfg.Quantization),
				UpProj:   gemma4SwitchLinear(weights, prefix+".experts.up_proj", cfg.Quantization),
				DownProj: gemma4SwitchLinear(weights, prefix+".experts.down_proj", cfg.Quantization),
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

	lmHeadWeight := gemma4WeightAny(weights, "lm_head.weight")
	if lmHeadWeight != nil {
		lmHeadScales := gemma4WeightAny(weights, "lm_head.scales")
		if lmHeadScales != nil && cfg.Quantization != nil {
			m.Output = NewQuantizedLinear(lmHeadWeight, lmHeadScales, gemma4WeightAny(weights, "lm_head.biases"), nil, cfg.Quantization.GroupSize, cfg.Quantization.Bits)
		} else {
			m.Output = NewLinear(lmHeadWeight, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}

	m.PreviousKVs = buildGemma4PreviousKVs(m.Layers, cfg.NumKVSharedLayers)

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

func (m *Gemma4Model) computePerLayerInputs(tokens, hidden *Array) []*Array {
	if m.EmbedTokensPerLayer == nil || m.PerLayerModelProj == nil || m.PerLayerProjNorm == nil || m.PerLayerProjNormScaled == nil {
		return nil
	}
	B, L := tokens.Shape()[0], tokens.Shape()[1]
	perLayer := m.EmbedTokensPerLayer.Forward(tokens)
	scale := float32(math.Sqrt(float64(m.Cfg.HiddenSizePerLayerInput)))
	scaled := MulScalar(perLayer, scale)
	Free(perLayer)
	perLayer = Reshape(scaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	Free(scaled)

	projected := m.PerLayerModelProj.Forward(hidden)
	projectedScaled := MulScalar(projected, float32(math.Pow(float64(m.Cfg.HiddenSize), -0.5)))
	Free(projected)
	projected = Reshape(projectedScaled, B, L, m.Cfg.NumHiddenLayers, m.Cfg.HiddenSizePerLayerInput)
	Free(projectedScaled)
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
		if m.PreviousKVs[i] == int32(i) && i < len(caches) {
			cache = caches[i]
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
	var ff *Array
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

		ff = Add(h1Normed, h2Normed)
		Free(h1Normed, h2Normed)
	} else {
		ffIn := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
		ff = l.MLP.forward(ffIn)
		Free(ffIn)
	}

	ffNormed := RMSNorm(ff, l.PostFFNormScaled, cfg.RMSNormEps)
	Free(ff)
	hNext := Add(residual, ffNormed)
	Free(h, ffNormed)

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
	firstShared := m.Cfg.NumHiddenLayers - m.Cfg.NumKVSharedLayers
	if firstShared < 0 {
		firstShared = 0
	}
	caches := make([]Cache, firstShared)
	for i := int32(0); i < firstShared; i++ {
		if m.Layers[i].LayerType == "full_attention" {
			caches[i] = NewKVCache()
		} else {
			caches[i] = NewRotatingKVCache(int(m.Cfg.SlidingWindow))
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
