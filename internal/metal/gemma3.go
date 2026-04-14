//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"

	"dappco.re/go/core"

	coreio "dappco.re/go/core/io"
)

// TextConfig holds Gemma 3 text model configuration.
type TextConfig struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeLocalBaseFreq     float32 `json:"rope_local_base_freq"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	SlidingWindow         int32   `json:"sliding_window"`
	SlidingWindowPattern  int32   `json:"sliding_window_pattern"`

	Quantization *QuantizationConfig `json:"-"` // Parsed separately from top-level
	Scale        float32             `json:"-"` // Computed: 1/sqrt(head_dim)
}

// GemmaModel is the Gemma 3 text model.
type GemmaModel struct {
	EmbedTokens *Embedding
	Layers      []*DecoderLayer
	Norm        *RMSNormModule
	Output      *Linear // Tied to EmbedTokens

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	NormScaled *Array

	Tok *Tokenizer
	Cfg *TextConfig
}

// DecoderLayer is a single transformer block.
type DecoderLayer struct {
	InputNorm    *RMSNormModule
	Attention    *Attention
	PostAttnNorm *RMSNormModule
	PreFFNorm    *RMSNormModule
	MLP          *MLP
	PostFFNorm   *RMSNormModule

	// Precomputed scaled weights
	InputNormScaled    *Array
	PostAttnNormScaled *Array
	PreFFNormScaled    *Array
	PostFFNormScaled   *Array

	IsSliding bool
	LayerIdx  int32
}

// Attention implements Gemma 3 attention with Q/K normalization.
type Attention struct {
	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear
	QNorm *RMSNormModule
	KNorm *RMSNormModule

	QNormScaled *Array
	KNormScaled *Array
}

// MLP is the feed-forward network.
type MLP struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

// compiledGELU is a singleton for the compiled GELU function.
var compiledGELU *CompiledFunc

func getCompiledGELU() *CompiledFunc {
	if compiledGELU == nil {
		compiledGELU = CompileShapeless(func(inputs []*Array) []*Array {
			return []*Array{geluApprox(inputs[0])}
		}, true)
	}
	return compiledGELU
}

// geluApprox computes GELU using the tanh approximation:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluApprox(x *Array) *Array {
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715

	xSquared := Mul(x, x)
	x3 := Mul(xSquared, x)
	Free(xSquared)
	x3Scaled := MulScalar(x3, coeff)
	Free(x3)
	inner := Add(x, x3Scaled)
	Free(x3Scaled)
	scaled := MulScalar(inner, sqrt2OverPi)
	Free(inner)
	t := Tanh(scaled)
	Free(scaled)
	onePlusT := AddScalar(t, 1.0)
	Free(t)
	halfX := MulScalar(x, 0.5)
	result := Mul(halfX, onePlusT)
	Free(halfX, onePlusT)
	return result
}

// parseConfig handles both flat and nested (text_config) Gemma 3 configs.
func parseConfig(data []byte) (*TextConfig, error) {
	// Try parsing text_config from multimodal wrapper
	var wrapper struct {
		TextConfig   TextConfig          `json:"text_config"`
		ModelType    string              `json:"model_type"`
		Quantization *QuantizationConfig `json:"quantization"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gemma3.parseConfig", "parse config", nil)
	}

	cfg := wrapper.TextConfig

	// If text_config was empty, try top-level
	if cfg.NumHiddenLayers == 0 {
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("gemma3.parseConfig", "parse top-level config", nil)
		}
	}

	// Quantization is always top-level
	cfg.Quantization = wrapper.Quantization

	// Compute scale (head_dim may be inferred later from weights if not in config)
	if cfg.HeadDim > 0 {
		cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RopeLocalBaseFreq == 0 {
		cfg.RopeLocalBaseFreq = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 262208 // Gemma 3 default
	}

	return &cfg, nil
}

// LoadGemma3 loads a Gemma 3 text model from a directory.
func LoadGemma3(modelPath string) (*GemmaModel, error) {
	root := resolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseConfig(data)
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "parse config", err)
	}

	// Load tokenizer
	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "load tokenizer", err)
	}

	weights, err := loadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "load weights", err)
	}

	weight := func(name string) *Array { return resolveWeight(weights, name) }

	// Infer head_dim from q_proj weight shape when not in config.
	// Gemma 3 uses head_dim=256 which differs from hidden_size/num_heads.
	if cfg.HeadDim == 0 {
		qProjWeight := weight("model.layers.0.self_attn.q_proj.weight")
		if qProjWeight != nil {
			qShape := qProjWeight.Shape()
			if len(qShape) > 0 {
				cfg.HeadDim = qShape[0] / cfg.NumAttentionHeads
				cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
				core.Info("mlx: inferred head_dim from q_proj weight", "head_dim", cfg.HeadDim)
			}
		}
	}

	quantConfig := cfg.Quantization
	if quantConfig != nil {
		core.Info("mlx: using quantized inference", "bits", quantConfig.Bits, "group_size", quantConfig.GroupSize)
	}
	linear := func(prefix string) *Linear {
		layerWeight := weight(prefix + ".weight")
		scales := weight(prefix + ".scales")
		biases := weight(prefix + ".biases")
		if scales != nil && quantConfig != nil {
			return NewQuantizedLinear(layerWeight, scales, biases, nil, quantConfig.GroupSize, quantConfig.Bits)
		}
		return NewLinear(layerWeight, nil)
	}

	embed := &Embedding{Weight: weight("model.embed_tokens.weight")}
	if embedScales := weight("model.embed_tokens.scales"); embedScales != nil && quantConfig != nil {
		embed.Scales = embedScales
		embed.Biases = weight("model.embed_tokens.biases")
		embed.GroupSize = quantConfig.GroupSize
		embed.Bits = quantConfig.Bits
	}

	gemmaModel := &GemmaModel{
		EmbedTokens: embed,
		Layers:      make([]*DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: weight("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("model.layers.%d", i)
		gemmaModel.Layers[i] = &DecoderLayer{
			InputNorm:    &RMSNormModule{Weight: weight(prefix + ".input_layernorm.weight")},
			PostAttnNorm: &RMSNormModule{Weight: weight(prefix + ".post_attention_layernorm.weight")},
			PreFFNorm:    &RMSNormModule{Weight: weight(prefix + ".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &RMSNormModule{Weight: weight(prefix + ".post_feedforward_layernorm.weight")},
			Attention: &Attention{
				QProj: linear(prefix + ".self_attn.q_proj"),
				KProj: linear(prefix + ".self_attn.k_proj"),
				VProj: linear(prefix + ".self_attn.v_proj"),
				OProj: linear(prefix + ".self_attn.o_proj"),
				QNorm: &RMSNormModule{Weight: weight(prefix + ".self_attn.q_norm.weight")},
				KNorm: &RMSNormModule{Weight: weight(prefix + ".self_attn.k_norm.weight")},
			},
			MLP: &MLP{
				GateProj: linear(prefix + ".mlp.gate_proj"),
				UpProj:   linear(prefix + ".mlp.up_proj"),
				DownProj: linear(prefix + ".mlp.down_proj"),
			},
			LayerIdx:  i,
			IsSliding: isLayerSliding(i, cfg.SlidingWindowPattern),
		}
	}

	// lm_head: separate weight if present, else tied to embed_tokens
	lmHeadWeight := weight("lm_head.weight")
	if lmHeadWeight != nil {
		lmHeadScales := weight("lm_head.scales")
		if lmHeadScales != nil && quantConfig != nil {
			gemmaModel.Output = NewQuantizedLinear(lmHeadWeight, lmHeadScales, weight("lm_head.biases"), nil, quantConfig.GroupSize, quantConfig.Bits)
		} else {
			gemmaModel.Output = NewLinear(lmHeadWeight, nil)
		}
	} else {
		gemmaModel.Output = gemmaModel.EmbedTokens.AsLinear() // tied embeddings
	}

	var allArrays []*Array
	for _, arr := range weights {
		allArrays = append(allArrays, arr)
	}
	Materialize(allArrays...)
	precomputeScaledWeights(gemmaModel) // Gemma-style: weight → (1 + weight)

	return gemmaModel, nil
}

func precomputeScaledWeights(m *GemmaModel) {
	m.NormScaled = AddScalar(m.Norm.Weight, 1.0)

	for _, layer := range m.Layers {
		layer.InputNormScaled = AddScalar(layer.InputNorm.Weight, 1.0)
		layer.PostAttnNormScaled = AddScalar(layer.PostAttnNorm.Weight, 1.0)
		layer.PreFFNormScaled = AddScalar(layer.PreFFNorm.Weight, 1.0)
		layer.PostFFNormScaled = AddScalar(layer.PostFFNorm.Weight, 1.0)
		layer.Attention.QNormScaled = AddScalar(layer.Attention.QNorm.Weight, 1.0)
		layer.Attention.KNormScaled = AddScalar(layer.Attention.KNorm.Weight, 1.0)
	}

	var scaled []*Array
	scaled = append(scaled, m.NormScaled)
	for _, layer := range m.Layers {
		scaled = append(scaled, layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled,
			layer.Attention.QNormScaled, layer.Attention.KNormScaled)
	}
	Materialize(scaled...)
}

func isLayerSliding(layerIdx, pattern int32) bool {
	if pattern <= 0 {
		return false
	}
	return (layerIdx+1)%pattern != 0
}

// Forward runs the text model forward pass.
func (m *GemmaModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *GemmaModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	shape := tokens.Shape()
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)
	embeddingScale := float32(math.Sqrt(float64(m.Cfg.HiddenSize)))
	h2 := MulScalar(h, embeddingScale)
	Free(h)
	h = h2

	for i, layer := range m.Layers {
		hNext := layer.forward(h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}

	normed := RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
}

func (l *DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *TextConfig) *Array {
	normed := RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, l.IsSliding, mask, cfg)
	Free(normed)
	attnOutNormed := RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	Free(attnOut)
	h := Add(x, attnOutNormed)
	Free(attnOutNormed)

	normed2 := RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
	mlpOut := l.MLP.forward(normed2)
	Free(normed2)
	mlpOutNormed := RMSNorm(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	Free(mlpOut)
	result := Add(h, mlpOutNormed)
	Free(h, mlpOutNormed)
	return result
}

func (a *Attention) forward(x *Array, c Cache, B, L int32, isSliding bool, mask *Array, cfg *TextConfig) *Array {
	qProj := a.QProj.Forward(x)
	kProj := a.KProj.Forward(x)
	vProj := a.VProj.Forward(x)

	// Virtual transpose [B,L,H*D] → [B,H,L,D] via stride manipulation.
	// AsStrided creates a view (C refcount keeps source alive), so Free source after.
	q := AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	Free(qProj)
	k := AsStrided(kProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	Free(kProj)
	v := AsStrided(vProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	Free(vProj)

	// Q/K normalization
	oldQ := q
	q = RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	Free(oldQ)
	oldK := k
	k = RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)
	Free(oldK)

	// RoPE with appropriate theta
	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}
	oldQ = q
	q = RoPE(q, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	Free(oldQ)
	oldK = k
	k = RoPE(k, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	Free(oldK)

	// Update cache — returns Slice views into cache buffer; free our pre-update handles.
	oldK, oldV := k, v
	k, v = c.Update(k, v, int(L))
	Free(oldK, oldV)

	// GQA: repeat K/V heads
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	kAttn, vAttn := k, v
	if repeatFactor > 1 {
		kAttn = RepeatKV(k, repeatFactor)
		vAttn = RepeatKV(v, repeatFactor)
		Free(k, v) // Free Slice views from cache.Update; RepeatKV holds copies
	}

	// Scaled dot-product attention
	var out *Array
	if mask != nil {
		out = ScaledDotProductAttentionWithMask(q, kAttn, vAttn, mask, cfg.Scale)
	} else {
		out = ScaledDotProductAttention(q, kAttn, vAttn, cfg.Scale, L > 1)
	}
	Free(q, kAttn, vAttn) // Always free — when repeatFactor==1 this frees the Slice views

	transposed := Transpose(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	Free(transposed)
	result := a.OProj.Forward(reshaped)
	Free(reshaped)
	return result
}

func (m *MLP) forward(x *Array) *Array {
	gateProj := m.GateProj.Forward(x)
	gate := getCompiledGELU().Call(gateProj)[0]
	Free(gateProj)
	upProj := m.UpProj.Forward(x)
	activated := Mul(gate, upProj)
	Free(gate, upProj)
	result := m.DownProj.Forward(activated)
	Free(activated)
	return result
}

// NewCache creates per-layer caches for generation.
func (m *GemmaModel) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		if m.Layers[i].IsSliding {
			caches[i] = NewRotatingKVCache(int(m.Cfg.SlidingWindow))
		} else {
			caches[i] = NewKVCache()
		}
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *GemmaModel) NumLayers() int { return len(m.Layers) }

// Tokenizer returns the model's tokenizer.
func (m *GemmaModel) Tokenizer() *Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *GemmaModel) ModelType() string { return "gemma3" }

// ApplyLoRA wraps target projection layers with LoRA adapters.
// Supports attention targets (q_proj, k_proj, v_proj, o_proj) and
// MLP targets (gate_proj, up_proj, down_proj).
func (m *GemmaModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
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
