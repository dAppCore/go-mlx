//go:build darwin && arm64

package metal

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"os"
	"path/filepath"
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

	x3 := Mul(Mul(x, x), x)
	inner := Add(x, MulScalar(x3, coeff))
	scaled := MulScalar(inner, sqrt2OverPi)
	t := Tanh(scaled)
	onePlusT := AddScalar(t, 1.0)
	return Mul(MulScalar(x, 0.5), onePlusT)
}

// parseConfig handles both flat and nested (text_config) Gemma 3 configs.
func parseConfig(data []byte) (*TextConfig, error) {
	// Try parsing text_config from multimodal wrapper
	var wrapper struct {
		TextConfig   TextConfig          `json:"text_config"`
		ModelType    string              `json:"model_type"`
		Quantization *QuantizationConfig `json:"quantization"`
	}
	if err := json.Unmarshal(data, &wrapper); err != nil {
		return nil, err
	}

	cfg := wrapper.TextConfig

	// If text_config was empty, try top-level
	if cfg.NumHiddenLayers == 0 {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, err
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
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("gemma3: load config: %w", err)
	}

	cfg, err := parseConfig(data)
	if err != nil {
		return nil, fmt.Errorf("gemma3: parse config: %w", err)
	}

	// Load tokenizer
	tok, err := LoadTokenizer(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("gemma3: load tokenizer: %w", err)
	}

	// Load weights from all safetensors files
	weights := make(map[string]*Array)
	matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	if len(matches) == 0 {
		return nil, fmt.Errorf("gemma3: no .safetensors files found in %s", modelPath)
	}
	for _, path := range matches {
		maps.Insert(weights, LoadSafetensors(path))
		if err := lastError(); err != nil {
			return nil, fmt.Errorf("gemma3: load weights %s: %w", filepath.Base(path), err)
		}
	}

	// Helper to resolve weight with language_model. prefix fallback
	w := func(name string) *Array { return resolveWeight(weights, name) }

	// Infer head_dim from q_proj weight shape when not in config.
	// Gemma 3 uses head_dim=256 which differs from hidden_size/num_heads.
	if cfg.HeadDim == 0 {
		qWeight := w("model.layers.0.self_attn.q_proj.weight")
		if qWeight != nil {
			qShape := qWeight.Shape()
			if len(qShape) > 0 {
				cfg.HeadDim = qShape[0] / cfg.NumAttentionHeads
				cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
				slog.Info("mlx: inferred head_dim from q_proj weight", "head_dim", cfg.HeadDim)
			}
		}
	}

	// Helper to create linear layer (quantized or dense)
	q := cfg.Quantization
	if q != nil {
		slog.Info("mlx: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(prefix string) *Linear {
		weight := w(prefix + ".weight")
		scales := w(prefix + ".scales")
		biases := w(prefix + ".biases")
		if scales != nil && q != nil {
			return NewQuantizedLinear(weight, scales, biases, nil, q.GroupSize, q.Bits)
		}
		return NewLinear(weight, nil)
	}

	// Create embedding (quantized or dense)
	embed := &Embedding{Weight: w("model.embed_tokens.weight")}
	if embedScales := w("model.embed_tokens.scales"); embedScales != nil && q != nil {
		embed.Scales = embedScales
		embed.Biases = w("model.embed_tokens.biases")
		embed.GroupSize = q.GroupSize
		embed.Bits = q.Bits
	}

	m := &GemmaModel{
		EmbedTokens: embed,
		Layers:      make([]*DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
	}

	// Initialize layers
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)
		m.Layers[i] = &DecoderLayer{
			InputNorm:    &RMSNormModule{Weight: w(prefix + ".input_layernorm.weight")},
			PostAttnNorm: &RMSNormModule{Weight: w(prefix + ".post_attention_layernorm.weight")},
			PreFFNorm:    &RMSNormModule{Weight: w(prefix + ".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &RMSNormModule{Weight: w(prefix + ".post_feedforward_layernorm.weight")},
			Attention: &Attention{
				QProj: linear(prefix + ".self_attn.q_proj"),
				KProj: linear(prefix + ".self_attn.k_proj"),
				VProj: linear(prefix + ".self_attn.v_proj"),
				OProj: linear(prefix + ".self_attn.o_proj"),
				QNorm: &RMSNormModule{Weight: w(prefix + ".self_attn.q_norm.weight")},
				KNorm: &RMSNormModule{Weight: w(prefix + ".self_attn.k_norm.weight")},
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

	// Output head — check for separate lm_head first, else tie to embeddings
	lmHeadWeight := w("lm_head.weight")
	if lmHeadWeight != nil {
		lmHeadScales := w("lm_head.scales")
		if lmHeadScales != nil && q != nil {
			m.Output = NewQuantizedLinear(lmHeadWeight, lmHeadScales, w("lm_head.biases"), nil, q.GroupSize, q.Bits)
		} else {
			m.Output = NewLinear(lmHeadWeight, nil)
		}
	} else {
		// Tied embeddings — reuse embed_tokens weights (with quantization if present)
		m.Output = m.EmbedTokens.AsLinear()
	}

	// Materialize all weights
	var allArrays []*Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	Materialize(allArrays...)

	// Precompute (1 + weight) for Gemma-style RMSNorm
	precomputeScaledWeights(m)

	return m, nil
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
	h = MulScalar(h, float32(math.Sqrt(float64(m.Cfg.HiddenSize))))

	for i, layer := range m.Layers {
		h = layer.forward(h, caches[i], B, L, mask, m.Cfg)
	}

	return m.Output.Forward(RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps))
}

func (l *DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *TextConfig) *Array {
	normed := RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, l.IsSliding, mask, cfg)
	attnOut = RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := Add(x, attnOut)

	normed = RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
	mlpOut := l.MLP.forward(normed)
	mlpOut = RMSNorm(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	return Add(h, mlpOut)
}

func (a *Attention) forward(x *Array, c Cache, B, L int32, isSliding bool, mask *Array, cfg *TextConfig) *Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Virtual transpose [B,L,H*D] → [B,H,L,D] via stride manipulation.
	// Strides: batch = L*H*D (full sequence), head = D (adjacent heads in memory),
	// seq = H*D (jump one full row of heads), elem = 1 (contiguous within head).
	q = AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	// Q/K normalization
	q = RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	k = RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)

	// RoPE with appropriate theta
	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}
	q = RoPE(q, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	k = RoPE(k, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())

	// Update cache
	k, v = c.Update(k, v, int(L))

	// GQA: repeat K/V heads
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = RepeatKV(k, repeatFactor)
		v = RepeatKV(v, repeatFactor)
	}

	// Scaled dot-product attention
	var out *Array
	if mask != nil {
		out = ScaledDotProductAttentionWithMask(q, k, v, mask, cfg.Scale)
	} else {
		out = ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	}
	out = Reshape(Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) forward(x *Array) *Array {
	gate := getCompiledGELU().Call(m.GateProj.Forward(x))[0]
	return m.DownProj.Forward(Mul(gate, m.UpProj.Forward(x)))
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
func (m *GemmaModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	adapter := &LoRAAdapter{
		Layers: make(map[string]*LoRALinear),
		Config: cfg,
	}

	for i, layer := range m.Layers {
		prefix := fmt.Sprintf("model.layers.%d.self_attn", i)
		for _, target := range cfg.TargetKeys {
			var proj *Linear
			switch target {
			case "q_proj":
				proj = layer.Attention.QProj
			case "k_proj":
				proj = layer.Attention.KProj
			case "v_proj":
				proj = layer.Attention.VProj
			case "o_proj":
				proj = layer.Attention.OProj
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
