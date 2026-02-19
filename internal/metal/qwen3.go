//go:build darwin && arm64

package metal

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
)

// Qwen3Config holds Qwen 3 model configuration.
type Qwen3Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`

	Quantization *QuantizationConfig `json:"-"`
	Scale        float32             `json:"-"` // 1/sqrt(head_dim)
}

// Qwen3Model is the Qwen 3 text model.
type Qwen3Model struct {
	EmbedTokens *Embedding
	Layers      []*Qwen3DecoderLayer
	Norm        *RMSNormModule
	Output      *Linear

	Tok *Tokenizer
	Cfg *Qwen3Config
}

// Qwen3DecoderLayer is a single transformer block.
// Qwen 3 uses standard pre-norm residual: norm→attn→add, norm→mlp→add.
type Qwen3DecoderLayer struct {
	InputNorm   *RMSNormModule // Pre-attention norm
	PostAttnNorm *RMSNormModule // Pre-MLP norm (confusingly named post_attention_layernorm)
	Attention   *Qwen3Attention
	MLP         *Qwen3MLP
}

// Qwen3Attention implements Qwen 3 GQA with Q/K RMS normalization.
type Qwen3Attention struct {
	QProj *Linear
	KProj *Linear
	VProj *Linear
	OProj *Linear
	QNorm *RMSNormModule
	KNorm *RMSNormModule
}

// Qwen3MLP is the SwiGLU feed-forward network: down(silu(gate(x)) * up(x)).
type Qwen3MLP struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

func parseQwen3Config(data []byte) (*Qwen3Config, error) {
	var cfg Qwen3Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	// Top-level quantization
	var wrapper struct {
		Quantization *QuantizationConfig `json:"quantization"`
	}
	json.Unmarshal(data, &wrapper)
	cfg.Quantization = wrapper.Quantization

	// Compute scale
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// Defaults
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 151936
	}

	return &cfg, nil
}

// LoadQwen3 loads a Qwen 3 model from a safetensors directory.
func LoadQwen3(modelPath string) (*Qwen3Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3: load config: %w", err)
	}

	cfg, err := parseQwen3Config(data)
	if err != nil {
		return nil, fmt.Errorf("qwen3: parse config: %w", err)
	}

	tok, err := LoadTokenizer(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3: load tokenizer: %w", err)
	}

	// Load weights from all safetensors files
	weights := make(map[string]*Array)
	matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	for _, path := range matches {
		for name, arr := range LoadSafetensors(path) {
			weights[name] = arr
		}
	}

	w := func(name string) *Array { return resolveWeight(weights, name) }

	// Quantization setup
	q := cfg.Quantization
	if q != nil {
		slog.Info("qwen3: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(prefix string) *Linear {
		weight := w(prefix + ".weight")
		scales := w(prefix + ".scales")
		biases := w(prefix + ".biases")
		bias := w(prefix + ".bias")
		if scales != nil && q != nil {
			return NewQuantizedLinear(weight, scales, biases, bias, q.GroupSize, q.Bits)
		}
		return NewLinear(weight, bias)
	}

	// Embedding
	embed := &Embedding{Weight: w("model.embed_tokens.weight")}
	if embedScales := w("model.embed_tokens.scales"); embedScales != nil && q != nil {
		embed.Scales = embedScales
		embed.Biases = w("model.embed_tokens.biases")
		embed.GroupSize = q.GroupSize
		embed.Bits = q.Bits
	}

	m := &Qwen3Model{
		EmbedTokens: embed,
		Layers:      make([]*Qwen3DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := fmt.Sprintf("model.layers.%d", i)
		m.Layers[i] = &Qwen3DecoderLayer{
			InputNorm:    &RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
			PostAttnNorm: &RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
			Attention: &Qwen3Attention{
				QProj: linear(p + ".self_attn.q_proj"),
				KProj: linear(p + ".self_attn.k_proj"),
				VProj: linear(p + ".self_attn.v_proj"),
				OProj: linear(p + ".self_attn.o_proj"),
				QNorm: &RMSNormModule{Weight: w(p + ".self_attn.q_norm.weight")},
				KNorm: &RMSNormModule{Weight: w(p + ".self_attn.k_norm.weight")},
			},
			MLP: &Qwen3MLP{
				GateProj: linear(p + ".mlp.gate_proj"),
				UpProj:   linear(p + ".mlp.up_proj"),
				DownProj: linear(p + ".mlp.down_proj"),
			},
		}
	}

	// Output head — Qwen 3 has tie_word_embeddings=false, so lm_head is separate
	lmHeadWeight := w("lm_head.weight")
	if lmHeadWeight != nil {
		lmHeadScales := w("lm_head.scales")
		if lmHeadScales != nil && q != nil {
			m.Output = NewQuantizedLinear(lmHeadWeight, lmHeadScales, w("lm_head.biases"), nil, q.GroupSize, q.Bits)
		} else {
			m.Output = NewLinear(lmHeadWeight, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}

	// Materialise all weights onto Metal
	var allArrays []*Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	Materialize(allArrays...)

	slog.Info("qwen3: model loaded",
		"layers", cfg.NumHiddenLayers,
		"hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads,
		"kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim,
		"vocab", cfg.VocabSize,
	)

	return m, nil
}

// Forward runs the Qwen 3 forward pass.
// Unlike Gemma, Qwen does NOT scale embeddings by sqrt(hidden_size).
func (m *Qwen3Model) Forward(tokens *Array, caches []Cache) *Array {
	shape := tokens.Shape()
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		h = layer.forward(h, caches[i], B, L, m.Cfg)
	}

	return m.Output.Forward(m.Norm.Forward(h, m.Cfg.RMSNormEps))
}

func (l *Qwen3DecoderLayer) forward(x *Array, c Cache, B, L int32, cfg *Qwen3Config) *Array {
	// Pre-attention norm → attention → residual add
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, cfg)
	h := Add(x, attnOut)

	// Pre-MLP norm → MLP → residual add
	normed = l.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.forward(normed)
	return Add(h, mlpOut)
}

func (a *Qwen3Attention) forward(x *Array, c Cache, B, L int32, cfg *Qwen3Config) *Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim]
	q = AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	// Q/K RMS normalization (Qwen 3 has this)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	// RoPE — single theta for all layers (no sliding window)
	q = RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	k = RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())

	// Update KV cache
	k, v = c.Update(k, v, int(L))

	// GQA: repeat K/V heads to match Q heads
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = RepeatKV(k, repeatFactor)
		v = RepeatKV(v, repeatFactor)
	}

	// Scaled dot-product attention
	out := ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	out = Reshape(Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// forward computes SwiGLU: down(silu(gate(x)) * up(x)).
func (m *Qwen3MLP) forward(x *Array) *Array {
	gate := SiLU(m.GateProj.Forward(x))
	return m.DownProj.Forward(Mul(gate, m.UpProj.Forward(x)))
}

// NewCache creates per-layer KV caches. Qwen 3 uses global attention only.
func (m *Qwen3Model) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *Qwen3Model) NumLayers() int { return len(m.Layers) }

// Tokenizer returns the model's tokenizer.
func (m *Qwen3Model) Tokenizer() *Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *Qwen3Model) ModelType() string { return "qwen3" }

// ApplyLoRA wraps target projection layers with LoRA adapters.
func (m *Qwen3Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
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
				lora := NewLoRALinear(proj, cfg.Rank, cfg.Alpha)
				proj.LoRA = lora
				adapter.Layers[prefix+"."+target] = lora
			}
		}
	}

	return adapter
}
