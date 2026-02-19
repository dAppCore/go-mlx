//go:build darwin && arm64

package model

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"

	"forge.lthn.ai/core/go-mlx"
	"forge.lthn.ai/core/go-mlx/cache"
	"forge.lthn.ai/core/go-mlx/tokenizer"
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
	EmbedTokens *mlx.Embedding
	Layers      []*Qwen3DecoderLayer
	Norm        *mlx.RMSNormModule
	Output      *mlx.Linear

	Tok *tokenizer.Tokenizer
	Cfg *Qwen3Config
}

// Qwen3DecoderLayer is a single transformer block.
// Qwen 3 uses standard pre-norm residual: norm→attn→add, norm→mlp→add.
type Qwen3DecoderLayer struct {
	InputNorm   *mlx.RMSNormModule // Pre-attention norm
	PostAttnNorm *mlx.RMSNormModule // Pre-MLP norm (confusingly named post_attention_layernorm)
	Attention   *Qwen3Attention
	MLP         *Qwen3MLP
}

// Qwen3Attention implements Qwen 3 GQA with Q/K RMS normalization.
type Qwen3Attention struct {
	QProj *mlx.Linear
	KProj *mlx.Linear
	VProj *mlx.Linear
	OProj *mlx.Linear
	QNorm *mlx.RMSNormModule
	KNorm *mlx.RMSNormModule
}

// Qwen3MLP is the SwiGLU feed-forward network: down(silu(gate(x)) * up(x)).
type Qwen3MLP struct {
	GateProj *mlx.Linear
	UpProj   *mlx.Linear
	DownProj *mlx.Linear
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

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("qwen3: load tokenizer: %w", err)
	}

	// Load weights from all safetensors files
	weights := make(map[string]*mlx.Array)
	matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
	for _, path := range matches {
		for name, arr := range mlx.LoadSafetensors(path) {
			weights[name] = arr
		}
	}

	w := func(name string) *mlx.Array { return resolveWeight(weights, name) }

	// Quantization setup
	q := cfg.Quantization
	if q != nil {
		slog.Info("qwen3: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(prefix string) *mlx.Linear {
		weight := w(prefix + ".weight")
		scales := w(prefix + ".scales")
		biases := w(prefix + ".biases")
		bias := w(prefix + ".bias")
		if scales != nil && q != nil {
			return mlx.NewQuantizedLinear(weight, scales, biases, bias, q.GroupSize, q.Bits)
		}
		return mlx.NewLinear(weight, bias)
	}

	// Embedding
	embed := &mlx.Embedding{Weight: w("model.embed_tokens.weight")}
	if embedScales := w("model.embed_tokens.scales"); embedScales != nil && q != nil {
		embed.Scales = embedScales
		embed.Biases = w("model.embed_tokens.biases")
		embed.GroupSize = q.GroupSize
		embed.Bits = q.Bits
	}

	m := &Qwen3Model{
		EmbedTokens: embed,
		Layers:      make([]*Qwen3DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &mlx.RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := fmt.Sprintf("model.layers.%d", i)
		m.Layers[i] = &Qwen3DecoderLayer{
			InputNorm:    &mlx.RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
			PostAttnNorm: &mlx.RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
			Attention: &Qwen3Attention{
				QProj: linear(p + ".self_attn.q_proj"),
				KProj: linear(p + ".self_attn.k_proj"),
				VProj: linear(p + ".self_attn.v_proj"),
				OProj: linear(p + ".self_attn.o_proj"),
				QNorm: &mlx.RMSNormModule{Weight: w(p + ".self_attn.q_norm.weight")},
				KNorm: &mlx.RMSNormModule{Weight: w(p + ".self_attn.k_norm.weight")},
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
			m.Output = mlx.NewQuantizedLinear(lmHeadWeight, lmHeadScales, w("lm_head.biases"), nil, q.GroupSize, q.Bits)
		} else {
			m.Output = mlx.NewLinear(lmHeadWeight, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}

	// Materialise all weights onto Metal
	var allArrays []*mlx.Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	mlx.Materialize(allArrays...)

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
func (m *Qwen3Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	shape := tokens.Shape()
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		h = layer.forward(h, caches[i], B, L, m.Cfg)
	}

	return m.Output.Forward(m.Norm.Forward(h, m.Cfg.RMSNormEps))
}

func (l *Qwen3DecoderLayer) forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Qwen3Config) *mlx.Array {
	// Pre-attention norm → attention → residual add
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, cfg)
	h := mlx.Add(x, attnOut)

	// Pre-MLP norm → MLP → residual add
	normed = l.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.forward(normed)
	return mlx.Add(h, mlpOut)
}

func (a *Qwen3Attention) forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Qwen3Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim]
	q = mlx.AsStrided(q, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	k = mlx.AsStrided(k, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	v = mlx.AsStrided(v, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)

	// Q/K RMS normalization (Qwen 3 has this)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	// RoPE — single theta for all layers (no sliding window)
	q = mlx.RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	k = mlx.RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())

	// Update KV cache
	k, v = c.Update(k, v, int(L))

	// GQA: repeat K/V heads to match Q heads
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = mlx.RepeatKV(k, repeatFactor)
		v = mlx.RepeatKV(v, repeatFactor)
	}

	// Scaled dot-product attention
	out := mlx.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

// forward computes SwiGLU: down(silu(gate(x)) * up(x)).
func (m *Qwen3MLP) forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(m.GateProj.Forward(x))
	return m.DownProj.Forward(mlx.Mul(gate, m.UpProj.Forward(x)))
}

// NewCache creates per-layer KV caches. Qwen 3 uses global attention only.
func (m *Qwen3Model) NewCache() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *Qwen3Model) NumLayers() int { return len(m.Layers) }

// Tokenizer returns the model's tokenizer.
func (m *Qwen3Model) Tokenizer() *tokenizer.Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *Qwen3Model) ModelType() string { return "qwen3" }

// ApplyLoRA wraps target projection layers with LoRA adapters.
func (m *Qwen3Model) ApplyLoRA(cfg mlx.LoRAConfig) *mlx.LoRAAdapter {
	adapter := &mlx.LoRAAdapter{
		Layers: make(map[string]*mlx.LoRALinear),
		Config: cfg,
	}

	for i, layer := range m.Layers {
		prefix := fmt.Sprintf("model.layers.%d.self_attn", i)
		for _, target := range cfg.TargetKeys {
			var proj *mlx.Linear
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
				lora := mlx.NewLoRALinear(proj, cfg.Rank, cfg.Alpha)
				proj.LoRA = lora
				adapter.Layers[prefix+"."+target] = lora
			}
		}
	}

	return adapter
}
