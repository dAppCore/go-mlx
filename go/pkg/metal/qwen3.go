// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// Qwen3Model is the dense Llama-family text model used by Qwen 2/3, Llama,
// Mistral, Hermes, Granite, Phi, and GLM-style checkpoints. Qwen 3 adds optional
// Q/K RMS normalization. It composes the SDK's neutral dense-layer algos
// (DenseConfig, DenseDecoderLayer, GQAAttention, SiLUMLP).
type Qwen3Model struct {
	EmbedTokens *Embedding
	Layers      []*DenseDecoderLayer
	Norm        *RMSNormModule
	Output      *Linear

	Tok       *Tokenizer
	Cfg       *DenseConfig
	modelType string // "qwen2", "qwen3", "llama", "mistral", "hermes", "granite", "phi", or "glm"
}

// LoadQwen3 loads a Qwen 2/3, Llama, Mistral, Hermes, Granite, or Phi-style
// dense decoder model from a safetensors directory. These families share the
// pre-norm SwiGLU GQA topology; Qwen 3 adds optional Q/K RMS normalization.
func LoadQwen3(modelPath string) (*Qwen3Model, error) {
	root := ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load config", err)
	}
	data := []byte(str)

	cfg, err := ParseDenseConfig(data)
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "parse config", err)
	}
	if cfg.IsQwen36Hybrid() {
		return nil, core.E("qwen3.LoadQwen3", Qwen36NativeGuardMessage(cfg.ModelType), nil)
	}
	if cfg.IsMoE() {
		return nil, core.E("qwen3.LoadQwen3", "qwen3_moe sparse expert routing is not implemented in the native Go loader yet", nil)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load tokenizer", err)
	}

	weights, err := LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load weights", err)
	}

	w := func(name string) *Array { return ResolveWeight(weights, name) }

	q := cfg.Quantization
	if q != nil {
		core.Info("qwen3: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(prefix string) *Linear {
		weight := w(prefix + ".weight")
		scales := w(prefix + ".scales")
		biases := w(prefix + ".biases")
		bias := w(prefix + ".bias")
		if scales != nil {
			groupSize, bits := 0, 0
			if q != nil {
				groupSize = q.GroupSize
				bits = q.Bits
			}
			return NewQuantizedLinear(weight, scales, biases, bias, groupSize, bits)
		}
		return NewLinear(weight, bias)
	}

	embed := &Embedding{Weight: w("model.embed_tokens.weight")}
	if embedScales := w("model.embed_tokens.scales"); embedScales != nil {
		embed.Scales = embedScales
		embed.Biases = w("model.embed_tokens.biases")
		if q != nil {
			embed.GroupSize = q.GroupSize
			embed.Bits = q.Bits
		}
	}

	// Preserve the architecture selected during top-level probing so configs
	// that rely on the `architectures` field (common for Llama checkpoints)
	// still get the correct runtime model type and chat template.
	detectedType := DetectDenseModelType(data, weights)

	m := &Qwen3Model{
		EmbedTokens: embed,
		Layers:      make([]*DenseDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   detectedType,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		m.Layers[i] = &DenseDecoderLayer{
			InputNorm:    &RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
			PostAttnNorm: &RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
			Attention: &GQAAttention{
				QProj: linear(p + ".self_attn.q_proj"),
				KProj: linear(p + ".self_attn.k_proj"),
				VProj: linear(p + ".self_attn.v_proj"),
				OProj: linear(p + ".self_attn.o_proj"),
				QNorm: &RMSNormModule{Weight: w(p + ".self_attn.q_norm.weight")},
				KNorm: &RMSNormModule{Weight: w(p + ".self_attn.k_norm.weight")},
			},
			MLP: &SiLUMLP{
				GateProj: linear(p + ".mlp.gate_proj"),
				UpProj:   linear(p + ".mlp.up_proj"),
				DownProj: linear(p + ".mlp.down_proj"),
			},
		}
	}

	// lm_head: Qwen3 has tie_word_embeddings=false; use tied embed_tokens as fallback
	lmHeadWeight := w("lm_head.weight")
	if lmHeadWeight != nil {
		lmHeadScales := w("lm_head.scales")
		if lmHeadScales != nil {
			groupSize, bits := 0, 0
			if q != nil {
				groupSize = q.GroupSize
				bits = q.Bits
			}
			m.Output = NewQuantizedLinear(lmHeadWeight, lmHeadScales, w("lm_head.biases"), nil, groupSize, bits)
		} else {
			m.Output = NewLinear(lmHeadWeight, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}

	var allArrays []*Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	Materialize(allArrays...)
	core.Info("model loaded",
		"arch", detectedType, "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
	)

	return m, nil
}

// Forward runs the Qwen 3 forward pass.
// Unlike Gemma, Qwen does NOT scale embeddings by sqrt(hidden_size).
func (m *Qwen3Model) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

// ForwardMasked runs the forward pass with an explicit attention mask.
// mask shape: [B, 1, L, L] — additive mask (0 = attend, -inf = ignore).
// When mask is nil, standard causal attention is used.
func (m *Qwen3Model) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	// Stack-allocated shape scratch — per-forward-pass hot path. Avoids
	// the per-call []int32 heap alloc from tokens.Shape().
	var shapeBuf [MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		hNext := layer.forward(h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}

	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
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

// NumQueryHeads reports the attention query-head count for KV/attention
// extraction (QueryHeadCounter). Zero when the config is unavailable.
func (m *Qwen3Model) NumQueryHeads() int {
	if m.Cfg != nil {
		return int(m.Cfg.NumAttentionHeads)
	}
	return 0
}

// ResolveLoRALinear resolves a LoRA-targetable projection by path
// (LoRALinearResolver). Returns nil for an unknown layer or path.
func (m *Qwen3Model) ResolveLoRALinear(layerIdx int, projPath string) *Linear {
	if layerIdx >= len(m.Layers) {
		return nil
	}
	layer := m.Layers[layerIdx]
	switch projPath {
	case "self_attn.q_proj":
		return layer.Attention.QProj
	case "self_attn.k_proj":
		return layer.Attention.KProj
	case "self_attn.v_proj":
		return layer.Attention.VProj
	case "self_attn.o_proj":
		return layer.Attention.OProj
	case "mlp.gate_proj":
		return layer.MLP.GateProj
	case "mlp.up_proj":
		return layer.MLP.UpProj
	case "mlp.down_proj":
		return layer.MLP.DownProj
	}
	return nil
}

// Tokenizer returns the model's tokenizer.
func (m *Qwen3Model) Tokenizer() *Tokenizer { return m.Tok }

// ModelType returns the architecture identifier ("qwen2", "qwen3", "llama",
// "mistral", "hermes", "granite", "phi", or "glm").
func (m *Qwen3Model) ModelType() string { return m.modelType }

// ApplyLoRA wraps target projection layers with LoRA adapters.
// Supports attention targets (q_proj, k_proj, v_proj, o_proj) and
// MLP targets (gate_proj, up_proj, down_proj).
func (m *Qwen3Model) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
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
