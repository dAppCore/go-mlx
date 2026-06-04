// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"math"

	core "dappco.re/go"

	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// TextConfig holds Gemma 3 text model configuration.
type TextConfig struct {
	ModelType             string  `json:"model_type"`
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

	Quantization   *metal.QuantizationConfig `json:"-"` // Parsed separately from top-level
	Scale          float32                   `json:"-"` // Computed: 1/sqrt(head_dim)
	EmbeddingScale float32                   `json:"-"` // Computed: sqrt(hidden_size); cached to skip per-token math.Sqrt
}

// GemmaModel is the Gemma 3 text model.
type GemmaModel struct {
	EmbedTokens *metal.Embedding
	Layers      []*DecoderLayer
	Norm        *metal.RMSNormModule
	Output      *metal.Linear // Tied to EmbedTokens

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	NormScaled *metal.Array

	Tok *metal.Tokenizer
	Cfg *TextConfig

	modelType string
}

// DecoderLayer is a single transformer block.
type DecoderLayer struct {
	InputNorm    *metal.RMSNormModule
	Attention    *Attention
	PostAttnNorm *metal.RMSNormModule
	PreFFNorm    *metal.RMSNormModule
	MLP          *metal.MLP
	PostFFNorm   *metal.RMSNormModule

	// Precomputed scaled weights
	InputNormScaled    *metal.Array
	PostAttnNormScaled *metal.Array
	PreFFNormScaled    *metal.Array
	PostFFNormScaled   *metal.Array

	IsSliding bool
	LayerIdx  int32
}

// Attention implements Gemma 3 attention with Q/K normalization.
type Attention struct {
	QProj *metal.Linear
	KProj *metal.Linear
	VProj *metal.Linear
	OProj *metal.Linear
	QNorm *metal.RMSNormModule
	KNorm *metal.RMSNormModule

	QNormScaled *metal.Array
	KNormScaled *metal.Array
}

// parseConfig handles both flat and nested (text_config) Gemma 3 configs.
func parseConfig(data []byte) (*TextConfig, error) {
	// Try parsing text_config from multimodal wrapper
	var wrapper struct {
		TextConfig   TextConfig                `json:"text_config"`
		ModelType    string                    `json:"model_type"`
		Quantization *metal.QuantizationConfig `json:"quantization"`
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
	if cfg.ModelType == "" && wrapper.ModelType != "" {
		cfg.ModelType = wrapper.ModelType
	}

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
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma3"
	}
	if cfg.HiddenSize > 0 {
		cfg.EmbeddingScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
	}

	return &cfg, nil
}

// LoadGemma3 loads a Gemma 3 text model from a directory.
func LoadGemma3(modelPath string) (*GemmaModel, error) {
	root := metal.ResolveModelRoot(modelPath)
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
	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "load tokenizer", err)
	}

	weights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma3.LoadGemma3", "load weights", err)
	}

	weight := func(name string) *metal.Array { return metal.ResolveWeight(weights, name) }

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
	linear := func(prefix string) *metal.Linear {
		layerWeight := weight(prefix + ".weight")
		scales := weight(prefix + ".scales")
		biases := weight(prefix + ".biases")
		if scales != nil {
			groupSize, bits := 0, 0
			if quantConfig != nil {
				groupSize = quantConfig.GroupSize
				bits = quantConfig.Bits
			}
			return metal.NewQuantizedLinear(layerWeight, scales, biases, nil, groupSize, bits)
		}
		return metal.NewLinear(layerWeight, nil)
	}

	embed := &metal.Embedding{Weight: weight("model.embed_tokens.weight")}
	if embedScales := weight("model.embed_tokens.scales"); embedScales != nil {
		embed.Scales = embedScales
		embed.Biases = weight("model.embed_tokens.biases")
		if quantConfig != nil {
			embed.GroupSize = quantConfig.GroupSize
			embed.Bits = quantConfig.Bits
		}
	}

	gemmaModel := &GemmaModel{
		EmbedTokens: embed,
		Layers:      make([]*DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &metal.RMSNormModule{Weight: weight("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   cfg.ModelType,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("model.layers.%d", i)
		gemmaModel.Layers[i] = &DecoderLayer{
			InputNorm:    &metal.RMSNormModule{Weight: weight(prefix + ".input_layernorm.weight")},
			PostAttnNorm: &metal.RMSNormModule{Weight: weight(prefix + ".post_attention_layernorm.weight")},
			PreFFNorm:    &metal.RMSNormModule{Weight: weight(prefix + ".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &metal.RMSNormModule{Weight: weight(prefix + ".post_feedforward_layernorm.weight")},
			Attention: &Attention{
				QProj: linear(prefix + ".self_attn.q_proj"),
				KProj: linear(prefix + ".self_attn.k_proj"),
				VProj: linear(prefix + ".self_attn.v_proj"),
				OProj: linear(prefix + ".self_attn.o_proj"),
				QNorm: &metal.RMSNormModule{Weight: weight(prefix + ".self_attn.q_norm.weight")},
				KNorm: &metal.RMSNormModule{Weight: weight(prefix + ".self_attn.k_norm.weight")},
			},
			MLP: &metal.MLP{
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
		if lmHeadScales != nil {
			groupSize, bits := 0, 0
			if quantConfig != nil {
				groupSize = quantConfig.GroupSize
				bits = quantConfig.Bits
			}
			gemmaModel.Output = metal.NewQuantizedLinear(lmHeadWeight, lmHeadScales, weight("lm_head.biases"), nil, groupSize, bits)
		} else {
			gemmaModel.Output = metal.NewLinear(lmHeadWeight, nil)
		}
	} else {
		gemmaModel.Output = gemmaModel.EmbedTokens.AsLinear() // tied embeddings
	}

	var allArrays []*metal.Array
	for _, arr := range weights {
		allArrays = append(allArrays, arr)
	}
	metal.Materialize(allArrays...)
	precomputeScaledWeights(gemmaModel) // Gemma-style: weight → (1 + weight)

	return gemmaModel, nil
}

func precomputeScaledWeights(m *GemmaModel) {
	m.NormScaled = metal.AddScalar(m.Norm.Weight, 1.0)

	for _, layer := range m.Layers {
		layer.InputNormScaled = metal.AddScalar(layer.InputNorm.Weight, 1.0)
		layer.PostAttnNormScaled = metal.AddScalar(layer.PostAttnNorm.Weight, 1.0)
		layer.PreFFNormScaled = metal.AddScalar(layer.PreFFNorm.Weight, 1.0)
		layer.PostFFNormScaled = metal.AddScalar(layer.PostFFNorm.Weight, 1.0)
		layer.Attention.QNormScaled = metal.AddScalar(layer.Attention.QNorm.Weight, 1.0)
		layer.Attention.KNormScaled = metal.AddScalar(layer.Attention.KNorm.Weight, 1.0)
	}

	var scaled []*metal.Array
	scaled = append(scaled, m.NormScaled)
	for _, layer := range m.Layers {
		scaled = append(scaled, layer.InputNormScaled, layer.PostAttnNormScaled,
			layer.PreFFNormScaled, layer.PostFFNormScaled,
			layer.Attention.QNormScaled, layer.Attention.KNormScaled)
	}
	metal.Materialize(scaled...)
}

func isLayerSliding(layerIdx, pattern int32) bool {
	if pattern <= 0 {
		return false
	}
	return (layerIdx+1)%pattern != 0
}

// Forward runs the text model forward pass.
func (m *GemmaModel) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *GemmaModel) ForwardMasked(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	// Stack-allocated shape scratch — per-forward-pass hot path. Avoids
	// the per-call []int32 heap alloc from tokens.Shape().
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)
	h2 := metal.MulScalar(h, m.Cfg.EmbeddingScale)
	metal.Free(h)
	h = h2

	for i, layer := range m.Layers {
		hNext := layer.forward(h, caches[i], B, L, mask, m.Cfg)
		metal.Free(h)
		h = hNext
	}

	normed := metal.RMSNorm(h, m.NormScaled, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	return out
}

func (l *DecoderLayer) forward(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, cfg *TextConfig) *metal.Array {
	normed := metal.RMSNorm(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, l.IsSliding, mask, cfg)
	metal.Free(normed)
	attnOutNormed := metal.RMSNorm(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(x, attnOutNormed)
	metal.Free(attnOutNormed)

	normed2 := metal.RMSNorm(h, l.PreFFNormScaled, cfg.RMSNormEps)
	mlpOut := l.MLP.Forward(normed2)
	metal.Free(normed2)
	mlpOutNormed := metal.RMSNorm(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(mlpOut)
	result := metal.Add(h, mlpOutNormed)
	metal.Free(h, mlpOutNormed)
	return result
}

func (a *Attention) forward(x *metal.Array, c metal.Cache, B, L int32, isSliding bool, mask *metal.Array, cfg *TextConfig) *metal.Array {
	qProj := a.QProj.Forward(x)
	kProj := a.KProj.Forward(x)
	vProj := a.VProj.Forward(x)

	// Virtual transpose [B,L,H*D] → [B,H,L,D] via stride manipulation.
	// AsStrided creates a view (C refcount keeps source alive), so Free source after.
	q := metal.AsStrided(qProj, []int32{B, cfg.NumAttentionHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumAttentionHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumAttentionHeads * cfg.HeadDim), 1}, 0)
	metal.Free(qProj)
	k := metal.AsStrided(kProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	metal.Free(kProj)
	v := metal.AsStrided(vProj, []int32{B, cfg.NumKeyValueHeads, L, cfg.HeadDim},
		[]int64{int64(L * cfg.NumKeyValueHeads * cfg.HeadDim), int64(cfg.HeadDim), int64(cfg.NumKeyValueHeads * cfg.HeadDim), 1}, 0)
	metal.Free(vProj)

	// Q/K normalization
	oldQ := q
	q = metal.RMSNorm(q, a.QNormScaled, cfg.RMSNormEps)
	metal.Free(oldQ)
	oldK := k
	k = metal.RMSNorm(k, a.KNormScaled, cfg.RMSNormEps)
	metal.Free(oldK)

	// RoPE with appropriate theta
	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}
	oldQ = q
	q = metal.RoPE(q, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	metal.Free(oldQ)
	oldK = k
	k = metal.RoPE(k, int(cfg.HeadDim), false, ropeTheta, 1.0, c.Offset())
	metal.Free(oldK)

	// Scaled dot-product attention
	var out *metal.Array
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if paged, ok := c.(*metal.PagedKVCache); ok && L == 1 && mask == nil {
		oldK, oldV := k, v
		pages := paged.UpdatePages(k, v, int(L))
		metal.Free(oldK, oldV)
		kPages, vPages := pages.Keys, pages.Values
		var repeatedPages []*metal.Array
		if metal.PagedStateNeedsMaterializedRepeat(pages, repeatFactor) {
			kPages, vPages, repeatedPages = metal.RepeatPagedState(pages, repeatFactor)
		}
		out = metal.ScaledDotProductAttentionPaged(q, kPages, vPages, cfg.Scale)
		metal.Free(repeatedPages...)
		pages.Free()
	} else {
		// Update cache — returns Slice views into cache buffer; free our pre-update handles.
		oldK, oldV := k, v
		k, v = c.Update(k, v, int(L))
		metal.Free(oldK, oldV)

		// GQA: repeat K/V heads
		kAttn, vAttn := k, v
		if repeatFactor > 1 {
			kAttn = metal.RepeatKV(k, repeatFactor)
			vAttn = metal.RepeatKV(v, repeatFactor)
			metal.Free(k, v) // Free Slice views from cache.Update; RepeatKV holds copies
		}

		if mask != nil {
			out = metal.ScaledDotProductAttentionWithMask(q, kAttn, vAttn, mask, cfg.Scale)
		} else {
			out = metal.ScaledDotProductAttention(q, kAttn, vAttn, cfg.Scale, L > 1)
		}
		metal.Free(kAttn, vAttn) // Always free — when repeatFactor==1 this frees the Slice views
	}
	metal.Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — use the
	// scalar-pass Transpose4 form (eliminates the []int axes heap alloc).
	transposed := metal.Transpose4(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	metal.Free(transposed)
	result := a.OProj.Forward(reshaped)
	metal.Free(reshaped)
	return result
}

// NewCache creates per-layer caches for generation.
func (m *GemmaModel) NewCache() []metal.Cache {
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		if m.Layers[i].IsSliding {
			caches[i] = metal.NewRotatingKVCache(int(m.Cfg.SlidingWindow))
		} else {
			caches[i] = metal.NewKVCache()
		}
	}
	return caches
}

// NumLayers returns the number of transformer layers.
func (m *GemmaModel) NumLayers() int { return len(m.Layers) }

// NumQueryHeads reports the attention query-head count for KV/attention
// extraction (QueryHeadCounter). Zero when the config is unavailable.
func (m *GemmaModel) NumQueryHeads() int {
	if m.Cfg != nil {
		return int(m.Cfg.NumAttentionHeads)
	}
	return 0
}

// ResolveLoRALinear resolves a LoRA-targetable projection by path
// (LoRALinearResolver). Returns nil for an unknown layer or path.
func (m *GemmaModel) ResolveLoRALinear(layerIdx int, projPath string) *metal.Linear {
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
	}
	return nil
}

// Tokenizer returns the model's tokenizer.
func (m *GemmaModel) Tokenizer() *metal.Tokenizer { return m.Tok }

// ModelType returns the architecture identifier.
func (m *GemmaModel) ModelType() string {
	if m.modelType != "" {
		return m.modelType
	}
	return "gemma3"
}

// ApplyLoRA wraps target projection layers with LoRA adapters.
// Supports attention targets (q_proj, k_proj, v_proj, o_proj) and
// MLP targets (gate_proj, up_proj, down_proj).
func (m *GemmaModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	cfg = metal.NormalizeLoRAConfig(cfg)
	adapter := &metal.LoRAAdapter{
		Layers: make(map[string]*metal.LoRALinear),
		Config: cfg,
		Model:  m,
	}

	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *metal.Linear
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
				lora := metal.NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
				proj.LoRA = lora
				adapter.Layers[prefix+"."+target] = lora
			}
		}
	}

	return adapter
}
