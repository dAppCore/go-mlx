// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// Qwen3Config holds Qwen 3 model configuration.
type Qwen3Config struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int32    `json:"hidden_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	IntermediateSize      int32    `json:"intermediate_size"`
	MoEIntermediateSize   int32    `json:"moe_intermediate_size"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	NumExperts            int32    `json:"num_experts"`
	NumExpertsPerTok      int32    `json:"num_experts_per_tok"`
	DecoderSparseStep     int32    `json:"decoder_sparse_step"`
	HeadDim               int32    `json:"head_dim"`
	VocabSize             int32    `json:"vocab_size"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	RopeTheta             float32  `json:"rope_theta"`
	PartialRotaryFactor   float32  `json:"partial_rotary_factor"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	LayerTypes            []string `json:"layer_types"`

	Quantization *QuantizationConfig `json:"-"`
	Scale        float32             `json:"-"` // 1/sqrt(head_dim)
}

// Qwen3Model is the dense Llama-family text model used by Qwen 2/3, Llama,
// Mistral, Hermes, Granite, Phi, and GLM-style checkpoints. Qwen 3 adds optional
// Q/K RMS normalization.
type Qwen3Model struct {
	EmbedTokens *Embedding
	Layers      []*Qwen3DecoderLayer
	Norm        *RMSNormModule
	Output      *Linear

	Tok       *Tokenizer
	Cfg       *Qwen3Config
	modelType string // "qwen2", "qwen3", "llama", "mistral", "hermes", "granite", "phi", or "glm"
}

// Qwen3DecoderLayer is a single transformer block.
// Qwen 3 uses standard pre-norm residual: norm→attn→add, norm→mlp→add.
type Qwen3DecoderLayer struct {
	InputNorm    *RMSNormModule // Pre-attention norm
	PostAttnNorm *RMSNormModule // Pre-MLP norm (confusingly named post_attention_layernorm)
	Attention    *Qwen3Attention
	MLP          *Qwen3MLP
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
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("qwen3.parseConfig", "parse config", nil)
	}

	var wrapper struct {
		TextConfig         *Qwen3Config        `json:"text_config"`
		Quantization       *QuantizationConfig `json:"quantization"`
		QuantizationConfig *QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("qwen3.parseConfig", "parse nested config", nil)
	}
	if wrapper.TextConfig != nil {
		cfg = mergeQwen3TextConfig(cfg, *wrapper.TextConfig)
	}
	cfg.ModelType = normalizeProbeModelType(cfg.ModelType)
	cfg.Quantization = firstQwen3Quantization(wrapper.Quantization, wrapper.QuantizationConfig, cfg.Quantization)

	// Compute scale when the config carries enough attention metadata.
	if cfg.HeadDim == 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim > 0 {
		cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	}

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

func mergeQwen3TextConfig(top, text Qwen3Config) Qwen3Config {
	if text.ModelType == "" {
		text.ModelType = top.ModelType
	}
	text.Quantization = firstQwen3Quantization(text.Quantization, top.Quantization)
	if text.VocabSize == 0 {
		text.VocabSize = top.VocabSize
	}
	if text.HiddenSize == 0 {
		text.HiddenSize = top.HiddenSize
	}
	if text.NumHiddenLayers == 0 {
		text.NumHiddenLayers = top.NumHiddenLayers
	}
	if text.IntermediateSize == 0 {
		text.IntermediateSize = top.IntermediateSize
	}
	if text.MoEIntermediateSize == 0 {
		text.MoEIntermediateSize = top.MoEIntermediateSize
	}
	if text.NumAttentionHeads == 0 {
		text.NumAttentionHeads = top.NumAttentionHeads
	}
	if text.NumKeyValueHeads == 0 {
		text.NumKeyValueHeads = top.NumKeyValueHeads
	}
	if text.NumExperts == 0 {
		text.NumExperts = top.NumExperts
	}
	if text.NumExpertsPerTok == 0 {
		text.NumExpertsPerTok = top.NumExpertsPerTok
	}
	if text.DecoderSparseStep == 0 {
		text.DecoderSparseStep = top.DecoderSparseStep
	}
	if text.HeadDim == 0 {
		text.HeadDim = top.HeadDim
	}
	if text.RMSNormEps == 0 {
		text.RMSNormEps = top.RMSNormEps
	}
	if text.RopeTheta == 0 {
		text.RopeTheta = top.RopeTheta
	}
	if text.PartialRotaryFactor == 0 {
		text.PartialRotaryFactor = top.PartialRotaryFactor
	}
	if text.MaxPositionEmbeddings == 0 {
		text.MaxPositionEmbeddings = top.MaxPositionEmbeddings
	}
	if len(text.LayerTypes) == 0 && len(top.LayerTypes) > 0 {
		text.LayerTypes = append([]string(nil), top.LayerTypes...)
	}
	return text
}

func firstQwen3Quantization(configs ...*QuantizationConfig) *QuantizationConfig {
	for _, cfg := range configs {
		if cfg != nil {
			return cfg
		}
	}
	return nil
}

func (cfg *Qwen3Config) IsMoE() bool {
	return cfg != nil && (cfg.ModelType == "qwen3_moe" || cfg.ModelType == "qwen3_6_moe" || cfg.NumExperts > 0 || cfg.NumExpertsPerTok > 0 || cfg.MoEIntermediateSize > 0)
}

func (cfg *Qwen3Config) IsQwen36Hybrid() bool {
	if cfg == nil {
		return false
	}
	switch normalizeProbeModelType(cfg.ModelType) {
	case "qwen3_6", "qwen3_6_moe":
		return true
	}
	for _, layerType := range cfg.LayerTypes {
		if normalizeQwen3LayerType(layerType) == "linear_attention" {
			return true
		}
	}
	return cfg.PartialRotaryFactor > 0 && cfg.PartialRotaryFactor < 1
}

func normalizeQwen3LayerType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	return core.Replace(value, ".", "_")
}

func qwen36NativeGuardMessage(modelType string) string {
	if normalizeProbeModelType(modelType) == "qwen3_6_moe" {
		return "qwen3_6_moe hybrid linear attention and sparse expert routing are not implemented in the native Go loader yet"
	}
	return "qwen3_6 hybrid linear attention is not implemented in the native Go loader yet"
}

func detectQwenModelType(configData []byte, weights map[string]*Array) string {
	if detected, err := probeModelType(configData); err == nil {
		switch detected {
		case "llama", "mistral", "hermes", "granite", "phi", "glm", "qwen2", "qwen3", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe":
			return detected
		}
	}

	if hasResolvedWeight(weights, "model.layers.0.self_attn.q_norm.weight") {
		return "qwen3"
	}
	return "qwen2"
}

// LoadQwen3 loads a Qwen 2/3, Llama, Mistral, Hermes, Granite, or Phi-style
// dense decoder model from a safetensors directory. These families share the
// pre-norm SwiGLU GQA topology; Qwen 3 adds optional Q/K RMS normalization.
func LoadQwen3(modelPath string) (*Qwen3Model, error) {
	root := resolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseQwen3Config(data)
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "parse config", err)
	}
	if cfg.IsQwen36Hybrid() {
		return nil, core.E("qwen3.LoadQwen3", qwen36NativeGuardMessage(cfg.ModelType), nil)
	}
	if cfg.IsMoE() {
		return nil, core.E("qwen3.LoadQwen3", "qwen3_moe sparse expert routing is not implemented in the native Go loader yet", nil)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load tokenizer", err)
	}

	weights, err := loadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("qwen3.LoadQwen3", "load weights", err)
	}

	w := func(name string) *Array { return resolveWeight(weights, name) }

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
	detectedType := detectQwenModelType(data, weights)

	m := &Qwen3Model{
		EmbedTokens: embed,
		Layers:      make([]*Qwen3DecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   detectedType,
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
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
	var shapeBuf [maxTensorRank]int32
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

func (l *Qwen3DecoderLayer) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *Qwen3Config) *Array {
	// Pre-attention norm → attention → residual add
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Attention.forward(normed, c, B, L, mask, cfg)
	Free(normed)
	h := Add(x, attnOut)
	Free(attnOut)

	// Pre-MLP norm → MLP → residual add
	normed2 := l.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.forward(normed2)
	Free(normed2)
	result := Add(h, mlpOut)
	Free(h, mlpOut)
	return result
}

func (a *Qwen3Attention) forward(x *Array, c Cache, B, L int32, mask *Array, cfg *Qwen3Config) *Array {
	qProj := a.QProj.Forward(x)
	kProj := a.KProj.Forward(x)
	vProj := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim] via stride manipulation.
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

	// Q/K RMS normalization (Qwen 3 has this; Qwen 2 does not)
	if a.QNorm != nil && a.QNorm.Weight != nil {
		oldQ := q
		q = a.QNorm.Forward(q, cfg.RMSNormEps)
		Free(oldQ)
	}
	if a.KNorm != nil && a.KNorm.Weight != nil {
		oldK := k
		k = a.KNorm.Forward(k, cfg.RMSNormEps)
		Free(oldK)
	}

	// RoPE — single theta for all layers (no sliding window)
	oldQ := q
	q = RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	Free(oldQ)
	oldK := k
	k = RoPE(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, c.Offset())
	Free(oldK)

	// Scaled dot-product attention
	var out *Array
	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if paged, ok := c.(*PagedKVCache); ok && L == 1 && mask == nil {
		oldK, oldV := k, v
		pages := paged.UpdatePages(k, v, int(L))
		Free(oldK, oldV)
		kPages, vPages := pages.Keys, pages.Values
		var repeatedPages []*Array
		if pagedStateNeedsMaterializedRepeat(pages, repeatFactor) {
			kPages, vPages, repeatedPages = repeatPagedState(pages, repeatFactor)
		}
		out = ScaledDotProductAttentionPaged(q, kPages, vPages, cfg.Scale)
		Free(repeatedPages...)
		pages.Free()
	} else {
		// Update KV cache — returns Slice views into cache buffer; free our pre-update handles.
		oldK, oldV := k, v
		k, v = c.Update(k, v, int(L))
		Free(oldK, oldV)

		// GQA: repeat K/V heads to match Q heads
		kAttn, vAttn := k, v
		if repeatFactor > 1 {
			kAttn = RepeatKV(k, repeatFactor)
			vAttn = RepeatKV(v, repeatFactor)
			Free(k, v) // Free Slice views from cache.Update; RepeatKV holds copies
		}

		if mask != nil {
			out = ScaledDotProductAttentionWithMask(q, kAttn, vAttn, mask, cfg.Scale)
		} else {
			out = ScaledDotProductAttention(q, kAttn, vAttn, cfg.Scale, L > 1)
		}
		Free(kAttn, vAttn) // Always free — when repeatFactor==1 this frees the Slice views
	}
	Free(q)

	// Rank-4 attention output transpose [B,H,L,D] → [B,L,H,D] — scalar-pass
	// Transpose4 form (eliminates the []int axes heap alloc).
	transposed := Transpose4(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	Free(transposed)
	result := a.OProj.Forward(reshaped)
	Free(reshaped)
	return result
}

// forward computes SwiGLU: down(silu(gate(x)) * up(x)).
func (m *Qwen3MLP) forward(x *Array) *Array {
	gateProj := m.GateProj.Forward(x)
	upProj := m.UpProj.Forward(x)
	activated := siluGateMul(gateProj, upProj)
	Free(gateProj, upProj)
	result := m.DownProj.Forward(activated)
	Free(activated)
	return result
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

// numQueryHeads reports the attention query-head count for KV/attention
// extraction (queryHeadCounter). Zero when the config is unavailable.
func (m *Qwen3Model) numQueryHeads() int {
	if m.Cfg != nil {
		return int(m.Cfg.NumAttentionHeads)
	}
	return 0
}

// resolveLoRALinear resolves a LoRA-targetable projection by path
// (loRALinearResolver). Returns nil for an unknown layer or path.
func (m *Qwen3Model) resolveLoRALinear(layerIdx int, projPath string) *Linear {
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
