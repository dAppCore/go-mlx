// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
)

type MixtralModel struct {
	EmbedTokens *Embedding
	Layers      []*MixtralDecoderLayer
	Norm        *RMSNormModule
	Output      *Linear
	Tok         *Tokenizer
	Cfg         *MixtralConfig
	modelType   string
}

type MixtralConfig struct {
	ModelType             string  `json:"model_type,omitempty"`
	HiddenSize            int32   `json:"hidden_size,omitempty"`
	NumHiddenLayers       int32   `json:"num_hidden_layers,omitempty"`
	IntermediateSize      int32   `json:"intermediate_size,omitempty"`
	NumAttentionHeads     int32   `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads,omitempty"`
	NumLocalExperts       int32   `json:"num_local_experts,omitempty"`
	NumExpertsPerTok      int32   `json:"num_experts_per_tok,omitempty"`
	HeadDim               int32   `json:"head_dim,omitempty"`
	VocabSize             int32   `json:"vocab_size,omitempty"`
	RMSNormEps            float32 `json:"rms_norm_eps,omitempty"`
	RopeTheta             float32 `json:"rope_theta,omitempty"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings,omitempty"`
	SparseStep            int32   `json:"decoder_sparse_step,omitempty"`

	Quantization *QuantizationConfig `json:"-"`
	Scale        float32             `json:"-"`
}

type MixtralDecoderLayer struct {
	Dense *DenseDecoderLayer
	MoE   *MixtralMoEBlock
}

type MixtralMoEBlock struct {
	Router           *MoERouter
	Experts          []*MixtralExpert
	SwitchExperts    *MoESwiGLUExperts
	NumLocalExperts  int32
	NumExpertsPerTok int32
}

type MixtralExpert struct {
	W1 *Linear
	W2 *Linear
	W3 *Linear
}

func (l *MixtralDecoderLayer) isMoELayer() bool {
	return l.MoE != nil && l.MoE.Router != nil && len(l.MoE.Experts) > 0
}

func parseMixtralConfig(data []byte) (*MixtralConfig, error) {
	var cfg MixtralConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("mixtral.parseConfig", "parse config", nil)
	}

	var wrapper struct {
		Quantization       *QuantizationConfig `json:"quantization"`
		QuantizationConfig *QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("mixtral.parseConfig", "parse nested config", nil)
	}
	cfg.ModelType = normalizeProbeModelType(cfg.ModelType)
	cfg.Quantization = firstQwen3Quantization(wrapper.Quantization, wrapper.QuantizationConfig)

	if cfg.HeadDim == 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim > 0 {
		cfg.Scale = float32(1.0)
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 32000
	}
	if cfg.NumLocalExperts == 0 {
		cfg.NumLocalExperts = 8
	}
	if cfg.NumExpertsPerTok == 0 {
		cfg.NumExpertsPerTok = 2
	}

	return &cfg, nil
}

func LoadMixtral(modelPath string) (*MixtralModel, error) {
	root := ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("mixtral.Load", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseMixtralConfig(data)
	if err != nil {
		return nil, core.E("mixtral.Load", "parse config", err)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("mixtral.Load", "load tokenizer", err)
	}

	weights, err := LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("mixtral.Load", "load weights", err)
	}

	w := func(name string) *Array { return ResolveWeight(weights, name) }

	q := cfg.Quantization
	if q != nil {
		core.Info("mixtral: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(weight, scales, biases, bias *Array) *Linear {
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

	m := &MixtralModel{
		EmbedTokens: embed,
		Layers:      make([]*MixtralDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   "mixtral",
	}

	isMoELayer := mixtralMoELayerMask(cfg)

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &MixtralDecoderLayer{
			Dense: &DenseDecoderLayer{
				InputNorm:    &RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
				PostAttnNorm: &RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
				Attention: &GQAAttention{
					QProj: linear(w(p+".self_attn.q_proj.weight"), w(p+".self_attn.q_proj.scales"), w(p+".self_attn.q_proj.biases"), w(p+".self_attn.q_proj.bias")),
					KProj: linear(w(p+".self_attn.k_proj.weight"), w(p+".self_attn.k_proj.scales"), w(p+".self_attn.k_proj.biases"), w(p+".self_attn.k_proj.bias")),
					VProj: linear(w(p+".self_attn.v_proj.weight"), w(p+".self_attn.v_proj.scales"), w(p+".self_attn.v_proj.biases"), w(p+".self_attn.v_proj.bias")),
					OProj: linear(w(p+".self_attn.o_proj.weight"), w(p+".self_attn.o_proj.scales"), w(p+".self_attn.o_proj.biases"), w(p+".self_attn.o_proj.bias")),
				},
				MLP: nil,
			},
		}

		if isMoELayer[i] {
			block := &MixtralMoEBlock{
				NumLocalExperts:  cfg.NumLocalExperts,
				NumExpertsPerTok: cfg.NumExpertsPerTok,
			}
			block.Router = mixtralLoadRouter(weights, int(i), q)
			block.Experts = make([]*MixtralExpert, cfg.NumLocalExperts)
			for e := int32(0); e < cfg.NumLocalExperts; e++ {
				block.Experts[e] = mixtralLoadExpert(w, int(i), int(e))
			}
			block.SwitchExperts, _ = mixtralSwitchExperts(block.Experts)
			layer.MoE = block
		} else {
			denseWeights := mixtralDenseMLPWeights(w, int(i))
			layer.Dense.MLP = &SiLUMLP{
				GateProj: linear(denseWeights.gateWeight, denseWeights.gateScales, denseWeights.gateBiases, denseWeights.gateBias),
				UpProj:   linear(denseWeights.upWeight, denseWeights.upScales, denseWeights.upBiases, denseWeights.upBias),
				DownProj: linear(denseWeights.downWeight, denseWeights.downScales, denseWeights.downBiases, denseWeights.downBias),
			}
		}

		m.Layers[i] = layer
	}

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
		"arch", "mixtral", "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
		"experts", cfg.NumLocalExperts, "experts_per_tok", cfg.NumExpertsPerTok,
	)

	return m, nil
}

type mixtralDenseWeights struct {
	gateWeight, gateScales, gateBiases, gateBias *Array
	upWeight, upScales, upBiases, upBias         *Array
	downWeight, downScales, downBiases, downBias *Array
}

func mixtralDenseMLPWeights(w func(string) *Array, layerIdx int) mixtralDenseWeights {
	p := core.Sprintf("model.layers.%d.mlp", layerIdx)
	return mixtralDenseWeights{
		gateWeight: w(p + ".gate_proj.weight"),
		gateScales: w(p + ".gate_proj.scales"),
		gateBiases: w(p + ".gate_proj.biases"),
		gateBias:   w(p + ".gate_proj.bias"),
		upWeight:   w(p + ".up_proj.weight"),
		upScales:   w(p + ".up_proj.scales"),
		upBiases:   w(p + ".up_proj.biases"),
		upBias:     w(p + ".up_proj.bias"),
		downWeight: w(p + ".down_proj.weight"),
		downScales: w(p + ".down_proj.scales"),
		downBiases: w(p + ".down_proj.biases"),
		downBias:   w(p + ".down_proj.bias"),
	}
}

func mixtralMoELayerMask(cfg *MixtralConfig) []bool {
	mask := make([]bool, cfg.NumHiddenLayers)
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		if cfg.SparseStep <= 0 {
			mask[i] = true
		} else {
			mask[i] = (i % cfg.SparseStep) == (cfg.SparseStep - 1)
		}
	}
	return mask
}

func mixtralLoadRouter(weights map[string]*Array, layerIdx int, q *QuantizationConfig) *MoERouter {
	p := core.Sprintf("model.layers.%d.block_sparse_moe", layerIdx)
	router := &MoERouter{}
	for _, suffix := range []string{".gate", ".router", ".gate_proj"} {
		name := p + suffix
		if w := ResolveWeight(weights, name+".weight"); w != nil {
			router.Weight = w
			router.Scales = ResolveWeight(weights, name+".scales")
			router.Biases = ResolveWeight(weights, name+".biases")
			if q != nil {
				router.GroupSize = q.GroupSize
				router.Bits = q.Bits
			}
			return router
		}
	}
	return router
}

func mixtralLoadExpert(w func(string) *Array, layerIdx, expertIdx int) *MixtralExpert {
	p := core.Sprintf("model.layers.%d.block_sparse_moe.experts.%d", layerIdx, expertIdx)
	return &MixtralExpert{
		W1: NewLinear(w(p+".w1.weight"), w(p+".w1.bias")),
		W2: NewLinear(w(p+".w2.weight"), w(p+".w2.bias")),
		W3: NewLinear(w(p+".w3.weight"), w(p+".w3.bias")),
	}
}

func mixtralSwitchExperts(experts []*MixtralExpert) (*MoESwiGLUExperts, bool) {
	gate := make([]*Linear, 0, len(experts))
	up := make([]*Linear, 0, len(experts))
	down := make([]*Linear, 0, len(experts))
	for _, expert := range experts {
		if expert == nil {
			return nil, false
		}
		gate = append(gate, expert.W1)
		up = append(up, expert.W3)
		down = append(down, expert.W2)
	}
	return newMoESwiGLUExpertsFromLinears(gate, up, down)
}

func (m *MixtralModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *MixtralModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	var shapeBuf [MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		hNext := mixtralDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}

	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
}

func mixtralDecoderLayerForward(l *MixtralDecoderLayer, x *Array, c Cache, B, L int32, mask *Array, cfg *MixtralConfig) *Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.forward(normed, c, B, L, mask, mixtralToQwen3Config(cfg))
	Free(normed)
	h := Add(x, attnOut)
	Free(attnOut)

	normed2 := l.Dense.PostAttnNorm.Forward(h, cfg.RMSNormEps)

	if !l.isMoELayer() && l.Dense.MLP != nil {
		mlpOut := l.Dense.MLP.Forward(normed2)
		Free(normed2)
		result := Add(h, mlpOut)
		Free(h, mlpOut)
		return result
	}

	if mlpOut, ok := moeSwiGLUForward(normed2, l.MoE.Router, int(cfg.NumExpertsPerTok), l.MoE.SwitchExperts); ok {
		Free(normed2)
		result := Add(h, mlpOut)
		Free(h, mlpOut)
		return result
	}

	// Diagnostic fallback: keep the layer inspectable until every production
	// sparse path for this architecture is enabled.
	result := Add(h, normed2)
	Free(h, normed2)
	return result
}

func mixtralToQwen3Config(cfg *MixtralConfig) *DenseConfig {
	if cfg == nil {
		return nil
	}
	return &DenseConfig{
		HiddenSize:            cfg.HiddenSize,
		NumHiddenLayers:       cfg.NumHiddenLayers,
		NumAttentionHeads:     cfg.NumAttentionHeads,
		NumKeyValueHeads:      cfg.NumKeyValueHeads,
		HeadDim:               cfg.HeadDim,
		VocabSize:             cfg.VocabSize,
		RMSNormEps:            cfg.RMSNormEps,
		RopeTheta:             cfg.RopeTheta,
		MaxPositionEmbeddings: cfg.MaxPositionEmbeddings,
		Scale:                 cfg.Scale,
	}
}

func (m *MixtralModel) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}

func (m *MixtralModel) NumLayers() int { return len(m.Layers) }

func (m *MixtralModel) Tokenizer() *Tokenizer { return m.Tok }

func (m *MixtralModel) ModelType() string { return m.modelType }

func (m *MixtralModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	cfg = normalizeLoRAConfig(cfg)
	adapter := &LoRAAdapter{
		Layers: make(map[string]*LoRALinear),
		Config: cfg,
		Model:  m,
	}
	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *Linear
			var key string
			switch target {
			case "q_proj":
				proj = layer.Dense.Attention.QProj
				key = core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "k_proj":
				proj = layer.Dense.Attention.KProj
				key = core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "v_proj":
				proj = layer.Dense.Attention.VProj
				key = core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "o_proj":
				proj = layer.Dense.Attention.OProj
				key = core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "gate_proj", "up_proj", "down_proj":
				if !layer.isMoELayer() && layer.Dense.MLP != nil {
					switch target {
					case "gate_proj":
						proj = layer.Dense.MLP.GateProj
					case "up_proj":
						proj = layer.Dense.MLP.UpProj
					case "down_proj":
						proj = layer.Dense.MLP.DownProj
					}
					key = core.Sprintf("model.layers.%d.mlp.%s", i, target)
				}
			}
			if proj != nil {
				lora := NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
				proj.LoRA = lora
				adapter.Layers[key] = lora
			}
		}
	}
	return adapter
}

func closeMixtral(m *MixtralModel) {
	if m == nil {
		return
	}
	FreeEmbedding(m.EmbedTokens)
	FreeRMSNorm(m.Norm)

	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		FreeLinear(m.Output)
	}

	for _, layer := range m.Layers {
		if layer == nil || layer.Dense == nil {
			continue
		}
		if layer.Dense.Attention != nil {
			FreeLinear(layer.Dense.Attention.QProj)
			FreeLinear(layer.Dense.Attention.KProj)
			FreeLinear(layer.Dense.Attention.VProj)
			FreeLinear(layer.Dense.Attention.OProj)
		}
		FreeRMSNorm(layer.Dense.InputNorm)
		FreeRMSNorm(layer.Dense.PostAttnNorm)
		if layer.Dense.MLP != nil {
			FreeLinear(layer.Dense.MLP.GateProj)
			FreeLinear(layer.Dense.MLP.UpProj)
			FreeLinear(layer.Dense.MLP.DownProj)
		}
		if layer.MoE != nil {
			if layer.MoE.Router != nil {
				Free(layer.MoE.Router.Weight, layer.MoE.Router.Scales, layer.MoE.Router.Biases)
			}
			freeMoESwiGLUExperts(layer.MoE.SwitchExperts)
			for _, expert := range layer.MoE.Experts {
				FreeLinear(expert.W1)
				FreeLinear(expert.W2)
				FreeLinear(expert.W3)
			}
		}
	}
	m.Layers = nil
}
