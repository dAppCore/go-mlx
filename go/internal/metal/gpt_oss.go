// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
)

type GptOssModel struct {
	EmbedTokens *Embedding
	Layers      []*GptOssDecoderLayer
	Norm        *RMSNormModule
	Output      *Linear
	Tok         *Tokenizer
	Cfg         *GptOssConfig
	modelType   string
}

type GptOssConfig struct {
	ModelType             string  `json:"model_type,omitempty"`
	HiddenSize            int32   `json:"hidden_size,omitempty"`
	NumHiddenLayers       int32   `json:"num_hidden_layers,omitempty"`
	IntermediateSize      int32   `json:"intermediate_size,omitempty"`
	NumAttentionHeads     int32   `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads,omitempty"`
	NumLocalExperts       int32   `json:"num_local_experts,omitempty"`
	NumExperts            int32   `json:"num_experts,omitempty"`
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

type GptOssDecoderLayer struct {
	Dense *Qwen3DecoderLayer
	MoE   *GptOssMoEBlock
}

type GptOssMoEBlock struct {
	Router        *Qwen3MoERouter
	Experts       []*GptOssExpert
	SwitchExperts *MoESwiGLUExperts
}

type GptOssExpert struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

func (cfg *GptOssConfig) expertCount() int {
	if cfg.NumLocalExperts > 0 {
		return int(cfg.NumLocalExperts)
	}
	if cfg.NumExperts > 0 {
		return int(cfg.NumExperts)
	}
	return 8
}

func (cfg *GptOssConfig) topK() int {
	if cfg.NumExpertsPerTok > 0 {
		return int(cfg.NumExpertsPerTok)
	}
	return 2
}

func (l *GptOssDecoderLayer) isMoELayer() bool {
	return l.MoE != nil && l.MoE.Router != nil && len(l.MoE.Experts) > 0
}

func parseGptOssConfig(data []byte) (*GptOssConfig, error) {
	var cfg GptOssConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("gpt_oss.parseConfig", "parse config", nil)
	}
	var wrapper struct {
		Quantization       *QuantizationConfig `json:"quantization"`
		QuantizationConfig *QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gpt_oss.parseConfig", "parse nested config", nil)
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
		cfg.VocabSize = 201088
	}
	return &cfg, nil
}

func LoadGptOss(modelPath string) (*GptOssModel, error) {
	root := resolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load config", err)
	}
	data := []byte(str)
	cfg, err := parseGptOssConfig(data)
	if err != nil {
		return nil, core.E("gpt_oss.Load", "parse config", err)
	}
	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load tokenizer", err)
	}
	weights, err := loadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load weights", err)
	}
	w := func(name string) *Array { return resolveWeight(weights, name) }
	q := cfg.Quantization
	if q != nil {
		core.Info("gpt_oss: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
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
	m := &GptOssModel{
		EmbedTokens: embed,
		Layers:      make([]*GptOssDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   "gpt_oss",
	}
	numExperts := cfg.expertCount()
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &GptOssDecoderLayer{
			Dense: &Qwen3DecoderLayer{
				InputNorm:    &RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
				PostAttnNorm: &RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
				Attention: &Qwen3Attention{
					QProj: linear(w(p+".self_attn.q_proj.weight"), w(p+".self_attn.q_proj.scales"), w(p+".self_attn.q_proj.biases"), w(p+".self_attn.q_proj.bias")),
					KProj: linear(w(p+".self_attn.k_proj.weight"), w(p+".self_attn.k_proj.scales"), w(p+".self_attn.k_proj.biases"), w(p+".self_attn.k_proj.bias")),
					VProj: linear(w(p+".self_attn.v_proj.weight"), w(p+".self_attn.v_proj.scales"), w(p+".self_attn.v_proj.biases"), w(p+".self_attn.v_proj.bias")),
					OProj: linear(w(p+".self_attn.o_proj.weight"), w(p+".self_attn.o_proj.scales"), w(p+".self_attn.o_proj.biases"), w(p+".self_attn.o_proj.bias")),
				},
				MLP: nil,
			},
		}
		isMoE := cfg.SparseStep <= 0 || (i%cfg.SparseStep) == (cfg.SparseStep-1)
		if isMoE && numExperts > 0 {
			block := &GptOssMoEBlock{}
			block.Router = gptOssLoadRouter(weights, int(i), q)
			block.Experts = make([]*GptOssExpert, numExperts)
			for e := 0; e < numExperts; e++ {
				block.Experts[e] = gptOssLoadExpert(w, int(i), e)
			}
			block.SwitchExperts, _ = gptOssSwitchExperts(block.Experts)
			layer.MoE = block
		} else {
			dw := gptOssDenseMLPWeights(w, int(i))
			layer.Dense.MLP = &Qwen3MLP{
				GateProj: linear(dw.gateWeight, dw.gateScales, dw.gateBiases, dw.gateBias),
				UpProj:   linear(dw.upWeight, dw.upScales, dw.upBiases, dw.upBias),
				DownProj: linear(dw.downWeight, dw.downScales, dw.downBiases, dw.downBias),
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
		"arch", "gpt_oss", "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
		"experts", numExperts, "topk", cfg.topK(),
	)
	return m, nil
}

type gptOssDenseWeights struct {
	gateWeight, gateScales, gateBiases, gateBias *Array
	upWeight, upScales, upBiases, upBias         *Array
	downWeight, downScales, downBiases, downBias *Array
}

func gptOssDenseMLPWeights(w func(string) *Array, layerIdx int) gptOssDenseWeights {
	p := core.Sprintf("model.layers.%d.mlp", layerIdx)
	return gptOssDenseWeights{
		gateWeight: w(p + ".gate_proj.weight"), gateScales: w(p + ".gate_proj.scales"),
		gateBiases: w(p + ".gate_proj.biases"), gateBias: w(p + ".gate_proj.bias"),
		upWeight: w(p + ".up_proj.weight"), upScales: w(p + ".up_proj.scales"),
		upBiases: w(p + ".up_proj.biases"), upBias: w(p + ".up_proj.bias"),
		downWeight: w(p + ".down_proj.weight"), downScales: w(p + ".down_proj.scales"),
		downBiases: w(p + ".down_proj.biases"), downBias: w(p + ".down_proj.bias"),
	}
}

func gptOssLoadRouter(weights map[string]*Array, layerIdx int, q *QuantizationConfig) *Qwen3MoERouter {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp", layerIdx),
		core.Sprintf("model.layers.%d.moe", layerIdx),
	}
	suffixes := []string{".gate", ".router", ".gate_proj"}
	for _, prefix := range prefixes {
		for _, suffix := range suffixes {
			name := prefix + suffix
			if w := resolveWeight(weights, name+".weight"); w != nil {
				router := &Qwen3MoERouter{Weight: w}
				router.Scales = resolveWeight(weights, name+".scales")
				router.Biases = resolveWeight(weights, name+".biases")
				if q != nil {
					router.GroupSize = q.GroupSize
					router.Bits = q.Bits
				}
				return router
			}
		}
	}
	return &Qwen3MoERouter{}
}

func gptOssLoadExpert(w func(string) *Array, layerIdx, expertIdx int) *GptOssExpert {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp.experts.%d", layerIdx, expertIdx),
		core.Sprintf("model.layers.%d.moe.experts.%d", layerIdx, expertIdx),
	}
	for _, p := range prefixes {
		if wt := w(p + ".gate_proj.weight"); wt != nil {
			return &GptOssExpert{
				GateProj: NewLinear(wt, w(p+".gate_proj.bias")),
				UpProj:   NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
				DownProj: NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
			}
		}
	}
	return &GptOssExpert{}
}

func gptOssSwitchExperts(experts []*GptOssExpert) (*MoESwiGLUExperts, bool) {
	gate := make([]*Linear, 0, len(experts))
	up := make([]*Linear, 0, len(experts))
	down := make([]*Linear, 0, len(experts))
	for _, expert := range experts {
		if expert == nil {
			return nil, false
		}
		gate = append(gate, expert.GateProj)
		up = append(up, expert.UpProj)
		down = append(down, expert.DownProj)
	}
	return newMoESwiGLUExpertsFromLinears(gate, up, down)
}

func (m *GptOssModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *GptOssModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	var shapeBuf [maxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]
	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		hNext := gptOssDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}
	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
}

func gptOssDecoderLayerForward(l *GptOssDecoderLayer, x *Array, c Cache, B, L int32, mask *Array, cfg *GptOssConfig) *Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.forward(normed, c, B, L, mask, gptOssToQwen3Config(cfg))
	Free(normed)
	h := Add(x, attnOut)
	Free(attnOut)
	normed2 := l.Dense.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	if !l.isMoELayer() && l.Dense.MLP != nil {
		mlpOut := l.Dense.MLP.forward(normed2)
		Free(normed2)
		result := Add(h, mlpOut)
		Free(h, mlpOut)
		return result
	}
	if mlpOut, ok := moeSwiGLUForward(normed2, l.MoE.Router, cfg.topK(), l.MoE.SwitchExperts); ok {
		Free(normed2)
		result := Add(h, mlpOut)
		Free(h, mlpOut)
		return result
	}
	result := Add(h, normed2)
	Free(h, normed2)
	return result
}

func gptOssToQwen3Config(cfg *GptOssConfig) *Qwen3Config {
	if cfg == nil {
		return nil
	}
	return &Qwen3Config{
		HiddenSize: cfg.HiddenSize, NumHiddenLayers: cfg.NumHiddenLayers,
		NumAttentionHeads: cfg.NumAttentionHeads, NumKeyValueHeads: cfg.NumKeyValueHeads,
		HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
		RMSNormEps: cfg.RMSNormEps, RopeTheta: cfg.RopeTheta,
		MaxPositionEmbeddings: cfg.MaxPositionEmbeddings, Scale: cfg.Scale,
	}
}

func (m *GptOssModel) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}

func (m *GptOssModel) NumLayers() int { return len(m.Layers) }

func (m *GptOssModel) Tokenizer() *Tokenizer { return m.Tok }

func (m *GptOssModel) ModelType() string { return m.modelType }

func (m *GptOssModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
	cfg = normalizeLoRAConfig(cfg)
	adapter := &LoRAAdapter{Layers: make(map[string]*LoRALinear), Config: cfg, Model: m}
	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *Linear
			var key string
			switch target {
			case "q_proj":
				proj, key = layer.Dense.Attention.QProj, core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "k_proj":
				proj, key = layer.Dense.Attention.KProj, core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "v_proj":
				proj, key = layer.Dense.Attention.VProj, core.Sprintf("model.layers.%d.self_attn.%s", i, target)
			case "o_proj":
				proj, key = layer.Dense.Attention.OProj, core.Sprintf("model.layers.%d.self_attn.%s", i, target)
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

func closeGptOss(m *GptOssModel) {
	if m == nil {
		return
	}
	freeEmbedding(m.EmbedTokens)
	freeRMSNorm(m.Norm)
	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		freeLinear(m.Output)
	}
	for _, layer := range m.Layers {
		if layer == nil || layer.Dense == nil {
			continue
		}
		if layer.Dense.Attention != nil {
			freeLinear(layer.Dense.Attention.QProj)
			freeLinear(layer.Dense.Attention.KProj)
			freeLinear(layer.Dense.Attention.VProj)
			freeLinear(layer.Dense.Attention.OProj)
		}
		freeRMSNorm(layer.Dense.InputNorm)
		freeRMSNorm(layer.Dense.PostAttnNorm)
		if layer.Dense.MLP != nil {
			freeLinear(layer.Dense.MLP.GateProj)
			freeLinear(layer.Dense.MLP.UpProj)
			freeLinear(layer.Dense.MLP.DownProj)
		}
		if layer.MoE != nil {
			if layer.MoE.Router != nil {
				Free(layer.MoE.Router.Weight, layer.MoE.Router.Scales, layer.MoE.Router.Biases)
			}
			freeMoESwiGLUExperts(layer.MoE.SwitchExperts)
			for _, expert := range layer.MoE.Experts {
				freeLinear(expert.GateProj)
				freeLinear(expert.UpProj)
				freeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}
