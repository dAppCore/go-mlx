// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
)

type KimiModel struct {
	EmbedTokens *Embedding
	Layers      []*KimiDecoderLayer
	Norm        *RMSNormModule
	Output      *Linear
	Tok         *Tokenizer
	Cfg         *KimiConfig
	modelType   string
}

type KimiConfig struct {
	ModelType             string  `json:"model_type,omitempty"`
	HiddenSize            int32   `json:"hidden_size,omitempty"`
	NumHiddenLayers       int32   `json:"num_hidden_layers,omitempty"`
	IntermediateSize      int32   `json:"intermediate_size,omitempty"`
	NumAttentionHeads     int32   `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads,omitempty"`
	NumExperts            int32   `json:"num_experts,omitempty"`
	NumLocalExperts       int32   `json:"num_local_experts,omitempty"`
	NRoutedExperts        int32   `json:"n_routed_experts,omitempty"`
	NumExpertsPerTok      int32   `json:"num_experts_per_tok,omitempty"`
	MoETopK               int32   `json:"moe_topk,omitempty"`
	HeadDim               int32   `json:"head_dim,omitempty"`
	VocabSize             int32   `json:"vocab_size,omitempty"`
	RMSNormEps            float32 `json:"rms_norm_eps,omitempty"`
	RopeTheta             float32 `json:"rope_theta,omitempty"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings,omitempty"`
	SparseStep            int32   `json:"decoder_sparse_step,omitempty"`

	Quantization *QuantizationConfig `json:"-"`
	Scale        float32             `json:"-"`
}

type KimiDecoderLayer struct {
	Dense *Qwen3DecoderLayer
	MoE   *KimiMoEBlock
}

type KimiMoEBlock struct {
	Router        *Qwen3MoERouter
	Experts       []*KimiExpert
	SwitchExperts *MoESwiGLUExperts
}

type KimiExpert struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

func (cfg *KimiConfig) expertCount() int {
	for _, v := range []int32{cfg.NumExperts, cfg.NumLocalExperts, cfg.NRoutedExperts} {
		if v > 0 {
			return int(v)
		}
	}
	return 8
}

func (cfg *KimiConfig) topK() int {
	if cfg.NumExpertsPerTok > 0 {
		return int(cfg.NumExpertsPerTok)
	}
	if cfg.MoETopK > 0 {
		return int(cfg.MoETopK)
	}
	return 2
}

func (l *KimiDecoderLayer) isMoELayer() bool {
	return l.MoE != nil && l.MoE.Router != nil && len(l.MoE.Experts) > 0
}

func parseKimiConfig(data []byte) (*KimiConfig, error) {
	var cfg KimiConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("kimi.parseConfig", "parse config", nil)
	}
	var wrapper struct {
		Quantization       *QuantizationConfig `json:"quantization"`
		QuantizationConfig *QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("kimi.parseConfig", "parse nested config", nil)
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
		cfg.VocabSize = 128256
	}
	return &cfg, nil
}

func LoadKimi(modelPath string) (*KimiModel, error) {
	root := ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("kimi.Load", "load config", err)
	}
	data := []byte(str)
	cfg, err := parseKimiConfig(data)
	if err != nil {
		return nil, core.E("kimi.Load", "parse config", err)
	}
	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("kimi.Load", "load tokenizer", err)
	}
	weights, err := LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("kimi.Load", "load weights", err)
	}
	w := func(name string) *Array { return ResolveWeight(weights, name) }
	q := cfg.Quantization
	if q != nil {
		core.Info("kimi: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
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
	m := &KimiModel{
		EmbedTokens: embed,
		Layers:      make([]*KimiDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   "kimi",
	}
	numExperts := cfg.expertCount()
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &KimiDecoderLayer{
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
			block := &KimiMoEBlock{}
			block.Router = kimiLoadRouter(weights, int(i), q)
			block.Experts = make([]*KimiExpert, numExperts)
			for e := range numExperts {
				block.Experts[e] = kimiLoadExpert(w, int(i), e)
			}
			block.SwitchExperts, _ = kimiSwitchExperts(block.Experts)
			layer.MoE = block
		} else {
			dw := kimiDenseMLPWeights(w, int(i))
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
		"arch", "kimi", "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
		"experts", numExperts, "topk", cfg.topK(),
	)
	return m, nil
}

type kimiDenseWeights struct {
	gateWeight, gateScales, gateBiases, gateBias *Array
	upWeight, upScales, upBiases, upBias         *Array
	downWeight, downScales, downBiases, downBias *Array
}

func kimiDenseMLPWeights(w func(string) *Array, layerIdx int) kimiDenseWeights {
	p := core.Sprintf("model.layers.%d.mlp", layerIdx)
	return kimiDenseWeights{
		gateWeight: w(p + ".gate_proj.weight"), gateScales: w(p + ".gate_proj.scales"),
		gateBiases: w(p + ".gate_proj.biases"), gateBias: w(p + ".gate_proj.bias"),
		upWeight: w(p + ".up_proj.weight"), upScales: w(p + ".up_proj.scales"),
		upBiases: w(p + ".up_proj.biases"), upBias: w(p + ".up_proj.bias"),
		downWeight: w(p + ".down_proj.weight"), downScales: w(p + ".down_proj.scales"),
		downBiases: w(p + ".down_proj.biases"), downBias: w(p + ".down_proj.bias"),
	}
}

func kimiLoadRouter(weights map[string]*Array, layerIdx int, q *QuantizationConfig) *Qwen3MoERouter {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp", layerIdx),
		core.Sprintf("model.layers.%d.moe", layerIdx),
	}
	suffixes := []string{".gate", ".router", ".gate_proj", ".router.proj"}
	for _, prefix := range prefixes {
		for _, suffix := range suffixes {
			name := prefix + suffix
			if w := ResolveWeight(weights, name+".weight"); w != nil {
				router := &Qwen3MoERouter{Weight: w}
				router.Scales = ResolveWeight(weights, name+".scales")
				router.Biases = ResolveWeight(weights, name+".biases")
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

func kimiLoadExpert(w func(string) *Array, layerIdx, expertIdx int) *KimiExpert {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp.experts.%d", layerIdx, expertIdx),
		core.Sprintf("model.layers.%d.moe.experts.%d", layerIdx, expertIdx),
	}
	for _, p := range prefixes {
		if wt := w(p + ".gate_proj.weight"); wt != nil {
			return &KimiExpert{
				GateProj: NewLinear(wt, w(p+".gate_proj.bias")),
				UpProj:   NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
				DownProj: NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
			}
		}
	}
	return &KimiExpert{}
}

func kimiSwitchExperts(experts []*KimiExpert) (*MoESwiGLUExperts, bool) {
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

func (m *KimiModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *KimiModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	var shapeBuf [MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]
	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		hNext := kimiDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}
	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
}

func kimiDecoderLayerForward(l *KimiDecoderLayer, x *Array, c Cache, B, L int32, mask *Array, cfg *KimiConfig) *Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.forward(normed, c, B, L, mask, kimiToQwen3Config(cfg))
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

func kimiToQwen3Config(cfg *KimiConfig) *Qwen3Config {
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

func (m *KimiModel) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}

func (m *KimiModel) NumLayers() int { return len(m.Layers) }

func (m *KimiModel) Tokenizer() *Tokenizer { return m.Tok }

func (m *KimiModel) ModelType() string { return m.modelType }

func (m *KimiModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
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

func closeKimi(m *KimiModel) {
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
				FreeLinear(expert.GateProj)
				FreeLinear(expert.UpProj)
				FreeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}
