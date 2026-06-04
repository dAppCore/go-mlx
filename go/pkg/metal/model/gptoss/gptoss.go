// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

type GptOssModel struct {
	EmbedTokens *metal.Embedding
	Layers      []*GptOssDecoderLayer
	Norm        *metal.RMSNormModule
	Output      *metal.Linear
	Tok         *metal.Tokenizer
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

	Quantization *metal.QuantizationConfig `json:"-"`
	Scale        float32                   `json:"-"`
}

type GptOssDecoderLayer struct {
	Dense *metal.DenseDecoderLayer
	MoE   *GptOssMoEBlock
}

type GptOssMoEBlock struct {
	Router        *metal.MoERouter
	Experts       []*GptOssExpert
	SwitchExperts *metal.MoESwiGLUExperts
}

type GptOssExpert struct {
	GateProj *metal.Linear
	UpProj   *metal.Linear
	DownProj *metal.Linear
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

// MoETextRuntimeAvailable reports whether the native selected-expert decode
// kernels are linked for every layer (metal.MoETextRuntimeReporter).
func (m *GptOssModel) MoETextRuntimeAvailable() bool {
	if m == nil {
		return false
	}
	return metal.MoETextLayersRuntimeAvailable(m.Layers, func(layer *GptOssDecoderLayer) metal.MoETextLayerParts {
		if layer == nil {
			return metal.MoETextLayerParts{}
		}
		var router *metal.MoERouter
		var switchExperts *metal.MoESwiGLUExperts
		if layer.MoE != nil {
			router = layer.MoE.Router
			switchExperts = layer.MoE.SwitchExperts
		}
		return metal.MoETextLayerParts{
			Dense:         layer.Dense,
			IsMoE:         layer.isMoELayer(),
			Router:        router,
			SwitchExperts: switchExperts,
			OK:            true,
		}
	})
}

// MoETextDecodeFamily returns the canonical family token used in unavailable
// diagnostics (metal.MoETextRuntimeReporter).
func (m *GptOssModel) MoETextDecodeFamily() string { return "gpt_oss" }

func parseGptOssConfig(data []byte) (*GptOssConfig, error) {
	var cfg GptOssConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("gpt_oss.parseConfig", "parse config", nil)
	}
	var wrapper struct {
		Quantization       *metal.QuantizationConfig `json:"quantization"`
		QuantizationConfig *metal.QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gpt_oss.parseConfig", "parse nested config", nil)
	}
	cfg.ModelType = metal.NormalizeProbeModelType(cfg.ModelType)
	cfg.Quantization = metal.FirstQuantization(wrapper.Quantization, wrapper.QuantizationConfig)
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
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load config", err)
	}
	data := []byte(str)
	cfg, err := parseGptOssConfig(data)
	if err != nil {
		return nil, core.E("gpt_oss.Load", "parse config", err)
	}
	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load tokenizer", err)
	}
	weights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gpt_oss.Load", "load weights", err)
	}
	w := func(name string) *metal.Array { return metal.ResolveWeight(weights, name) }
	q := cfg.Quantization
	if q != nil {
		core.Info("gpt_oss: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(weight, scales, biases, bias *metal.Array) *metal.Linear {
		if scales != nil {
			groupSize, bits := 0, 0
			if q != nil {
				groupSize = q.GroupSize
				bits = q.Bits
			}
			return metal.NewQuantizedLinear(weight, scales, biases, bias, groupSize, bits)
		}
		return metal.NewLinear(weight, bias)
	}
	embed := &metal.Embedding{Weight: w("model.embed_tokens.weight")}
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
		Norm:        &metal.RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   "gpt_oss",
	}
	numExperts := cfg.expertCount()
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &GptOssDecoderLayer{
			Dense: &metal.DenseDecoderLayer{
				InputNorm:    &metal.RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
				PostAttnNorm: &metal.RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
				Attention: &metal.GQAAttention{
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
			for e := range numExperts {
				block.Experts[e] = gptOssLoadExpert(w, int(i), e)
			}
			block.SwitchExperts, _ = gptOssSwitchExperts(block.Experts)
			layer.MoE = block
		} else {
			dw := gptOssDenseMLPWeights(w, int(i))
			layer.Dense.MLP = &metal.SiLUMLP{
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
			m.Output = metal.NewQuantizedLinear(lmHeadWeight, lmHeadScales, w("lm_head.biases"), nil, groupSize, bits)
		} else {
			m.Output = metal.NewLinear(lmHeadWeight, nil)
		}
	} else {
		m.Output = m.EmbedTokens.AsLinear()
	}
	var allArrays []*metal.Array
	for _, a := range weights {
		allArrays = append(allArrays, a)
	}
	metal.Materialize(allArrays...)
	core.Info("model loaded",
		"arch", "gpt_oss", "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
		"experts", numExperts, "topk", cfg.topK(),
	)
	return m, nil
}

type gptOssDenseWeights struct {
	gateWeight, gateScales, gateBiases, gateBias *metal.Array
	upWeight, upScales, upBiases, upBias         *metal.Array
	downWeight, downScales, downBiases, downBias *metal.Array
}

func gptOssDenseMLPWeights(w func(string) *metal.Array, layerIdx int) gptOssDenseWeights {
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

func gptOssLoadRouter(weights map[string]*metal.Array, layerIdx int, q *metal.QuantizationConfig) *metal.MoERouter {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp", layerIdx),
		core.Sprintf("model.layers.%d.moe", layerIdx),
	}
	suffixes := []string{".gate", ".router", ".gate_proj"}
	for _, prefix := range prefixes {
		for _, suffix := range suffixes {
			name := prefix + suffix
			if w := metal.ResolveWeight(weights, name+".weight"); w != nil {
				router := &metal.MoERouter{Weight: w}
				router.Scales = metal.ResolveWeight(weights, name+".scales")
				router.Biases = metal.ResolveWeight(weights, name+".biases")
				if q != nil {
					router.GroupSize = q.GroupSize
					router.Bits = q.Bits
				}
				return router
			}
		}
	}
	return &metal.MoERouter{}
}

func gptOssLoadExpert(w func(string) *metal.Array, layerIdx, expertIdx int) *GptOssExpert {
	prefixes := []string{
		core.Sprintf("model.layers.%d.mlp.experts.%d", layerIdx, expertIdx),
		core.Sprintf("model.layers.%d.moe.experts.%d", layerIdx, expertIdx),
	}
	for _, p := range prefixes {
		if wt := w(p + ".gate_proj.weight"); wt != nil {
			return &GptOssExpert{
				GateProj: metal.NewLinear(wt, w(p+".gate_proj.bias")),
				UpProj:   metal.NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
				DownProj: metal.NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
			}
		}
	}
	return &GptOssExpert{}
}

func gptOssSwitchExperts(experts []*GptOssExpert) (*metal.MoESwiGLUExperts, bool) {
	gate := make([]*metal.Linear, 0, len(experts))
	up := make([]*metal.Linear, 0, len(experts))
	down := make([]*metal.Linear, 0, len(experts))
	for _, expert := range experts {
		if expert == nil {
			return nil, false
		}
		gate = append(gate, expert.GateProj)
		up = append(up, expert.UpProj)
		down = append(down, expert.DownProj)
	}
	return metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
}

func (m *GptOssModel) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *GptOssModel) ForwardMasked(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]
	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		hNext := gptOssDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		metal.Free(h)
		h = hNext
	}
	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	return out
}

func gptOssDecoderLayerForward(l *GptOssDecoderLayer, x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, cfg *GptOssConfig) *metal.Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.Forward(normed, c, B, L, mask, gptOssToQwen3Config(cfg))
	metal.Free(normed)
	h := metal.Add(x, attnOut)
	metal.Free(attnOut)
	normed2 := l.Dense.PostAttnNorm.Forward(h, cfg.RMSNormEps)
	if !l.isMoELayer() && l.Dense.MLP != nil {
		mlpOut := l.Dense.MLP.Forward(normed2)
		metal.Free(normed2)
		result := metal.Add(h, mlpOut)
		metal.Free(h, mlpOut)
		return result
	}
	if mlpOut, ok := metal.MoESwiGLUForward(normed2, l.MoE.Router, cfg.topK(), l.MoE.SwitchExperts); ok {
		metal.Free(normed2)
		result := metal.Add(h, mlpOut)
		metal.Free(h, mlpOut)
		return result
	}
	result := metal.Add(h, normed2)
	metal.Free(h, normed2)
	return result
}

func gptOssToQwen3Config(cfg *GptOssConfig) *metal.DenseConfig {
	if cfg == nil {
		return nil
	}
	return &metal.DenseConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:            cfg.HiddenSize,
			NumHiddenLayers:       cfg.NumHiddenLayers,
			NumAttentionHeads:     cfg.NumAttentionHeads,
			NumKeyValueHeads:      cfg.NumKeyValueHeads,
			HeadDim:               cfg.HeadDim,
			VocabSize:             cfg.VocabSize,
			RMSNormEps:            cfg.RMSNormEps,
			MaxPositionEmbeddings: cfg.MaxPositionEmbeddings,
		},
		RopeTheta: cfg.RopeTheta,
		Scale:     cfg.Scale,
	}
}

func (m *GptOssModel) NewCache() []metal.Cache {
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewKVCache()
	}
	return caches
}

func (m *GptOssModel) NumLayers() int { return len(m.Layers) }

func (m *GptOssModel) Tokenizer() *metal.Tokenizer { return m.Tok }

func (m *GptOssModel) ModelType() string { return m.modelType }

func (m *GptOssModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	cfg = metal.NormalizeLoRAConfig(cfg)
	adapter := &metal.LoRAAdapter{Layers: make(map[string]*metal.LoRALinear), Config: cfg, Model: m}
	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *metal.Linear
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
				lora := metal.NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
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
	metal.FreeEmbedding(m.EmbedTokens)
	metal.FreeRMSNorm(m.Norm)
	if m.Output != nil && m.Output.Weight != nil &&
		(m.EmbedTokens == nil || m.Output.Weight != m.EmbedTokens.Weight) {
		metal.FreeLinear(m.Output)
	}
	for _, layer := range m.Layers {
		if layer == nil || layer.Dense == nil {
			continue
		}
		if layer.Dense.Attention != nil {
			metal.FreeLinear(layer.Dense.Attention.QProj)
			metal.FreeLinear(layer.Dense.Attention.KProj)
			metal.FreeLinear(layer.Dense.Attention.VProj)
			metal.FreeLinear(layer.Dense.Attention.OProj)
		}
		metal.FreeRMSNorm(layer.Dense.InputNorm)
		metal.FreeRMSNorm(layer.Dense.PostAttnNorm)
		if layer.Dense.MLP != nil {
			metal.FreeLinear(layer.Dense.MLP.GateProj)
			metal.FreeLinear(layer.Dense.MLP.UpProj)
			metal.FreeLinear(layer.Dense.MLP.DownProj)
		}
		if layer.MoE != nil {
			if layer.MoE.Router != nil {
				metal.Free(layer.MoE.Router.Weight, layer.MoE.Router.Scales, layer.MoE.Router.Biases)
			}
			metal.FreeMoESwiGLUExperts(layer.MoE.SwitchExperts)
			for _, expert := range layer.MoE.Experts {
				metal.FreeLinear(expert.GateProj)
				metal.FreeLinear(expert.UpProj)
				metal.FreeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}

func (m *GptOssModel) CloseModel() { closeGptOss(m) }

func (m *GptOssModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(m.Cfg.VocabSize)
	info.HiddenSize = int(m.Cfg.HiddenSize)
	info.ContextLength = int(m.Cfg.MaxPositionEmbeddings)
	if m.Cfg.Quantization != nil {
		info.QuantBits = m.Cfg.Quantization.Bits
		info.QuantGroup = m.Cfg.Quantization.GroupSize
	}
}

func init() {
	metal.RegisterModelLoader("gpt_oss", func(modelPath string, _ []byte) (metal.InternalModel, error) {
		return LoadGptOss(modelPath)
	})
}
