// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

type Qwen3MoEModel struct {
	EmbedTokens *metal.Embedding
	Layers      []*Qwen3MoEDecoderLayer
	Norm        *metal.RMSNormModule
	Output      *metal.Linear
	Tok         *metal.Tokenizer
	Cfg         *metal.DenseConfig
	modelType   string
}

type Qwen3MoESharedExpert struct {
	GateProj *metal.Linear
	UpProj   *metal.Linear
	DownProj *metal.Linear
}

type Qwen3MoEExpert struct {
	GateProj *metal.Linear
	UpProj   *metal.Linear
	DownProj *metal.Linear
}

type Qwen3MoEBlock struct {
	Router           *metal.MoERouter
	SharedExpert     *Qwen3MoESharedExpert
	Experts          []*Qwen3MoEExpert
	SwitchExperts    *metal.MoESwiGLUExperts
	IntermediateSize int32
}

type Qwen3MoEDecoderLayer struct {
	Dense *metal.DenseDecoderLayer
	MoE   *Qwen3MoEBlock
}

func init() {
	metal.RegisterModelLoader("qwen3_moe", func(modelPath string, _ []byte) (metal.InternalModel, error) {
		model, err := LoadQwen3MoE(modelPath)
		if err != nil {
			return nil, core.E("model.loadModel", "load qwen3_moe native model", err)
		}
		return model, nil
	})
}

func (l *Qwen3MoEDecoderLayer) isMoELayer() bool {
	return l.MoE != nil && l.MoE.Router != nil && len(l.MoE.Experts) > 0
}

func (l *Qwen3MoEDecoderLayer) isDenseLayer() bool {
	return l.MoE == nil || l.MoE.Router == nil
}

// MoETextRuntimeAvailable reports whether the native selected-expert decode
// kernels are linked for every layer (metal.MoETextRuntimeReporter).
func (m *Qwen3MoEModel) MoETextRuntimeAvailable() bool {
	if m == nil {
		return false
	}
	return metal.MoETextLayersRuntimeAvailable(m.Layers, func(layer *Qwen3MoEDecoderLayer) metal.MoETextLayerParts {
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
func (m *Qwen3MoEModel) MoETextDecodeFamily() string { return "qwen3_moe" }

func (m *Qwen3MoEModel) FillModelInfo(info *metal.ModelInfo) {
	info.Architecture = m.ModelType()
	info.VocabSize = int(m.Cfg.VocabSize)
	info.NumLayers = int(m.Cfg.NumHiddenLayers)
	info.NumHeads = int(m.Cfg.NumAttentionHeads)
	info.HiddenSize = int(m.Cfg.HiddenSize)
	info.ContextLength = int(m.Cfg.MaxPositionEmbeddings)
	if m.Cfg.Quantization != nil {
		info.QuantBits = m.Cfg.Quantization.Bits
		info.QuantGroup = m.Cfg.Quantization.GroupSize
	}
}

func (m *Qwen3MoEModel) CloseModel() { closeQwen3MoE(m) }

func LoadQwen3MoE(modelPath string) (*Qwen3MoEModel, error) {
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load config", err)
	}
	data := []byte(str)

	cfg, err := metal.ParseDenseConfig(data)
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "parse config", err)
	}
	if cfg.IsQwen36Hybrid() {
		return nil, core.E("qwen3_moe.Load", metal.Qwen36NativeGuardMessage(cfg.ModelType), nil)
	}
	if !cfg.IsMoE() {
		return nil, core.E("qwen3_moe.Load", "config must have MoE metadata (num_experts, num_experts_per_tok, moe_intermediate_size)", nil)
	}

	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load tokenizer", err)
	}

	weights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load weights", err)
	}

	w := func(name string) *metal.Array { return metal.ResolveWeight(weights, name) }

	q := cfg.Quantization
	if q != nil {
		core.Info("qwen3_moe: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(weight, scales, biases, bias *metal.Array, prefix string) *metal.Linear {
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

	detectedType := metal.DetectDenseModelType(data, weights)

	m := &Qwen3MoEModel{
		EmbedTokens: embed,
		Layers:      make([]*Qwen3MoEDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &metal.RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   detectedType,
	}

	isMoELayer := qwen3MoELayerMask(cfg)

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &Qwen3MoEDecoderLayer{
			Dense: &metal.DenseDecoderLayer{
				InputNorm:    &metal.RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
				PostAttnNorm: &metal.RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
				Attention: &metal.GQAAttention{
					QProj: linear(w(p+".self_attn.q_proj.weight"), w(p+".self_attn.q_proj.scales"), w(p+".self_attn.q_proj.biases"), w(p+".self_attn.q_proj.bias"), p+".self_attn.q_proj"),
					KProj: linear(w(p+".self_attn.k_proj.weight"), w(p+".self_attn.k_proj.scales"), w(p+".self_attn.k_proj.biases"), w(p+".self_attn.k_proj.bias"), p+".self_attn.k_proj"),
					VProj: linear(w(p+".self_attn.v_proj.weight"), w(p+".self_attn.v_proj.scales"), w(p+".self_attn.v_proj.biases"), w(p+".self_attn.v_proj.bias"), p+".self_attn.v_proj"),
					OProj: linear(w(p+".self_attn.o_proj.weight"), w(p+".self_attn.o_proj.scales"), w(p+".self_attn.o_proj.biases"), w(p+".self_attn.o_proj.bias"), p+".self_attn.o_proj"),
					QNorm: &metal.RMSNormModule{Weight: w(p + ".self_attn.q_norm.weight")},
					KNorm: &metal.RMSNormModule{Weight: w(p + ".self_attn.k_norm.weight")},
				},
				MLP: nil,
			},
		}

		if isMoELayer[i] {
			block := &Qwen3MoEBlock{
				IntermediateSize: cfg.MoEIntermediateSize,
			}
			block.Router = qwen3MoELoadRouter(weights, int(i), q)
			block.SharedExpert = qwen3MoELoadSharedExpert(w, int(i))
			numExperts := int(cfg.NumExperts)
			if numExperts == 0 {
				numExperts = qwen3MoECountExperts(weights, int(i))
			}
			block.Experts = make([]*Qwen3MoEExpert, numExperts)
			for e := 0; e < numExperts; e++ {
				block.Experts[e] = qwen3MoELoadExpert(w, int(i), e)
			}
			block.SwitchExperts, _ = qwen3MoESwitchExperts(block.Experts)
			layer.MoE = block
		} else {
			layer.Dense.MLP = &metal.SiLUMLP{
				GateProj: linear(w(p+".mlp.gate_proj.weight"), w(p+".mlp.gate_proj.scales"), w(p+".mlp.gate_proj.biases"), w(p+".mlp.gate_proj.bias"), p+".mlp.gate_proj"),
				UpProj:   linear(w(p+".mlp.up_proj.weight"), w(p+".mlp.up_proj.scales"), w(p+".mlp.up_proj.biases"), w(p+".mlp.up_proj.bias"), p+".mlp.up_proj"),
				DownProj: linear(w(p+".mlp.down_proj.weight"), w(p+".mlp.down_proj.scales"), w(p+".mlp.down_proj.biases"), w(p+".mlp.down_proj.bias"), p+".mlp.down_proj"),
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
		"arch", detectedType, "layers", cfg.NumHiddenLayers, "hidden", cfg.HiddenSize,
		"heads", cfg.NumAttentionHeads, "kv_heads", cfg.NumKeyValueHeads,
		"head_dim", cfg.HeadDim, "vocab", cfg.VocabSize,
		"experts", cfg.NumExperts, "experts_per_tok", cfg.NumExpertsPerTok,
		"moe_intermediate", cfg.MoEIntermediateSize,
	)

	return m, nil
}

func qwen3MoELayerMask(cfg *metal.DenseConfig) []bool {
	mask := make([]bool, cfg.NumHiddenLayers)
	step := cfg.DecoderSparseStep
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		if step == 0 {
			mask[i] = i > 0
		} else {
			mask[i] = (i % step) == (step - 1)
		}
	}
	return mask
}

func qwen3MoELoadRouter(weights map[string]*metal.Array, layerIdx int, q *metal.QuantizationConfig) *metal.MoERouter {
	p := core.Sprintf("model.layers.%d.mlp", layerIdx)
	router := &metal.MoERouter{}
	for _, name := range []string{
		p + ".gate.weight",
		p + ".gate_proj.weight",
		p + ".router.weight",
		p + ".router.proj.weight",
	} {
		if w := metal.ResolveWeight(weights, name); w != nil {
			router.Weight = w
			router.Scales = metal.ResolveWeight(weights, core.TrimSuffix(name, ".weight")+".scales")
			router.Biases = metal.ResolveWeight(weights, core.TrimSuffix(name, ".weight")+".biases")
			if q != nil {
				router.GroupSize = q.GroupSize
				router.Bits = q.Bits
			}
			return router
		}
	}
	return router
}

func qwen3MoELoadSharedExpert(w func(string) *metal.Array, layerIdx int) *Qwen3MoESharedExpert {
	p := core.Sprintf("model.layers.%d.mlp.shared_expert", layerIdx)
	gateWeight := w(p + ".gate_proj.weight")
	if gateWeight == nil {
		gateWeight = w(core.Sprintf("model.layers.%d.mlp.shared_expert_gate_proj.weight", layerIdx))
	}
	if gateWeight == nil {
		return nil
	}
	return &Qwen3MoESharedExpert{
		GateProj: metal.NewLinear(gateWeight, w(p+".gate_proj.bias")),
		UpProj:   metal.NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
		DownProj: metal.NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
	}
}

func qwen3MoELoadExpert(w func(string) *metal.Array, layerIdx, expertIdx int) *Qwen3MoEExpert {
	p := core.Sprintf("model.layers.%d.mlp.experts.%d", layerIdx, expertIdx)
	return &Qwen3MoEExpert{
		GateProj: metal.NewLinear(w(p+".gate_proj.weight"), w(p+".gate_proj.bias")),
		UpProj:   metal.NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
		DownProj: metal.NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
	}
}

func qwen3MoECountExperts(weights map[string]*metal.Array, layerIdx int) int {
	prefix := core.Sprintf("model.layers.%d.mlp.experts.", layerIdx)
	count := 0
	for name := range weights {
		if core.HasPrefix(name, prefix) {
			count++
		}
	}
	if count > 0 {
		return count / 3
	}
	return int(32)
}

func qwen3MoESwitchExperts(experts []*Qwen3MoEExpert) (*metal.MoESwiGLUExperts, bool) {
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

func (m *Qwen3MoEModel) Forward(tokens *metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *Qwen3MoEModel) ForwardMasked(tokens *metal.Array, mask *metal.Array, caches []metal.Cache) *metal.Array {
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		hNext := qwen3MoEDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		metal.Free(h)
		h = hNext
	}

	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	metal.Free(h, normed)
	return out
}

func qwen3MoEDecoderLayerForward(l *Qwen3MoEDecoderLayer, x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, cfg *metal.DenseConfig) *metal.Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.Forward(normed, c, B, L, mask, cfg)
	metal.Free(normed)
	h := metal.Add(x, attnOut)
	metal.Free(attnOut)

	normed2 := l.Dense.PostAttnNorm.Forward(h, cfg.RMSNormEps)

	if l.isDenseLayer() && l.Dense.MLP != nil {
		mlpOut := l.Dense.MLP.Forward(normed2)
		metal.Free(normed2)
		result := metal.Add(h, mlpOut)
		metal.Free(h, mlpOut)
		return result
	}

	if mlpOut, ok := metal.MoESwiGLUForward(normed2, l.MoE.Router, int(cfg.NumExpertsPerTok), l.MoE.SwitchExperts); ok {
		metal.Free(normed2)
		result := metal.Add(h, mlpOut)
		metal.Free(h, mlpOut)
		return result
	}

	// Diagnostic fallback: keep the layer inspectable until every production
	// sparse path for this architecture is enabled.
	result := metal.Add(h, normed2)
	metal.Free(h, normed2)
	return result
}

func (m *Qwen3MoEModel) NewCache() []metal.Cache {
	caches := make([]metal.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = metal.NewKVCache()
	}
	return caches
}

func (m *Qwen3MoEModel) NumLayers() int { return len(m.Layers) }

func (m *Qwen3MoEModel) Tokenizer() *metal.Tokenizer { return m.Tok }

func (m *Qwen3MoEModel) ModelType() string { return m.modelType }

func (m *Qwen3MoEModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	cfg = metal.NormalizeLoRAConfig(cfg)
	adapter := &metal.LoRAAdapter{
		Layers: make(map[string]*metal.LoRALinear),
		Config: cfg,
		Model:  m,
	}
	for i, layer := range m.Layers {
		for _, target := range cfg.TargetKeys {
			var proj *metal.Linear
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
				if layer.isDenseLayer() && layer.Dense.MLP != nil {
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

func closeQwen3MoE(m *Qwen3MoEModel) {
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
			metal.FreeRMSNorm(layer.Dense.Attention.QNorm)
			metal.FreeRMSNorm(layer.Dense.Attention.KNorm)
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
			if layer.MoE.SharedExpert != nil {
				metal.FreeLinear(layer.MoE.SharedExpert.GateProj)
				metal.FreeLinear(layer.MoE.SharedExpert.UpProj)
				metal.FreeLinear(layer.MoE.SharedExpert.DownProj)
			}
			for _, expert := range layer.MoE.Experts {
				metal.FreeLinear(expert.GateProj)
				metal.FreeLinear(expert.UpProj)
				metal.FreeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}
