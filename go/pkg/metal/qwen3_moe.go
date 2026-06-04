// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

	coreio "dappco.re/go/io"
)

type Qwen3MoEModel struct {
	EmbedTokens *Embedding
	Layers      []*Qwen3MoEDecoderLayer
	Norm        *RMSNormModule
	Output      *Linear
	Tok         *Tokenizer
	Cfg         *DenseConfig
	modelType   string
}

type Qwen3MoESharedExpert struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

type Qwen3MoEExpert struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

type Qwen3MoEBlock struct {
	Router           *MoERouter
	SharedExpert     *Qwen3MoESharedExpert
	Experts          []*Qwen3MoEExpert
	SwitchExperts    *MoESwiGLUExperts
	IntermediateSize int32
}

type Qwen3MoEDecoderLayer struct {
	Dense *DenseDecoderLayer
	MoE   *Qwen3MoEBlock
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
	if m == nil || len(m.Layers) == 0 {
		return false
	}
	for _, layer := range m.Layers {
		if layer == nil {
			return false
		}
		var router *MoERouter
		var switchExperts *MoESwiGLUExperts
		if layer.MoE != nil {
			router = layer.MoE.Router
			switchExperts = layer.MoE.SwitchExperts
		}
		if !moeDenseLayerTextReady(layer.Dense, layer.isMoELayer(), router, switchExperts) {
			return false
		}
	}
	return true
}

// MoETextDecodeFamily returns the canonical family token used in unavailable
// diagnostics (metal.MoETextRuntimeReporter).
func (m *Qwen3MoEModel) MoETextDecodeFamily() string { return "qwen3_moe" }

func LoadQwen3MoE(modelPath string) (*Qwen3MoEModel, error) {
	root := ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load config", err)
	}
	data := []byte(str)

	cfg, err := ParseDenseConfig(data)
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "parse config", err)
	}
	if cfg.IsQwen36Hybrid() {
		return nil, core.E("qwen3_moe.Load", "qwen3_6_moe hybrid linear attention is not supported here; use the staged loader", nil)
	}
	if !cfg.IsMoE() {
		return nil, core.E("qwen3_moe.Load", "config must have MoE metadata (num_experts, num_experts_per_tok, moe_intermediate_size)", nil)
	}

	tok, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load tokenizer", err)
	}

	weights, err := LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("qwen3_moe.Load", "load weights", err)
	}

	w := func(name string) *Array { return ResolveWeight(weights, name) }

	q := cfg.Quantization
	if q != nil {
		core.Info("qwen3_moe: using quantized inference", "bits", q.Bits, "group_size", q.GroupSize)
	}
	linear := func(weight, scales, biases, bias *Array, prefix string) *Linear {
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

	detectedType := DetectDenseModelType(data, weights)

	m := &Qwen3MoEModel{
		EmbedTokens: embed,
		Layers:      make([]*Qwen3MoEDecoderLayer, cfg.NumHiddenLayers),
		Norm:        &RMSNormModule{Weight: w("model.norm.weight")},
		Tok:         tok,
		Cfg:         cfg,
		modelType:   detectedType,
	}

	isMoELayer := qwen3MoELayerMask(cfg)

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		layer := &Qwen3MoEDecoderLayer{
			Dense: &DenseDecoderLayer{
				InputNorm:    &RMSNormModule{Weight: w(p + ".input_layernorm.weight")},
				PostAttnNorm: &RMSNormModule{Weight: w(p + ".post_attention_layernorm.weight")},
				Attention: &GQAAttention{
					QProj: linear(w(p+".self_attn.q_proj.weight"), w(p+".self_attn.q_proj.scales"), w(p+".self_attn.q_proj.biases"), w(p+".self_attn.q_proj.bias"), p+".self_attn.q_proj"),
					KProj: linear(w(p+".self_attn.k_proj.weight"), w(p+".self_attn.k_proj.scales"), w(p+".self_attn.k_proj.biases"), w(p+".self_attn.k_proj.bias"), p+".self_attn.k_proj"),
					VProj: linear(w(p+".self_attn.v_proj.weight"), w(p+".self_attn.v_proj.scales"), w(p+".self_attn.v_proj.biases"), w(p+".self_attn.v_proj.bias"), p+".self_attn.v_proj"),
					OProj: linear(w(p+".self_attn.o_proj.weight"), w(p+".self_attn.o_proj.scales"), w(p+".self_attn.o_proj.biases"), w(p+".self_attn.o_proj.bias"), p+".self_attn.o_proj"),
					QNorm: &RMSNormModule{Weight: w(p + ".self_attn.q_norm.weight")},
					KNorm: &RMSNormModule{Weight: w(p + ".self_attn.k_norm.weight")},
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
			layer.Dense.MLP = &SiLUMLP{
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
		"experts", cfg.NumExperts, "experts_per_tok", cfg.NumExpertsPerTok,
		"moe_intermediate", cfg.MoEIntermediateSize,
	)

	return m, nil
}

func qwen3MoELayerMask(cfg *DenseConfig) []bool {
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

func qwen3MoELoadRouter(weights map[string]*Array, layerIdx int, q *QuantizationConfig) *MoERouter {
	p := core.Sprintf("model.layers.%d.mlp", layerIdx)
	router := &MoERouter{}
	for _, name := range []string{
		p + ".gate.weight",
		p + ".gate_proj.weight",
		p + ".router.weight",
		p + ".router.proj.weight",
	} {
		if w := ResolveWeight(weights, name); w != nil {
			router.Weight = w
			router.Scales = ResolveWeight(weights, core.TrimSuffix(name, ".weight")+".scales")
			router.Biases = ResolveWeight(weights, core.TrimSuffix(name, ".weight")+".biases")
			if q != nil {
				router.GroupSize = q.GroupSize
				router.Bits = q.Bits
			}
			return router
		}
	}
	return router
}

func qwen3MoELoadSharedExpert(w func(string) *Array, layerIdx int) *Qwen3MoESharedExpert {
	p := core.Sprintf("model.layers.%d.mlp.shared_expert", layerIdx)
	gateWeight := w(p + ".gate_proj.weight")
	if gateWeight == nil {
		gateWeight = w(core.Sprintf("model.layers.%d.mlp.shared_expert_gate_proj.weight", layerIdx))
	}
	if gateWeight == nil {
		return nil
	}
	return &Qwen3MoESharedExpert{
		GateProj: NewLinear(gateWeight, w(p+".gate_proj.bias")),
		UpProj:   NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
		DownProj: NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
	}
}

func qwen3MoELoadExpert(w func(string) *Array, layerIdx, expertIdx int) *Qwen3MoEExpert {
	p := core.Sprintf("model.layers.%d.mlp.experts.%d", layerIdx, expertIdx)
	return &Qwen3MoEExpert{
		GateProj: NewLinear(w(p+".gate_proj.weight"), w(p+".gate_proj.bias")),
		UpProj:   NewLinear(w(p+".up_proj.weight"), w(p+".up_proj.bias")),
		DownProj: NewLinear(w(p+".down_proj.weight"), w(p+".down_proj.bias")),
	}
}

func qwen3MoECountExperts(weights map[string]*Array, layerIdx int) int {
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

func qwen3MoESwitchExperts(experts []*Qwen3MoEExpert) (*MoESwiGLUExperts, bool) {
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

func (m *Qwen3MoEModel) Forward(tokens *Array, caches []Cache) *Array {
	return m.ForwardMasked(tokens, nil, caches)
}

func (m *Qwen3MoEModel) ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array {
	var shapeBuf [MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	B, L := shape[0], shape[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		hNext := qwen3MoEDecoderLayerForward(layer, h, caches[i], B, L, mask, m.Cfg)
		Free(h)
		h = hNext
	}

	normed := m.Norm.Forward(h, m.Cfg.RMSNormEps)
	out := m.Output.Forward(normed)
	Free(h, normed)
	return out
}

func qwen3MoEDecoderLayerForward(l *Qwen3MoEDecoderLayer, x *Array, c Cache, B, L int32, mask *Array, cfg *DenseConfig) *Array {
	normed := l.Dense.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Dense.Attention.forward(normed, c, B, L, mask, cfg)
	Free(normed)
	h := Add(x, attnOut)
	Free(attnOut)

	normed2 := l.Dense.PostAttnNorm.Forward(h, cfg.RMSNormEps)

	if l.isDenseLayer() && l.Dense.MLP != nil {
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

func (m *Qwen3MoEModel) NewCache() []Cache {
	caches := make([]Cache, len(m.Layers))
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}

func (m *Qwen3MoEModel) NumLayers() int { return len(m.Layers) }

func (m *Qwen3MoEModel) Tokenizer() *Tokenizer { return m.Tok }

func (m *Qwen3MoEModel) ModelType() string { return m.modelType }

func (m *Qwen3MoEModel) ApplyLoRA(cfg LoRAConfig) *LoRAAdapter {
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
				lora := NewLoRALinear(proj, cfg.Rank, cfg.Alpha, cfg.DType)
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
			FreeRMSNorm(layer.Dense.Attention.QNorm)
			FreeRMSNorm(layer.Dense.Attention.KNorm)
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
			if layer.MoE.SharedExpert != nil {
				FreeLinear(layer.MoE.SharedExpert.GateProj)
				FreeLinear(layer.MoE.SharedExpert.UpProj)
				FreeLinear(layer.MoE.SharedExpert.DownProj)
			}
			for _, expert := range layer.MoE.Experts {
				FreeLinear(expert.GateProj)
				FreeLinear(expert.UpProj)
				FreeLinear(expert.DownProj)
			}
		}
	}
	m.Layers = nil
}
