// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// foldRootSize multiplies a bf16 norm weight by RootSize = dModel^-0.5 (host), matching metal's
// cached Router.ScaleScaled = Scale·RootSize — the gemma4 MoE router norm MoERouterQuant expects
// pre-folded. nil passes through (an absent weight propagates the assembler's error).
func foldRootSize(w []byte, dModel int) []byte {
	if len(w) == 0 {
		return w
	}
	rootSize := float32(math.Pow(float64(dModel), -0.5))
	out := make([]byte, len(w))
	for i := 0; i+1 < len(w); i += 2 {
		h := f32ToBF16(bf16ToF32(w[i], w[i+1]) * rootSize)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// AssembleGemma4QuantLayers maps parsed 4-bit-quant safetensors tensors onto the native
// QuantizedLayerWeights per the gemma4 weight-name convention — the quant sibling of
// AssembleGemma4BF16's per-layer half. Each of the seven projections is an affine-quantised
// triple (.weight packed codes + .scales + .biases); the layer norms stay bf16 (norms aren't
// quantised — tiny vectors). groupSize/bits come from the checkpoint's quantization config
// (mlx: group_size, bits). Byte spans are validated against the Arch dims + groupSize:
// packed = out·in·bits/8, scales/biases = out·(in/groupSize) bf16 each. Returns the layers
// ready for NewQuantBackend.
//
// SCOPE: dense 4-bit decode layers only. The model-level embedding / LM head (themselves
// quantised in real gemma4-4bit, needing a dequantise-on-gather for the embedding and a tied
// quant matvec for the head) plus the quant session / directory loader are the follow-up
// slice; MoE layers (quantised experts) and the per-layer-input tower are later still.
func AssembleGemma4QuantLayers(tensors map[string]safetensors.Tensor, arch g4.Arch, quant *g4.QuantConfig) ([]QuantizedLayerWeights, error) {
	tensors = normalizeGemma4Names(tensors)
	if len(arch.Layer) == 0 || arch.Hidden <= 0 {
		return nil, core.NewError("native.AssembleGemma4QuantLayers: arch must have layers and hidden")
	}
	if quant == nil || quant.GroupSize <= 0 || quant.Bits <= 0 {
		return nil, core.NewError("native.AssembleGemma4QuantLayers: quant must have a default group_size/bits > 0")
	}
	dModel, dFF := arch.Hidden, arch.FF

	var ferr error
	// fetchNorm pulls a bf16 norm vector of an exact element count (norms aren't quantised).
	fetchNorm := func(name string, elems int, optional bool) []byte {
		if ferr != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			if !optional {
				ferr = core.NewError("native.AssembleGemma4QuantLayers: missing " + name)
			}
			return nil
		}
		if t.Dtype != "BF16" || len(t.Data) != elems*bf16Size {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + name + " must be BF16 of the arch element count")
			return nil
		}
		return t.Data
	}
	// fetchQuant pulls one affine-quant projection (.weight packed + .scales + .biases) of
	// logical shape [outDim, inDim], at the tensor's own (groupSize, bits) — quant.For(prefix),
	// so mixed-precision packs (gemma4 26B-A4B) get the right width per module. Byte spans
	// validated against the dims.
	fetchQuant := func(prefix string, outDim, inDim int) QuantWeight {
		if ferr != nil {
			return QuantWeight{}
		}
		groupSize, bits := quant.For(prefix)
		if groupSize <= 0 || inDim%groupSize != 0 {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + " inDim not a multiple of its groupSize")
			return QuantWeight{}
		}
		packed, ok1 := tensors[prefix+".weight"]
		scales, ok2 := tensors[prefix+".scales"]
		biases, ok3 := tensors[prefix+".biases"]
		if !ok1 || !ok2 || !ok3 {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + " missing .weight/.scales/.biases")
			return QuantWeight{}
		}
		wantPacked := outDim * inDim * bits / 8
		wantScale := outDim * (inDim / groupSize) * bf16Size
		if len(packed.Data) != wantPacked {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + ".weight packed byte span != out·in·bits/8")
			return QuantWeight{}
		}
		if scales.Dtype != "BF16" || len(scales.Data) != wantScale {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + ".scales must be BF16 of out·(in/groupSize)")
			return QuantWeight{}
		}
		if biases.Dtype != "BF16" || len(biases.Data) != wantScale {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + ".biases must be BF16 of out·(in/groupSize)")
			return QuantWeight{}
		}
		return QuantWeight{Packed: packed.Data, Scales: scales.Data, Biases: biases.Data}
	}

	layers := make([]QuantizedLayerWeights, len(arch.Layer))
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		l := &layers[i]
		// per-attention-type geometry: gemma4 full_attention layers use global_head_dim
		// (a larger head) and may differ in KV heads, so q/k/v/o spans are per layer.
		headDim, kvHeads := headDimOf(arch.Layer[i], arch.HeadDim), kvHeadsOf(arch.Layer[i], arch.KVHeads)
		qDim, kvDim := arch.Heads*headDim, kvHeads*headDim
		l.AttnNormW = fetchNorm(p+".input_layernorm.weight", dModel, false)
		l.Q = fetchQuant(p+".self_attn.q_proj", qDim, dModel)
		l.K = fetchQuant(p+".self_attn.k_proj", kvDim, dModel)
		if !arch.AttentionKEqV { // gemma4 K==V (12B/31B): no v_proj — V rides the k-proj output, value-normed
			l.V = fetchQuant(p+".self_attn.v_proj", kvDim, dModel)
		}
		l.O = fetchQuant(p+".self_attn.o_proj", dModel, qDim)
		l.GroupSize, l.Bits = quant.GroupSize, quant.Bits // attention uses the default width
		// gemma4 norms (bf16, optional/nil-default — applied by the decode when present).
		l.QNormW = fetchNorm(p+".self_attn.q_norm.weight", headDim, true)
		l.KNormW = fetchNorm(p+".self_attn.k_norm.weight", headDim, true)
		l.PostAttnNormW = fetchNorm(p+".post_attention_layernorm.weight", dModel, true)
		l.LayerScalarW = fetchNorm(p+".layer_scalar", 1, true) // gemma4 per-layer output scalar [1] bf16
		// FFN: the dense MLP, or the 4-bit dual-branch MoE block (gemma4 26B-A4B — mixed precision).
		if arch.Layer[i].MoE {
			egs, ebits := quant.For(p + ".experts.switch_glu.gate_proj")
			lgs, lbits := quant.For(p + ".mlp.gate_proj")
			rgs, rbits := quant.For(p + ".router.proj")
			l.MoE = &MoEQuantLayerWeights{
				NumExperts: arch.Experts, TopK: arch.TopK, ExpertDFF: arch.ExpertFF,
				ExpertGroupSize: egs, ExpertBits: ebits, LocalGroupSize: lgs, LocalBits: lbits, RouterGroupSize: rgs, RouterBits: rbits,
				PreFFNormW:        fetchNorm(p+".pre_feedforward_layernorm.weight", dModel, false),
				PreFFNorm2W:       fetchNorm(p+".pre_feedforward_layernorm_2.weight", dModel, false),
				PostFFNorm1W:      fetchNorm(p+".post_feedforward_layernorm_1.weight", dModel, false),
				PostFFNorm2W:      fetchNorm(p+".post_feedforward_layernorm_2.weight", dModel, false),
				PostFFNormW:       fetchNorm(p+".post_feedforward_layernorm.weight", dModel, false),
				LocalGate:         fetchQuant(p+".mlp.gate_proj", dFF, dModel),
				LocalUp:           fetchQuant(p+".mlp.up_proj", dFF, dModel),
				LocalDown:         fetchQuant(p+".mlp.down_proj", dModel, dFF),
				RouterNormWScaled: foldRootSize(fetchNorm(p+".router.scale", dModel, false), dModel),
				Router:            fetchQuant(p+".router.proj", arch.Experts, dModel),
				PerExpertScale:    fetchNorm(p+".router.per_expert_scale", arch.Experts, true),
				ExpGate:           fetchQuant(p+".experts.switch_glu.gate_proj", arch.Experts*arch.ExpertFF, dModel),
				ExpUp:             fetchQuant(p+".experts.switch_glu.up_proj", arch.Experts*arch.ExpertFF, dModel),
				ExpDown:           fetchQuant(p+".experts.switch_glu.down_proj", arch.Experts*dModel, arch.ExpertFF),
			}
		} else {
			l.MLPNormW = fetchNorm(p+".pre_feedforward_layernorm.weight", dModel, false)
			l.Gate = fetchQuant(p+".mlp.gate_proj", dFF, dModel)
			l.Up = fetchQuant(p+".mlp.up_proj", dFF, dModel)
			l.Down = fetchQuant(p+".mlp.down_proj", dModel, dFF)
			l.PostFFNormW = fetchNorm(p+".post_feedforward_layernorm.weight", dModel, true)
		}
		// per-layer-input gate (gemma4 E2B/E4B; absent on dense models). The gate + projection
		// are 4-bit, the post-norm bf16. Gated on presence so a dense pack stays PLE-free.
		if pliDim := arch.PerLayerInputHidden; pliDim > 0 {
			if _, ok := tensors[p+".per_layer_input_gate.weight"]; ok {
				l.PerLayerGate = fetchQuant(p+".per_layer_input_gate", pliDim, dModel)
				l.PerLayerProjection = fetchQuant(p+".per_layer_projection", dModel, pliDim)
				l.PostPerLayerInputNormW = fetchNorm(p+".post_per_layer_input_norm.weight", dModel, false)
			}
		}
	}
	if ferr != nil {
		return nil, ferr
	}
	return layers, nil
}

// Gemma4Quant is a 4-bit gemma4 model mapped onto the native structs: the quantised decode
// layers plus the model-level tensors. In a 4-bit checkpoint the embedding is itself quantised
// (mlx quantises nn.Embedding) and gemma ties the LM head to it, so Embed/EmbedScales/
// EmbedBiases are the affine triple and LMHead* alias them when tied (the usual gemma4 case).
type Gemma4Quant struct {
	Layers                             []QuantizedLayerWeights
	Embed, EmbedScales, EmbedBiases    []byte // quantised [vocab × dModel] input embedding
	FinalNorm                          []byte // bf16 [dModel] (model.norm.weight)
	LMHead, LMHeadScales, LMHeadBiases []byte // tied embedding, or a separate quant head
	Tied                               bool
	GroupSize, Bits                    int
	// per-layer-input tower (gemma4 E2B/E4B; nil for models without it). The per-layer
	// embedding is 4-bit, the model projection + norm bf16 — fed to PerLayerInputs each token.
	EmbedPerLayer, EmbedPerLayerScales, EmbedPerLayerBiases []byte // quant [vocabPLI × numLayers·pliDim]
	PerLayerModelProjW                                      []byte // bf16 [numLayers·pliDim × dModel]
	PerLayerProjNormW                                       []byte // bf16 [pliDim]
}

// HasPLE reports whether this model carries the gemma4 per-layer-input tower.
func (g *Gemma4Quant) HasPLE() bool { return len(g.EmbedPerLayer) > 0 }

// AssembleGemma4Quant maps parsed 4-bit-quant safetensors tensors onto a full Gemma4Quant:
// the decode layers (AssembleGemma4QuantLayers) plus the quantised input embedding, the bf16
// final norm, and the LM head — tied to the embedding when lm_head.weight is absent (gemma4
// ties). The embedding/head triples are validated against the Arch dims + groupSize.
func AssembleGemma4Quant(tensors map[string]safetensors.Tensor, arch g4.Arch, quant *g4.QuantConfig) (*Gemma4Quant, error) {
	tensors = normalizeGemma4Names(tensors) // its own embed/norm fetches read tensors too
	layers, err := AssembleGemma4QuantLayers(tensors, arch, quant)
	if err != nil {
		return nil, err
	}
	if arch.Vocab <= 0 {
		return nil, core.NewError("native.AssembleGemma4Quant: arch must have vocab")
	}
	if quant == nil || quant.GroupSize <= 0 {
		return nil, core.NewError("native.AssembleGemma4Quant: quant must have a default group_size > 0")
	}
	dModel, vocab := arch.Hidden, arch.Vocab

	var ferr error
	fetchNorm := func(name string, elems int) []byte {
		if ferr != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			ferr = core.NewError("native.AssembleGemma4Quant: missing " + name)
			return nil
		}
		if t.Dtype != "BF16" || len(t.Data) != elems*bf16Size {
			ferr = core.NewError("native.AssembleGemma4Quant: " + name + " must be BF16 of the arch element count")
			return nil
		}
		return t.Data
	}
	// fetchQuantTriple pulls a quant [outDim × inDim] projection's .weight/.scales/.biases.
	fetchQuantTriple := func(prefix string, outDim, inDim int) (packed, scales, biases []byte) {
		if ferr != nil {
			return
		}
		groupSize, bits := quant.For(prefix)
		if groupSize <= 0 || inDim%groupSize != 0 {
			ferr = core.NewError("native.AssembleGemma4Quant: " + prefix + " inDim not a multiple of its groupSize")
			return
		}
		p, ok1 := tensors[prefix+".weight"]
		s, ok2 := tensors[prefix+".scales"]
		b, ok3 := tensors[prefix+".biases"]
		if !ok1 || !ok2 || !ok3 {
			ferr = core.NewError("native.AssembleGemma4Quant: " + prefix + " missing .weight/.scales/.biases")
			return
		}
		wantPacked := outDim * inDim * bits / 8
		wantSB := outDim * (inDim / groupSize) * bf16Size
		if len(p.Data) != wantPacked {
			ferr = core.NewError("native.AssembleGemma4Quant: " + prefix + ".weight packed byte span mismatch")
			return
		}
		if s.Dtype != "BF16" || len(s.Data) != wantSB || b.Dtype != "BF16" || len(b.Data) != wantSB {
			ferr = core.NewError("native.AssembleGemma4Quant: " + prefix + " scales/biases must be BF16 of out·(in/groupSize)")
			return
		}
		return p.Data, s.Data, b.Data
	}

	g := &Gemma4Quant{Layers: layers, GroupSize: quant.GroupSize, Bits: quant.Bits}
	g.Embed, g.EmbedScales, g.EmbedBiases = fetchQuantTriple("model.embed_tokens", vocab, dModel)
	g.FinalNorm = fetchNorm("model.norm.weight", dModel)
	if _, ok := tensors["lm_head.weight"]; ok {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases = fetchQuantTriple("lm_head", vocab, dModel)
	} else {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases, g.Tied = g.Embed, g.EmbedScales, g.EmbedBiases, true
	}
	// per-layer-input tower (gemma4 E2B/E4B; absent on dense models). The per-layer embedding
	// is 4-bit, the model projection + norm bf16 — fed to PerLayerInputs each token.
	if pliDim := arch.PerLayerInputHidden; pliDim > 0 {
		if _, ok := tensors["model.embed_tokens_per_layer.weight"]; ok {
			plDim := len(arch.Layer) * pliDim
			g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases = fetchQuantTriple("model.embed_tokens_per_layer", arch.PerLayerInputVocab, plDim)
			g.PerLayerModelProjW = fetchNorm("model.per_layer_model_projection.weight", plDim*dModel)
			g.PerLayerProjNormW = fetchNorm("model.per_layer_projection_norm.weight", pliDim)
		}
	}
	if ferr != nil {
		return nil, ferr
	}
	return g, nil
}
