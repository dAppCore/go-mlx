// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

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
func AssembleGemma4QuantLayers(tensors map[string]safetensors.Tensor, arch g4.Arch, groupSize, bits int) ([]QuantizedLayerWeights, error) {
	tensors = normalizeGemma4Names(tensors)
	if arch.HasMoE() {
		return nil, core.NewError("native.AssembleGemma4QuantLayers: MoE arch not supported yet (dense 4-bit only)")
	}
	if len(arch.Layer) == 0 || arch.Hidden <= 0 {
		return nil, core.NewError("native.AssembleGemma4QuantLayers: arch must have layers and hidden")
	}
	if groupSize <= 0 || bits <= 0 {
		return nil, core.NewError("native.AssembleGemma4QuantLayers: groupSize and bits must be > 0")
	}
	dModel, headDim, dFF := arch.Hidden, arch.HeadDim, arch.FF
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim

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
	// logical shape [outDim, inDim], validating each byte span against the dims + groupSize.
	fetchQuant := func(prefix string, outDim, inDim int) QuantWeight {
		if ferr != nil {
			return QuantWeight{}
		}
		if inDim%groupSize != 0 {
			ferr = core.NewError("native.AssembleGemma4QuantLayers: " + prefix + " inDim not a multiple of groupSize")
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
		l.AttnNormW = fetchNorm(p+".input_layernorm.weight", dModel, false)
		l.MLPNormW = fetchNorm(p+".pre_feedforward_layernorm.weight", dModel, false)
		l.Q = fetchQuant(p+".self_attn.q_proj", qDim, dModel)
		l.K = fetchQuant(p+".self_attn.k_proj", kvDim, dModel)
		l.V = fetchQuant(p+".self_attn.v_proj", kvDim, dModel)
		l.O = fetchQuant(p+".self_attn.o_proj", dModel, qDim)
		l.Gate = fetchQuant(p+".mlp.gate_proj", dFF, dModel)
		l.Up = fetchQuant(p+".mlp.up_proj", dFF, dModel)
		l.Down = fetchQuant(p+".mlp.down_proj", dModel, dFF)
		l.GroupSize, l.Bits = groupSize, bits
		// gemma4 norms (bf16, optional/nil-default — applied by the decode when present).
		l.QNormW = fetchNorm(p+".self_attn.q_norm.weight", headDim, true)
		l.KNormW = fetchNorm(p+".self_attn.k_norm.weight", headDim, true)
		l.PostAttnNormW = fetchNorm(p+".post_attention_layernorm.weight", dModel, true)
		l.PostFFNormW = fetchNorm(p+".post_feedforward_layernorm.weight", dModel, true)
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
}

// AssembleGemma4Quant maps parsed 4-bit-quant safetensors tensors onto a full Gemma4Quant:
// the decode layers (AssembleGemma4QuantLayers) plus the quantised input embedding, the bf16
// final norm, and the LM head — tied to the embedding when lm_head.weight is absent (gemma4
// ties). The embedding/head triples are validated against the Arch dims + groupSize.
func AssembleGemma4Quant(tensors map[string]safetensors.Tensor, arch g4.Arch, groupSize, bits int) (*Gemma4Quant, error) {
	tensors = normalizeGemma4Names(tensors) // its own embed/norm fetches read tensors too
	layers, err := AssembleGemma4QuantLayers(tensors, arch, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if arch.Vocab <= 0 {
		return nil, core.NewError("native.AssembleGemma4Quant: arch must have vocab")
	}
	dModel, vocab := arch.Hidden, arch.Vocab
	if dModel%groupSize != 0 {
		return nil, core.NewError("native.AssembleGemma4Quant: groupSize must divide hidden")
	}

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

	g := &Gemma4Quant{Layers: layers, GroupSize: groupSize, Bits: bits}
	g.Embed, g.EmbedScales, g.EmbedBiases = fetchQuantTriple("model.embed_tokens", vocab, dModel)
	g.FinalNorm = fetchNorm("model.norm.weight", dModel)
	if _, ok := tensors["lm_head.weight"]; ok {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases = fetchQuantTriple("lm_head", vocab, dModel)
	} else {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases, g.Tied = g.Embed, g.EmbedScales, g.EmbedBiases, true
	}
	if ferr != nil {
		return nil, ferr
	}
	return g, nil
}
