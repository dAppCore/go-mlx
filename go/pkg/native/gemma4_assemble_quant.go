// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
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

// Gemma4Quant is a quantised gemma4 model mapped onto the native structs: the quantised decode
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
	PerLayerModelProjW                                      []byte // [numLayers·pliDim × dModel]: bf16 (regular packs, e2b) OR packed 4-bit (qat packs, e4b)
	PerLayerModelProjScales, PerLayerModelProjBiases        []byte // affine scales/biases when the projection is quantised (e4b); nil ⇒ PerLayerModelProjW is bf16
	PerLayerModelProjGS, PerLayerModelProjBits              int    // affine geometry for the quantised projection
	PerLayerProjNormW                                       []byte // bf16 [pliDim]
}

// HasPLE reports whether this model carries the gemma4 per-layer-input tower.
func (g *Gemma4Quant) HasPLE() bool { return len(g.EmbedPerLayer) > 0 }

// AssembleGemma4Quant maps a quantised gemma4 checkpoint onto the native Gemma4Quant through the
// shared loader: pkg/model/gemma4.Assemble makes the per-weight quant decision (bf16-vs-quant,
// group size, bit-width) from the tensor shapes, and loadedToQuant maps the result onto the native
// structs. The hand-coded per-weight fetchQuant/fetchNorm assembler it replaced is gone — a model
// in a different quant (4/5/6/8-bit, mixed precision) needs no native change.
func AssembleGemma4Quant(tensors map[string]safetensors.Tensor, arch model.Arch, quant *g4.QuantConfig) (*Gemma4Quant, error) {
	if quant == nil || quant.GroupSize <= 0 {
		return nil, core.NewError("native.AssembleGemma4Quant: quant must have a default group_size > 0")
	}
	m, err := g4.Assemble(tensors, arch)
	if err != nil {
		return nil, err
	}
	return loadedToQuant(m, quant.GroupSize, quant.Bits)
}
