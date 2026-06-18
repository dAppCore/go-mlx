// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// load_shared.go consumes the backend-agnostic gemma4.LoadedModel (pkg/model — where the
// per-weight quant decision is made ONCE, quant-agnostically, by reading the tensor shapes) and
// maps it onto the native decode structs. The hand-coded per-weight fetchQuant/fetchNorm walk that
// used to live in the native assembler is gone: this is a mechanical translation, not a second
// loader. A weight that one quant leaves bf16 while another quantises (e4b's per_layer_model_
// projection) is handled by the shared loader's .scales decision, so native never re-bugs it.

// loadedToQuant maps a LoadedModel onto the native 4-bit Gemma4Quant. The model-wide gs/bits are
// the native structs' single quant geometry (gemma4 quant packs are uniform across the projections;
// the per-weight geometry the shared loader read from shapes agrees with it). MoE (26B-A4B) is not
// yet routed here — it errors clearly rather than mis-assembling.
func loadedToQuant(m *g4.LoadedModel, gs, bits int) (*Gemma4Quant, error) {
	if m == nil || m.Embed == nil {
		return nil, core.NewError("native.loadedToQuant: nil model or embedding")
	}
	g := &Gemma4Quant{GroupSize: gs, Bits: bits, FinalNorm: m.FinalNorm}
	g.Embed, g.EmbedScales, g.EmbedBiases = m.Embed.Weight, m.Embed.Scales, m.Embed.Biases
	if m.LMHead != nil {
		g.LMHead, g.LMHeadScales, g.LMHeadBiases = m.LMHead.Weight, m.LMHead.Scales, m.LMHead.Biases
	} else { // tied: the head reuses the embedding triple
		g.LMHead, g.LMHeadScales, g.LMHeadBiases, g.Tied = m.Embed.Weight, m.Embed.Scales, m.Embed.Biases, true
	}
	if m.EmbedPerLayer != nil { // PLE tower (E2B/E4B)
		g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases = m.EmbedPerLayer.Weight, m.EmbedPerLayer.Scales, m.EmbedPerLayer.Biases
		g.PerLayerProjNormW = m.PerLayerProjNorm
	}
	if p := m.PerLayerModelProj; p != nil {
		// PerLayerModelProjW holds the packed weight (qat: e4b) or the bf16 weight (regular: e2b);
		// the scales (set only when quantised) tell PerLayerInputs which matvec to run.
		g.PerLayerModelProjW = p.Weight
		if p.Quantised() {
			g.PerLayerModelProjScales, g.PerLayerModelProjBiases = p.Scales, p.Biases
			g.PerLayerModelProjGS, g.PerLayerModelProjBits = p.GroupSize, p.Bits
		}
	}
	g.Layers = make([]QuantizedLayerWeights, len(m.Layers))
	for i := range m.Layers {
		L := &m.Layers[i]
		ql := &g.Layers[i]
		ql.AttnNormW, ql.PostAttnNormW = L.AttnNorm, L.PostAttnNorm
		ql.QNormW, ql.KNormW, ql.LayerScalarW = L.QNorm, L.KNorm, L.LayerScalar
		ql.GroupSize, ql.Bits = gs, bits
		ql.Q, ql.K, ql.V, ql.O = qw(L.Q), qw(L.K), qw(L.V), qw(L.O)
		ql.PerLayerGate, ql.PerLayerProjection = qw(L.PerLayerGate), qw(L.PerLayerProjection)
		ql.PostPerLayerInputNormW = L.PostPerLayerInputNorm
		if L.MoE != nil {
			return nil, core.NewError("native.loadedToQuant: MoE (26B-A4B) not yet routed through the shared loader")
		}
		ql.MLPNormW, ql.PostFFNormW = L.MLPNorm, L.PostFFNorm
		ql.Gate, ql.Up, ql.Down = qw(L.Gate), qw(L.Up), qw(L.Down)
		if L.Gate != nil { // per-layer MatFormer FFN width, read from the gate's output rows
			ql.DFF = L.Gate.OutDim
		}
	}
	return g, nil
}

// qw maps a shared model.Linear to the native quant-weight triple (packed codes + bf16 scales +
// biases). A nil Linear (an absent optional weight — a K==V layer's v_proj, a KV-shared layer's
// k_proj) yields the zero QuantWeight, which the projector treats as "skip".
func qw(lin *model.Linear) QuantWeight {
	if lin == nil {
		return QuantWeight{}
	}
	// GroupSize/Bits are the weight's OWN geometry (read from shapes by the shared loader) — this is
	// what carries e4b-qat's per-layer mixed precision (the 8-bit MLP beside the 4-bit attention)
	// through to the qmv kernel, instead of a single model-wide width.
	return QuantWeight{Packed: lin.Weight, Scales: lin.Scales, Biases: lin.Biases, GroupSize: lin.GroupSize, Bits: lin.Bits}
}
