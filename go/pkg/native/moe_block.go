// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// MoELayerWeights holds the bf16 weights AND the MoE-specific shape of one gemma4 MoE
// feed-forward block: the five independent RMSNorm weights, the local dense MLP, the
// router, and the experts. Norm weights are dModel bf16. RouterNormWScaled is the
// router's own norm weight ALREADY scaled by RootSize (folded at load like metal's
// cached ScaleScaled — see MoERouter). PerExpertScale is optional (nil to skip). The
// local MLP runs at the model-wide dFF; the experts run at ExpertDFF (gemma4 gives
// them a distinct MoEIntermediateSize). The MoE-specific dims (NumExperts/TopK/
// ExpertDFF) live here so a MoE layer is self-describing — model-wide dModel/dFF/eps
// stay executor parameters shared by dense and MoE layers alike.
type MoELayerWeights struct {
	NumExperts, TopK, ExpertDFF int // MoE shape (model-wide dModel/dFF/eps are args)

	PreFFNormW   []byte // local MLP input norm
	PreFFNorm2W  []byte // expert-branch input norm
	PostFFNorm1W []byte // post local-MLP norm
	PostFFNorm2W []byte // post-expert norm
	PostFFNormW  []byte // final combined-branch norm

	WGate, WUp, WDown []byte // local dense MLP (dFF)

	RouterNormWScaled []byte // router internal norm (pre-scaled by RootSize)
	RouterW           []byte // [NumExperts × dModel] expert-score projection
	PerExpertScale    []byte // [NumExperts] optional (nil to skip)

	ExpGateW, ExpUpW, ExpDownW []byte // experts ([NumExperts × …] at ExpertDFF)
}

// mlpTransformBF16 is the gemma SwiGLU MLP transform on an ALREADY-normed input:
// WDown·(gelu(WGate·x)·(WUp·x)) — no input norm, no residual (the MoE block applies
// those around it). Structurally one expert's computation; composed from the
// parity-proven bf16 ops.
func mlpTransformBF16(x, wGate, wUp, wDown []byte, dModel, dFF int) ([]byte, error) {
	gate, err := MatVecBF16(wGate, x, dFF, dModel)
	if err != nil {
		return nil, err
	}
	up, err := MatVecBF16(wUp, x, dFF, dModel)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		return nil, err
	}
	return MatVecBF16(wDown, gated, dModel, dFF)
}

// MoEBlockBF16 runs the dual-branch feed-forward of a gemma4 MoE layer on the
// post-attention residual h and returns h + ffResidual. BOTH branches run: the local
// dense MLP on rms(h, PreFFNorm), and the expert branch (router → topK experts) on
// rms(h, PreFFNorm2). Each branch output is independently normed (PostFFNorm1 /
// PostFFNorm2), summed, post-normed (PostFFNorm), then added back to the residual
// once. Mirrors pkg/metal/model/gemma4 decoder_layer.go's MoE branch op-for-op.
//
// The router operates on the RAW residual h (it applies its own internal norm); the
// experts operate on the separately-normed h2In. The router runs host top-k (see
// MoERouter) so this block is not a single command buffer; everything else is the
// parity-proven bf16 ops composed. Byte-for-byte against an independent reference
// that rebuilds both branches from primitives (TestMoEBlock). The per-layer-input
// gate, the LayerScalar, and the FFN-memory augmenter are out of scope (later
// slices / nil for standard gemma4) — this block ends at residual + ffResidual.
// NumExperts/TopK/ExpertDFF come from w; dModel/dFF/eps are the model-wide args.
func MoEBlockBF16(h []byte, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockBF16: h must be dModel bf16 bytes")
	}
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF

	// router decision on the raw residual (the router applies its own norm).
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		return nil, err
	}

	// local dense MLP branch: transform on rms(h, PreFFNorm), no residual.
	h1In, err := RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h1, err := mlpTransformBF16(h1In, w.WGate, w.WUp, w.WDown, dModel, dFF)
	if err != nil {
		return nil, err
	}

	// expert branch: topK experts on rms(h, PreFFNorm2), router-weighted.
	h2In, err := RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h2, err := MoEExperts(h2In, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, expertDFF)
	if err != nil {
		return nil, err
	}

	// each branch independently normed, summed, post-normed, residual add (once).
	h1Normed, err := RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h2Normed, err := RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	combined, err := AddBF16(h1Normed, h2Normed)
	if err != nil {
		return nil, err
	}
	ffResidual, err := RMSNormBF16(combined, w.PostFFNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(h, ffResidual)
}

// MoEQuantLayerWeights is MoELayerWeights for a 4-bit MoE layer (gemma4 26B-A4B): the local
// dense MLP, the router score projection, and the batched SwitchGLU experts are all affine-
// quantised; the five norms stay bf16. RouterNormWScaled is the router norm pre-folded by
// RootSize (as MoERouter expects); PerExpertScale is optional. Local dFF (IntermediateSize) and
// expert dFF (ExpertDFF / MoEIntermediateSize) differ, as in the bf16 block.
type MoEQuantLayerWeights struct {
	NumExperts, TopK, ExpertDFF, GroupSize, Bits int

	PreFFNormW, PreFFNorm2W                 []byte
	PostFFNorm1W, PostFFNorm2W, PostFFNormW []byte

	LocalGate, LocalUp, LocalDown QuantWeight // local dense MLP (dFF)

	RouterNormWScaled []byte
	Router            QuantWeight // [NumExperts × dModel] expert-score projection
	PerExpertScale    []byte      // [NumExperts] (nil to skip)

	ExpGate, ExpUp, ExpDown QuantWeight // batched SwitchGLU experts (ExpertDFF)
}

// mlpTransformQuant is mlpTransformBF16 for a 4-bit MLP: gate/up (dModel→dFF) and down
// (dFF→dModel) via QMVBF16, with the SwiGLU activation between — no residual.
func mlpTransformQuant(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	g, err := QMVBF16(x, gate.Packed, gate.Scales, gate.Biases, dFF, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	u, err := QMVBF16(x, up.Packed, up.Scales, up.Biases, dFF, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMulBF16(g, u)
	if err != nil {
		return nil, err
	}
	return QMVBF16(gated, down.Packed, down.Scales, down.Biases, dModel, dFF, groupSize, bits)
}

// MoEBlockQuant is MoEBlockBF16 for a 4-bit MoE layer — the same dual-branch feed-forward
// (local dense MLP + router→topK experts, each independently normed, summed, post-normed,
// residual added once), with QMVBF16 / MoERouterQuant / MoEExpertsQuant in place of the bf16
// ops. The router runs on the raw residual; the local MLP uses dFF, the experts ExpertDFF.
func MoEBlockQuant(h []byte, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockQuant: h must be dModel bf16 bytes")
	}
	numExperts, topK, gs, bits := w.NumExperts, w.TopK, w.GroupSize, w.Bits

	idx, weights, err := MoERouterQuant(h, w.RouterNormWScaled, w.Router, w.PerExpertScale, numExperts, topK, dModel, gs, bits, eps)
	if err != nil {
		return nil, err
	}
	h1In, err := RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h1, err := mlpTransformQuant(h1In, w.LocalGate, w.LocalUp, w.LocalDown, dModel, dFF, gs, bits)
	if err != nil {
		return nil, err
	}
	h2In, err := RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h2, err := MoEExpertsQuant(h2In, idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, w.ExpertDFF, gs, bits)
	if err != nil {
		return nil, err
	}
	h1Normed, err := RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	h2Normed, err := RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	combined, err := AddBF16(h1Normed, h2Normed)
	if err != nil {
		return nil, err
	}
	ffResidual, err := RMSNormBF16(combined, w.PostFFNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(h, ffResidual)
}
