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
