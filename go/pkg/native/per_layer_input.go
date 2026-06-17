// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// gemma4PerLayerCombineScale is gemma4's 1/√2 factor that combines the two per-layer-input
// branches (the per-layer embedding and the projected main embedding).
const gemma4PerLayerCombineScale = 0.70710678118654752440

// PerLayerInputs computes gemma4's per-layer-input tensor for ONE token — the auxiliary
// embedding each layer's per-layer-input gate (PerLayerInputGateBF16) consumes, returned as
// [numLayers · pliDim] bf16 (numLayers contiguous rows of pliDim). Mirrors pkg/metal/model/
// gemma4 perLayerInputTensor op-for-op:
//
//	perLayer  = embed_tokens_per_layer[token] · √pliDim                        (4-bit gather + scale)
//	projected = rms( (per_layer_model_projection · hidden) · 1/√dModel, projNorm )  (per layer-row)
//	combined  = (projected + perLayer) · 1/√2
//
// Mixed weights, matching the checkpoint: the per-layer embedding is 4-bit (packed/scales/
// biases), the model projection + projection norm are bf16. hidden is the main token embedding
// (dModel bf16). projNormW is the PLAIN [pliDim] norm weight, applied per layer-row (rows =
// numLayers, axis = pliDim). Composed from the parity-proven ops.
func PerLayerInputs(
	embedPacked, embedScales, embedBiases []byte,
	projW, projNormW []byte,
	tokenID int32, hidden []byte,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits int, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: hidden must be dModel bf16 bytes")
	}
	plDim := numLayers * pliDim
	if len(projW) != plDim*dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: projW must be (numLayers·pliDim)*dModel bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: projNormW must be pliDim bf16 bytes")
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))

	// (1) per-layer embedding: 4-bit gather the token's [numLayers·pliDim] row, × √pliDim.
	embs, err := EmbedTokensQuant(embedPacked, embedScales, embedBiases, []int32{tokenID}, vocabPLI, plDim, groupSize, bits, embScale)
	if err != nil {
		return nil, err
	}
	perLayer := embs[0]
	// (2) project the main embedding (bf16) → [numLayers·pliDim], × 1/√dModel.
	projected, err := MatVecBF16(projW, hidden, plDim, dModel)
	if err != nil {
		return nil, err
	}
	projected, err = MulBF16(projected, bf16ConstBytes(plDim, projScale))
	if err != nil {
		return nil, err
	}
	// (3) RMSNorm each layer-row with the plain projection norm (rows = numLayers, axis = pliDim).
	projNormed, err := RMSNormBF16(projected, projNormW, numLayers, pliDim, eps)
	if err != nil {
		return nil, err
	}
	// (4) combine the two branches + × 1/√2.
	combined, err := AddBF16(projNormed, perLayer)
	if err != nil {
		return nil, err
	}
	return MulBF16(combined, bf16ConstBytes(plDim, gemma4PerLayerCombineScale))
}

// PerLayerInputGateBF16 applies the gemma4 per-layer-input gate to a layer's output
// hNext (dModel) and returns the gated result. Mirrors pkg/metal/model/gemma4
// decoder_layer.go's per-layer-input block op-for-op:
//
//	gate       = WGate · hNext            (dModel → pliDim)
//	multiplied = gelu(gate) · perLayerInput   (pliDim, the SwiGLU gate-mul)
//	projected  = WProj · multiplied       (pliDim → dModel)
//	hNext      = hNext + rms(projected, PostPerLayerInputNorm)
//
// perLayerInput is this layer's per-token, per-layer input (pliDim bf16) — the slice
// of the per-layer embedding the layer consumes. PostPerLayerInputNorm is the PLAIN
// norm weight (NOT RootSize-scaled like the router's — metal caches this one as a
// plain Copy). Bias-free, matching the rest of the gemma4 native path (q/k/v/o/
// gate/up/down are all bias-free); a checkpoint with per-layer biases is a
// cross-cutting load-time concern. Composed from the parity-proven bf16 ops.
func PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW []byte, dModel, pliDim int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hNext) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: hNext must be dModel bf16 bytes")
	}
	if len(perLayerInput) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: perLayerInput must be pliDim bf16 bytes")
	}
	if len(gateW) != pliDim*dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: gateW must be pliDim*dModel bf16 bytes")
	}
	if len(projW) != dModel*pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: projW must be dModel*pliDim bf16 bytes")
	}
	if len(postNormW) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateBF16: postNormW must be dModel bf16 bytes")
	}

	gate, err := MatVecBF16(gateW, hNext, pliDim, dModel)
	if err != nil {
		return nil, err
	}
	multiplied, err := GeluGateMulBF16(gate, perLayerInput)
	if err != nil {
		return nil, err
	}
	projected, err := MatVecBF16(projW, multiplied, dModel, pliDim)
	if err != nil {
		return nil, err
	}
	projNormed, err := RMSNormBF16(projected, postNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(hNext, projNormed)
}
