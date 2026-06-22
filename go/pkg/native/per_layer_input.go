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
	projW, projScales, projBiases, projNormW []byte,
	tokenID int32, hidden []byte,
	vocabPLI, numLayers, pliDim, dModel, groupSize, bits, projGS, projBits int, eps float32, projView bufView,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: hidden must be dModel bf16 bytes")
	}
	plDim := numLayers * pliDim
	// projScales present ⇒ the model projection is 4-bit (qat packs, e4b); its packed weight has a
	// different byte span, so only validate the bf16 span when the projection is dense (e2b).
	if len(projScales) == 0 && len(projW) != plDim*dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: bf16 projW must be (numLayers·pliDim)*dModel bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputs: projNormW must be pliDim bf16 bytes")
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))

	// (1) per-layer embedding: gather the token's [numLayers·pliDim] row, × √pliDim. bf16 in regular
	// packs (e2b), 4-bit in qat packs (e4b) — dispatch on the .scales decision, exactly like the
	// projection below, so a bf16 model is a non-event (the shared loader already decided the format).
	var perLayer []byte
	var err error
	if len(embedScales) > 0 {
		var embs [][]byte
		if embs, err = EmbedTokensQuant(embedPacked, embedScales, embedBiases, []int32{tokenID}, vocabPLI, plDim, groupSize, bits, embScale); err != nil {
			return nil, err
		}
		perLayer = embs[0]
	} else {
		var embs [][]byte
		if embs, err = EmbedTokensBF16(embedPacked, []int32{tokenID}, vocabPLI, plDim, embScale); err != nil {
			return nil, err
		}
		perLayer = embs[0]
	}
	// (2) project the main embedding → [numLayers·pliDim], × 1/√dModel. The model projection is
	// bf16 in regular packs (e2b) and 4-bit in qat packs (e4b); dispatch on the presence of scales,
	// so a quantised projection is a non-event — the shared loader already made the .scales decision.
	// (2-6) on the resident bf16 path, run the whole projection chain as ONE command buffer (five GPU
	// round-trips → one). Byte-identical to the unbatched ops below.
	if len(projScales) == 0 && projView.buf != nil {
		return perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
	}
	var projected []byte
	if len(projScales) > 0 {
		projected, err = QMVBF16(hidden, projW, projScales, projBiases, plDim, dModel, projGS, projBits)
	} else {
		projected, err = MatVecBF16(projW, hidden, plDim, dModel)
	}
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

// PerLayerInputGateQuant is PerLayerInputGateBF16 for a 4-bit checkpoint: the gate and
// projection are affine-quantised (per_layer_input_gate / per_layer_projection are 4-bit in the
// served E2B/E4B packs), the post-norm stays bf16. gate is the [pliDim × dModel] quant weight,
// proj the [dModel × pliDim] quant weight; the chain matches PerLayerInputGateBF16 with QMVBF16
// in place of the two bf16 matvecs. perLayerInput is this layer's pliDim slice of the
// PerLayerInputs tensor.
func PerLayerInputGateQuant(hNext []byte, gate QuantWeight, perLayerInput []byte, proj QuantWeight, postNormW []byte, dModel, pliDim, groupSize, bits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hNext) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: hNext must be dModel bf16 bytes")
	}
	if len(perLayerInput) != pliDim*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: perLayerInput must be pliDim bf16 bytes")
	}
	if len(postNormW) != dModel*bf16Size {
		return nil, core.NewError("native.PerLayerInputGateQuant: postNormW must be dModel bf16 bytes")
	}
	g, err := QMVBF16(hNext, gate.Packed, gate.Scales, gate.Biases, pliDim, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	multiplied, err := GeluGateMulBF16(g, perLayerInput)
	if err != nil {
		return nil, err
	}
	projected, err := QMVBF16(multiplied, proj.Packed, proj.Scales, proj.Biases, dModel, pliDim, groupSize, bits)
	if err != nil {
		return nil, err
	}
	projNormed, err := RMSNormBF16(projected, postNormW, 1, dModel, eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(hNext, projNormed)
}
