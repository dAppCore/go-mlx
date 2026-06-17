// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

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
