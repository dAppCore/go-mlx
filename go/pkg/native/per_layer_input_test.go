// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
)

// perLayerInputGateRef rebuilds the per-layer-input gate from the parity-proven
// primitives following the metal rule — the oracle for PerLayerInputGateBF16.
func perLayerInputGateRef(t *testing.T, hNext, gateW, perLayerInput, projW, postNormW []byte, dModel, pliDim int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("perLayerInputGateRef op: %v", err)
		}
		return b
	}
	gate := must(MatVecBF16(gateW, hNext, pliDim, dModel))
	multiplied := must(GeluGateMulBF16(gate, perLayerInput))
	projected := must(MatVecBF16(projW, multiplied, dModel, pliDim))
	projNormed := must(RMSNormBF16(projected, postNormW, 1, dModel, eps))
	return must(AddBF16(hNext, projNormed))
}

// TestPerLayerInputGate gates the gemma4 per-layer-input gate. PerLayerInputGateBF16
// is byte-for-byte the independent reference that wires the gate (gate → gelu-mul →
// project → norm → residual) from primitives — proving the WIRING (each dim/op in the
// right place, the residual), since the sub-ops are gated elsewhere. A non-vacuous
// check confirms the gate genuinely modifies the layer output (out ≠ hNext). pliDim ≠
// dModel deliberately, to catch a dimension mixup in the two projections.
func TestPerLayerInputGate(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, pliDim = 256, 128
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	hNext := toBF16Bytes(mk(dModel, 29))
	gateW := toBF16Bytes(mk(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(mk(pliDim, 7))
	projW := toBF16Bytes(mk(dModel*pliDim, 23))
	postNormW := toBF16Bytes(mk(dModel, 5))

	got, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	if err != nil {
		t.Fatalf("PerLayerInputGateBF16: %v", err)
	}
	want := perLayerInputGateRef(t, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	eqBytes(t, "PerLayerInputGateBF16", got, want)

	// non-vacuous: the gate must change the layer output (the projected, normed
	// per-layer contribution is summed in, not a no-op).
	same := len(got) == len(hNext)
	for i := range got {
		if i < len(hNext) && got[i] != hNext[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("PerLayerInputGateBF16 output equals hNext unchanged — the gate did not contribute")
	}
	t.Logf("per-layer-input gate (dModel %d, pliDim %d): ≡ composed reference and modifies the layer output", dModel, pliDim)
}
