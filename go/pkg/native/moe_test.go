// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
)

// moeExpertsRef is the oracle for MoEExperts, composed from the parity-proven
// standalone ops: per selected expert, MatVec gate/up, GeluGateMul, MatVec down,
// scale by the router weight, accumulate. Mirrors MoEExperts op-for-op.
func moeExpertsRef(t *testing.T, x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int) []byte {
	t.Helper()
	gateSz, downSz := dFF*dModel*bf16Size, dModel*dFF*bf16Size
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moeExpertsRef op: %v", err)
		}
		return b
	}
	var acc []byte
	for i := 0; i < topK; i++ {
		e := int(idx[i])
		gate := must(MatVecBF16(gateW[e*gateSz:(e+1)*gateSz], x, dFF, dModel))
		up := must(MatVecBF16(upW[e*gateSz:(e+1)*gateSz], x, dFF, dModel))
		act := must(GeluGateMulBF16(gate, up))
		downE := must(MatVecBF16(downW[e*downSz:(e+1)*downSz], act, dModel, dFF))
		scaled := must(MulBF16(downE, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	return acc
}

// TestMoEExperts gates the MoE expert branch: the chained on-device MoEExperts (top-k
// experts' SwiGLU + router-weighted combine) is byte-for-byte the composed reference
// of proven standalone ops, on the same routing + expert weights. The routing
// (top-k + softmax) and the dual-branch composition are separate sub-slices.
func TestMoEExperts(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF = 8, 2, 256, 512
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := toBF16Bytes(mk(dModel, 37))
	gateW := toBF16Bytes(mk(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(mk(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(mk(numExperts*dModel*dFF, 47))
	idx := []int32{5, 2} // an arbitrary top-2 selection
	weights := toBF16Bytes([]float32{0.6, 0.4})

	got, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	if err != nil {
		t.Fatalf("MoEExperts: %v", err)
	}
	want := moeExpertsRef(t, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	eqBytes(t, "MoEExperts", got, want)
	t.Logf("MoEExperts(%d experts, top-%d): chained expert branch ≡ composed reference (per-expert SwiGLU + weighted combine)", numExperts, topK)
}
