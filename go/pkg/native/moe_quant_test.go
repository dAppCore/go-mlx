// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"bytes"
	"os"
	"testing"
)

// TestMoEExpertsQuant gates the 4-bit batched experts: MoEExpertsQuant over a SwitchGLU-style
// batched quant tensor must equal a composed reference (per selected expert: QMVBF16 gate/up →
// GeluGateMulBF16 → QMVBF16 down, weighted-summed) byte-for-byte, and differ for a different
// expert selection (the routing is genuinely consumed).
func TestMoEExpertsQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	// batch each expert's [outDim × inDim] quant weight into one tensor (the SwitchGLU layout).
	buildBatched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := 0; e < numExperts; e++ {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	gate := buildBatched(dFF, dModel, 3)
	up := buildBatched(dFF, dModel, 51)
	down := buildBatched(dModel, dFF, 91)
	x := toBF16Bytes(mk(dModel, 5))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.7, 0.3})

	got, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant: %v", err)
	}

	gp, gsz := dFF*dModel*bits/8, dFF*(dModel/gs)*bf16Size
	dp, dsz := dModel*dFF*bits/8, dModel*(dFF/gs)*bf16Size
	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	var acc []byte
	for i, e := range idx {
		ee := int(e)
		ge := must(QMVBF16(x, gate.Packed[ee*gp:(ee+1)*gp], gate.Scales[ee*gsz:(ee+1)*gsz], gate.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		ue := must(QMVBF16(x, up.Packed[ee*gp:(ee+1)*gp], up.Scales[ee*gsz:(ee+1)*gsz], up.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		gg := must(GeluGateMulBF16(ge, ue))
		de := must(QMVBF16(gg, down.Packed[ee*dp:(ee+1)*dp], down.Scales[ee*dsz:(ee+1)*dsz], down.Biases[ee*dsz:(ee+1)*dsz], dModel, dFF, gs, bits))
		scaled := must(MulBF16(de, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	if !bytes.Equal(got, acc) {
		t.Fatal("MoEExpertsQuant != composed quant reference")
	}
	// non-vacuous: a different expert selection changes the result.
	other, err := MoEExpertsQuant(x, []int32{1, 3}, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant(other): %v", err)
	}
	if bytes.Equal(got, other) {
		t.Fatal("different expert selection produced the same output (routing not consumed)")
	}
	t.Logf("4-bit batched experts: topK SwiGLU over the SwitchGLU tensor ≡ composed QMV reference, selection-sensitive")
}
