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

// TestMoERouterQuant gates the 4-bit router: MoERouterQuant ≡ the manual RMSNorm → QMVBF16 →
// routerSelect composition, and its selected expert SET matches an independent max-scan top-k
// over the same scores (non-circular on the selection).
func TestMoERouterQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, gs, bits = 8, 3, 64, 32, 4
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	pp, ps, pb := quantizeProj(t, numExperts, dModel, gs, bits, 13)
	proj := QuantWeight{Packed: pp, Scales: ps, Biases: pb}
	x := toBF16Bytes(mk(dModel, 7))
	norm := toBF16Bytes(mk(dModel, 9))
	scale := toBF16Bytes(mk(numExperts, 5))

	idx, weights, err := MoERouterQuant(x, norm, proj, scale, numExperts, topK, dModel, gs, bits, eps)
	if err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		t.Fatalf("idx %d / weights %d, want topK %d", len(idx), len(weights)/bf16Size, topK)
	}

	// wiring: ≡ manual RMSNorm → QMVBF16 → routerSelect.
	normed, err := RMSNormBF16(x, norm, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	scoresB, err := QMVBF16(normed, proj.Packed, proj.Scales, proj.Biases, numExperts, dModel, gs, bits)
	if err != nil {
		t.Fatalf("QMVBF16: %v", err)
	}
	wantIdx, wantW := routerSelect(scoresB, scale, numExperts, topK)
	if !bytes.Equal(weights, wantW) {
		t.Fatal("MoERouterQuant weights != manual routerSelect")
	}
	set := func(ids []int32) map[int32]bool {
		m := map[int32]bool{}
		for _, e := range ids {
			m[e] = true
		}
		return m
	}
	got, want := set(idx), set(wantIdx)
	if len(got) != len(want) {
		t.Fatal("idx set size mismatch vs manual")
	}
	for e := range want {
		if !got[e] {
			t.Fatalf("idx set != manual (missing %d)", e)
		}
	}

	// independent selection: the topK highest scores by max-scan must equal idx as a set.
	sc := make([]float32, numExperts)
	for e := 0; e < numExperts; e++ {
		sc[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	used := map[int32]bool{}
	for n := 0; n < topK; n++ {
		best, bv := int32(-1), float32(-1e30)
		for e := 0; e < numExperts; e++ {
			if !used[int32(e)] && sc[e] > bv {
				best, bv = int32(e), sc[e]
			}
		}
		used[best] = true
	}
	for e := range got {
		if !used[e] {
			t.Fatalf("selected expert %d not in the independent max-scan top-k", e)
		}
	}
	t.Logf("4-bit router: MoERouterQuant ≡ RMSNorm→QMV→routerSelect, expert set ≡ independent max-scan top-k %v", idx)
}

// TestMoEBlockQuant gates the 4-bit dual-branch MoE block: MoEBlockQuant ≡ the composed
// reference (router → local quant MLP + quant experts, each normed, summed, post-normed,
// residual) byte-for-byte, and transforms the residual (non-vacuous).
func TestMoEBlockQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, dFF, expertDFF, numExperts, topK, gs, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	qw := func(outDim, inDim, salt int) QuantWeight {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	batched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := 0; e < numExperts; e++ {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	nrm := func(salt int) []byte { return toBF16Bytes(mk(dModel, salt)) }
	w := MoEQuantLayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		ExpertGroupSize: gs, ExpertBits: bits, LocalGroupSize: gs, LocalBits: bits, RouterGroupSize: gs, RouterBits: bits,
		PreFFNormW: nrm(13), PreFFNorm2W: nrm(17), PostFFNorm1W: nrm(19), PostFFNorm2W: nrm(23), PostFFNormW: nrm(29),
		LocalGate: qw(dFF, dModel, 3), LocalUp: qw(dFF, dModel, 31), LocalDown: qw(dModel, dFF, 37),
		RouterNormWScaled: nrm(41), Router: qw(numExperts, dModel, 43), PerExpertScale: toBF16Bytes(mk(numExperts, 47)),
		ExpGate: batched(expertDFF, dModel, 53), ExpUp: batched(expertDFF, dModel, 101), ExpDown: batched(dModel, expertDFF, 149),
	}
	h := toBF16Bytes(mk(dModel, 5))

	got, err := MoEBlockQuant(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockQuant: %v", err)
	}

	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	idx, weights, err := MoERouterQuant(h, w.RouterNormWScaled, w.Router, w.PerExpertScale, numExperts, topK, dModel, gs, bits, eps)
	if err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}
	h1 := must(mlpTransformQuant(must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps)), w.LocalGate, w.LocalUp, w.LocalDown, dModel, dFF, gs, bits))
	h2 := must(MoEExpertsQuant(must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps)), idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, gs, bits))
	combined := must(AddBF16(must(RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps)), must(RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps))))
	want := must(AddBF16(h, must(RMSNormBF16(combined, w.PostFFNormW, 1, dModel, eps))))
	if !bytes.Equal(got, want) {
		t.Fatal("MoEBlockQuant != composed dual-branch reference")
	}
	if bytes.Equal(got, h) {
		t.Fatal("MoEBlockQuant did not transform the residual")
	}
	t.Logf("4-bit MoE block: dual-branch (quant local MLP + quant experts, router-gated) ≡ composed reference, byte-for-byte")
}
