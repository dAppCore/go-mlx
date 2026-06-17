// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
)

// moeBlockRef is the oracle for MoEBlockBF16: it rebuilds BOTH branches from the
// parity-proven primitives (local MLP inline; expert branch via moeExpertsRef) and
// wires the five norms + dual-branch sum + residual exactly as
// pkg/metal/model/gemma4 decoder_layer.go's MoE branch. It calls the SAME MoERouter
// as the block, so the expert accumulation order — and thus the bf16 rounding —
// matches, allowing a byte-for-byte gate.
func moeBlockRef(t *testing.T, h []byte, w MoELayerWeights, numExperts, topK, dModel, dFF, expertDFF int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moeBlockRef op: %v", err)
		}
		return b
	}
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		t.Fatalf("moeBlockRef router: %v", err)
	}
	// local dense MLP branch, rebuilt from primitives (no residual).
	h1In := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	g := must(MatVecBF16(w.WGate, h1In, dFF, dModel))
	u := must(MatVecBF16(w.WUp, h1In, dFF, dModel))
	h1 := must(MatVecBF16(w.WDown, must(GeluGateMulBF16(g, u)), dModel, dFF))
	// expert branch on the separately-normed input, rebuilt via moeExpertsRef.
	h2In := must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps))
	h2 := moeExpertsRef(t, h2In, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, expertDFF)
	// independent norms, sum, post-norm, residual.
	h1n := must(RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps))
	h2n := must(RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps))
	ff := must(RMSNormBF16(must(AddBF16(h1n, h2n)), w.PostFFNormW, 1, dModel, eps))
	return must(AddBF16(h, ff))
}

// denseFFNRef is the NON-MoE feed-forward (decoder_layer.go's else branch): a single
// dense MLP, rms(h, PreFFNorm) → MLP → rms(·, PostFFNorm) → + h. Used only to prove
// MoEBlockBF16's expert branch genuinely contributes (the dual-branch output must
// differ from running the local MLP alone).
func denseFFNRef(t *testing.T, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("denseFFNRef op: %v", err)
		}
		return b
	}
	ffIn := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	ff := must(mlpTransformBF16(ffIn, w.WGate, w.WUp, w.WDown, dModel, dFF))
	return must(AddBF16(h, must(RMSNormBF16(ff, w.PostFFNormW, 1, dModel, eps))))
}

// TestMoEBlock gates the dual-branch MoE feed-forward composition. MoEBlockBF16 is
// byte-for-byte the independent reference that rebuilds both branches from primitives
// and wires the five norms + dual-branch sum + residual per the metal rule — proving
// the WIRING (each norm in the right place, both branches live, the single residual)
// since the sub-ops are individually gated elsewhere. A non-vacuous check confirms
// the expert branch actually contributes: the dual-branch output differs from the
// dense-MLP-only FFN. Local dFF and expertDFF deliberately differ (catch a dim mixup).
func TestMoEBlock(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, expertDFF = 8, 2, 256, 512, 384
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	h := toBF16Bytes(mk(dModel, 29))
	w := MoELayerWeights{
		PreFFNormW:        toBF16Bytes(mk(dModel, 3)),
		PreFFNorm2W:       toBF16Bytes(mk(dModel, 5)),
		PostFFNorm1W:      toBF16Bytes(mk(dModel, 7)),
		PostFFNorm2W:      toBF16Bytes(mk(dModel, 11)),
		PostFFNormW:       toBF16Bytes(mk(dModel, 13)),
		WGate:             toBF16Bytes(mk(dFF*dModel, 17)),
		WUp:               toBF16Bytes(mk(dFF*dModel, 19)),
		WDown:             toBF16Bytes(mk(dModel*dFF, 23)),
		RouterNormWScaled: toBF16Bytes(mk(dModel, 2)),
		RouterW:           toBF16Bytes(mk(numExperts*dModel, 43)),
		PerExpertScale:    toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1}),
		ExpGateW:          toBF16Bytes(mk(numExperts*expertDFF*dModel, 53)),
		ExpUpW:            toBF16Bytes(mk(numExperts*expertDFF*dModel, 71)),
		ExpDownW:          toBF16Bytes(mk(numExperts*dModel*expertDFF, 47)),
	}

	got, err := MoEBlockBF16(h, w, numExperts, topK, dModel, dFF, expertDFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}
	want := moeBlockRef(t, h, w, numExperts, topK, dModel, dFF, expertDFF, eps)
	eqBytes(t, "MoEBlockBF16", got, want)

	// non-vacuous: the dual-branch output must differ from the dense-MLP-only FFN
	// (i.e. the expert branch is genuinely summed in, not silently dropped).
	dense := denseFFNRef(t, h, w, dModel, dFF, eps)
	same := len(dense) == len(got)
	for i := range got {
		if i < len(dense) && got[i] != dense[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("MoEBlockBF16 output equals the dense-MLP-only FFN — the expert branch did not contribute")
	}
	t.Logf("MoEBlock (%d experts, top-%d, dFF %d / expertDFF %d): dual-branch ≡ composed reference and differs from dense-only FFN", numExperts, topK, dFF, expertDFF)
}
