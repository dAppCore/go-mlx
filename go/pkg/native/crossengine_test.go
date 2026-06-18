// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"

	g4metal "dappco.re/go/mlx/pkg/metal/model/gemma4"
	"dappco.re/go/mlx/pkg/model"
)

// TestCrossEngine12BPerStep localises the native 12B decode bug against the
// TRUSTED metal engine (metal 12B is coherent). It loads the SAME checkpoint into
// both, opens an incremental decode session on each (the exact path the bug lives
// in), and steps a fixed token sequence through both — comparing, per position,
// the EMBEDDING cosine (native.Embed vs metal.Embed) and the output-HIDDEN cosine
// (native.Step vs metal.Step). Separating the two isolates the cause: if embCos≈1
// but hidCos drops, the divergence is in the decode step (attention/cache), not the
// embed. The first low-hidCos position is where native departs from metal — the
// position the #3 fix targets. Tokens are arbitrary-but-fixed: the bug is in the
// cache/attention mechanism, not token-specific. Real-model load (functional gate,
// Snider-cleared); set CROSS_12B_DIR to the 4bit snapshot.
func TestCrossEngine12BPerStep(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("CROSS_12B_DIR")
	if dir == "" {
		t.Skip("set CROSS_12B_DIR to the gemma-4-12B-it-4bit snapshot")
	}
	const maxLen = 64
	nm, err := LoadGemma4TokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("native load: %v", err)
	}
	sm, ok := nm.(model.SessionModel)
	if !ok {
		t.Fatalf("native model is not a SessionModel")
	}
	ns, err := sm.OpenSession()
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	mm, err := g4metal.LoadGemma4(dir)
	if err != nil {
		t.Fatalf("metal load: %v", err)
	}
	mb, err := g4metal.NewBackend(mm)
	if err != nil {
		t.Fatalf("metal backend: %v", err)
	}
	ms, err := mb.OpenSession()
	if err != nil {
		t.Fatalf("metal session: %v", err)
	}

	ids := make([]int32, 40)
	for i := range ids {
		ids[i] = int32(1000 + i*97) // spread across the vocab, all < 262144
	}
	const captureStep = 6 // a position with meaningful divergence (cross-engine showed hidCos ~0.96 here)
	firstEmb, firstHid := -1, -1
	for i, id := range ids {
		ne, err := nm.Embed(id)
		if err != nil {
			t.Fatalf("native embed pos %d: %v", i, err)
		}
		me, err := mb.Embed(id)
		if err != nil {
			t.Fatalf("metal embed pos %d: %v", i, err)
		}
		ec := cosineBF16(ne, me)
		if i == captureStep { // arm per-layer capture on both engines for this step
			capturedLayerHiddens = nil
			capturedAttnHiddens = nil
			captureLayerHiddens = true
			g4metal.CaptureLayerHiddens(true)
		}
		nh, err := ns.Step(ne)
		if err != nil {
			t.Fatalf("native step %d: %v", i, err)
		}
		mh, err := ms.Step(me)
		if err != nil {
			t.Fatalf("metal step %d: %v", i, err)
		}
		hc := cosineBF16(nh, mh)
		if i == captureStep { // read both engines' per-layer hiddens for this step + diff
			captureLayerHiddens = false
			nl, na := capturedLayerHiddens, capturedAttnHiddens
			ml, ma := g4metal.CapturedLayerHiddens(), g4metal.CapturedAttnHiddens()
			g4metal.CaptureLayerHiddens(false)
			diff := func(label string, nv, mv [][]byte) {
				n := len(nv)
				if len(mv) < n {
					n = len(mv)
				}
				t.Logf("--- %s cross-engine cosine at pos %d (native %d, metal %d) ---", label, i, len(nv), len(mv))
				worst, worstL := 2.0, -1
				for L := 0; L < n; L++ {
					c := cosineBF16(nv[L], mv[L])
					if c < worst {
						worst, worstL = c, L
					}
					t.Logf("  %s L%2d cosine=%.5f", label, L, c)
				}
				t.Logf("--- %s worst layer %d cosine=%.5f ---", label, worstL, worst)
			}
			diff("POST-ATTN", na, ma) // isolates attention (the global-layer suspect)
			diff("POST-LAYER", nl, ml)
		}
		if ec < 0.99 && firstEmb < 0 {
			firstEmb = i
		}
		if hc < 0.99 && firstHid < 0 {
			firstHid = i
		}
		flag := ""
		if hc < 0.99 {
			flag = "  <-- HIDDEN DIVERGES"
		}
		t.Logf("pos %2d  embCos=%.5f  hidCos=%.5f%s", i, ec, hc, flag)
	}
	t.Logf("first embed divergence pos: %d | first hidden divergence pos: %d", firstEmb, firstHid)
}

// cosineBF16 is the cosine similarity of two equal-length bf16 byte vectors — 1.0
// when identical in direction, lower as they diverge (robust to the small
// numerical differences between metal's mlx-c ops and native's kernels, unlike a
// byte compare).
func cosineBF16(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i+1 < len(a); i += 2 {
		av := float64(bf16ToF32(a[i], a[i+1]))
		bv := float64(bf16ToF32(b[i], b[i+1]))
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
