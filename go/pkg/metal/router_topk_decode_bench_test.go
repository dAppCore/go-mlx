// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// gemma4's Gemma4Router.forward selects top-k experts with 6 generic ops
// (Argpartition + SliceAxis + TakeAlongAxis + Softmax + Take + Mul) while the
// fused nativeMoERouterTopK (used in production by the other MoE families via
// moe_router.go) does it in ONE Metal kernel. These benches measure the gap on
// the decode shape; the gemma4 router is ~8% of e4b's per-token budget and this
// is the bulk of it.

func benchRouterScores() (*Array, *Array) {
	const experts = 128
	scores := RandomUniform(-2, 2, []int32{1, 1, experts}, DTypeFloat32)
	perExpertScale := RandomUniform(0.5, 1.5, []int32{experts}, DTypeFloat32)
	Materialize(scores, perExpertScale)
	return scores, perExpertScale
}

// Generic 6-op top-k, mirroring router.go.
func genericRouterTopK(scores, perExpertScale *Array, experts, topK int) (*Array, *Array) {
	kth := experts - topK
	idx := Argpartition(scores, kth, -1)
	sliced := SliceAxis(idx, -1, int32(kth), int32(experts))
	Free(idx)
	weights := TakeAlongAxis(scores, sliced, -1)
	soft := Softmax(weights)
	Free(weights)
	scale := Take(perExpertScale, sliced, 0)
	weighted := Mul(soft, scale)
	Free(soft, scale)
	return sliced, weighted
}

// TestRouterTopK_FusedMatchesGeneric guards the gemma4 router rewire: the fused
// nativeMoERouterTopK must produce the same selected experts + weights as the
// generic 6-op path on random scores (set equality — order may differ but the
// downstream weighted sum is order-invariant).
func TestRouterTopK_FusedMatchesGeneric(t *testing.T) {
	requireMetalRuntime(t)
	const experts, topK = 128, 4
	scores := RandomUniform(-2, 2, []int32{1, 1, experts}, DTypeFloat32)
	perExpertScale := RandomUniform(0.5, 1.5, []int32{experts}, DTypeFloat32)
	Materialize(scores, perExpertScale)
	defer Free(scores, perExpertScale)

	gi, gw := genericRouterTopK(scores, perExpertScale, experts, topK)
	fi, fw, ok, err := nativeMoERouterTopK(scores, perExpertScale, topK)
	if !ok || err != nil {
		t.Fatalf("fused ok=%v err=%v", ok, err)
	}
	if err := Eval(gi, gw, fi, fw); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer Free(gi, gw, fi, fw)

	gIdx, gWt := gi.DataInt32(), gw.Floats()
	fIdx, fWt := fi.DataInt32(), fw.Floats()
	if len(gIdx) != topK || len(fIdx) != topK {
		t.Fatalf("topK count: generic %d fused %d, want %d", len(gIdx), len(fIdx), topK)
	}
	gMap := map[int32]float32{}
	for i, id := range gIdx {
		gMap[id] = gWt[i]
	}
	for i, id := range fIdx {
		gv, present := gMap[id]
		if !present {
			t.Fatalf("fused selected expert %d not in generic set %v", id, gIdx)
		}
		if d := gv - fWt[i]; d > 1e-3 || d < -1e-3 {
			t.Errorf("weight mismatch expert %d: generic %v fused %v", id, gv, fWt[i])
		}
	}
}

func BenchmarkRouterTopK_Generic_Batched32(b *testing.B) {
	const experts, topK, N = 128, 4, 32
	scores, perExpertScale := benchRouterScores()
	defer Free(scores, perExpertScale)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		for range N {
			idx, w := genericRouterTopK(scores, perExpertScale, experts, topK)
			outs = append(outs, idx, w)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkRouterTopK_Fused_Batched32(b *testing.B) {
	const topK, N = 4, 32
	scores, perExpertScale := benchRouterScores()
	defer Free(scores, perExpertScale)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		for range N {
			idx, w, ok, err := nativeMoERouterTopK(scores, perExpertScale, topK)
			if !ok || err != nil {
				b.Fatalf("fused topk ok=%v err=%v", ok, err)
			}
			outs = append(outs, idx, w)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}
