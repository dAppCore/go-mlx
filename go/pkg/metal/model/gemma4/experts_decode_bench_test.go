// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// benchMakeQ4Switch builds a q4 affine expert-indexed weight for perf benches.
// Values are arbitrary (timing is value-independent); only the packed shape
// [experts, outDim, inDim/8] + group sidecars must be valid for GatherQMM.
func benchMakeQ4Switch(experts, outDim, inDim int) *metal.SwitchLinear {
	packedIn := inDim / 8 // q4: 8 values per uint32
	groups := inDim / 64
	weightWords := make([]uint32, experts*outDim*packedIn)
	for i := range weightWords {
		weightWords[i] = uint32(i*1664525 + 1013904223)
	}
	scales := make([]float32, experts*outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.005 * float32((i%17)+1)
		biases[i] = -0.03 + 0.002*float32(i%31)
	}
	return metal.NewQuantizedSwitchLinear(
		metal.FromValues(weightWords, experts, outDim, packedIn),
		metal.FromValues(scales, experts, outDim, groups),
		metal.FromValues(biases, experts, outDim, groups),
		nil, 64, 4,
	)
}

// benchmarkGemma4ExpertsDecode measures the REAL MoE expert decode path
// (Gemma4Experts.forward: gate+up+down GatherQMM over top-k experts + GELU +
// weighted sum) — the bulk of 26B-A4B's per-token expert cost (the ONLY
// gemma-4 size with MoE; E2B/E4B/12B/31B are all dense and never run this
// path — a prior version of this comment mislabelled it "e4b"). Single token,
// chained, one Eval, so ns/op / N = real per-token expert cost below the sync
// floor. Run at topK 1/2/4 as a scaling diagnostic (the real 26B-A4B config
// declares top_k_experts=8 out of num_experts=128 — see the experts/hidden/
// moeDim constants below, which now match that real config's
// hidden_size=2816 / moe_intermediate_size=704 rather than arbitrary toy
// values): if per-call scales ~linearly with topK the cost is the per-expert
// gather (inherent); if it's ~flat there's large FIXED overhead (the combine /
// 5D reshape / gather setup) that's the fixable target.
func benchmarkGemma4ExpertsDecode(b *testing.B, experts, hidden, moeDim, topK, n int) {
	// Build GateUpProj (fused gate+up, output 2*moeDim) exactly as the loader
	// does (load.go:231). The prior shape left it nil, so the bench ran the
	// 3-gather fallback while serve runs the fused 2-gather path for single-token
	// decode (gemma4UseFusedExpertGateUp: Dim(1)==1) — over-stating by one gather.
	layer := &Gemma4Experts{
		GateUpProj: benchMakeQ4Switch(experts, 2*moeDim, hidden),
		GateProj:   benchMakeQ4Switch(experts, moeDim, hidden),
		UpProj:     benchMakeQ4Switch(experts, moeDim, hidden),
		DownProj:   benchMakeQ4Switch(experts, hidden, moeDim),
	}
	defer func() {
		metal.FreeSwitchLinear(layer.GateUpProj)
		metal.FreeSwitchLinear(layer.GateProj)
		metal.FreeSwitchLinear(layer.UpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	x0 := metal.RandomUniform(-1, 1, []int32{1, 1, int32(hidden)}, metal.DTypeFloat32)
	idxVals := make([]int32, topK)
	wVals := make([]float32, topK)
	for i := range idxVals {
		idxVals[i] = int32(i % experts)
		wVals[i] = 0.5
	}
	topKIndices := metal.FromValues(idxVals, 1, 1, topK)
	topKWeights := metal.FromValues(wVals, 1, 1, topK)
	metal.Materialize(x0, topKIndices, topKWeights)
	defer metal.Free(x0, topKIndices, topKWeights)

	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*metal.Array, 0, n)
		x := x0
		for range n {
			out := layer.forward(x, topKIndices, topKWeights, "")
			outs = append(outs, out)
			x = out
		}
		if err := metal.Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(outs...)
	}
}

// experts=128, hidden=2816, moeDim=704 match the real 26B-A4B MoE config
// (num_experts / hidden_size / moe_intermediate_size); topK is swept below
// 1/2/4 as the scaling diagnostic, plus 8 for the real deployment's
// top_k_experts.
func BenchmarkGemma4Experts_Decode_Q4_TopK1_Batched32(b *testing.B) {
	benchmarkGemma4ExpertsDecode(b, 128, 2816, 704, 1, 32)
}
func BenchmarkGemma4Experts_Decode_Q4_TopK2_Batched32(b *testing.B) {
	benchmarkGemma4ExpertsDecode(b, 128, 2816, 704, 2, 32)
}
func BenchmarkGemma4Experts_Decode_Q4_TopK4_Batched32(b *testing.B) {
	benchmarkGemma4ExpertsDecode(b, 128, 2816, 704, 4, 32)
}
func BenchmarkGemma4Experts_Decode_Q4_TopK8_Batched32(b *testing.B) {
	benchmarkGemma4ExpertsDecode(b, 128, 2816, 704, 8, 32)
}

// Direct GatherQMM shape probe: is the 5D input experts.go feeds the gather
// (x[1,1,h] -> ExpandDims x2 -> [1,1,1,1,h]) forcing a slow path vs a 3D input?
// Same gate weight + indices, only the input rank differs. MulScalar (constant
// across both) makes each call distinct so MLX cannot dedup. If 3D is much faster
// AND produces the same logits (checked in a separate test), the 5D expand is the
// fixable overhead; if it errors, the rank is required by gather_qmm.
func benchmarkExpertGatherQMMShape(b *testing.B, rank, n int) {
	// Real 26B-A4B MoE config shape (num_experts/hidden_size/moe_intermediate_size).
	const experts, hidden, moeDim, topK = 128, 2816, 704, 2
	gate := benchMakeQ4Switch(experts, moeDim, hidden)
	defer metal.FreeSwitchLinear(gate)
	idx := make([]int32, topK)
	for i := range idx {
		idx[i] = int32(i % experts)
	}
	var inShape []int32
	var idxShape []int
	switch rank {
	case 5:
		inShape = []int32{1, 1, 1, 1, hidden}
		idxShape = []int{1, 1, topK}
	case 3:
		inShape = []int32{1, 1, hidden}
		idxShape = []int{1, topK}
	}
	topKIndices := metal.FromValues(idx, idxShape...)
	base := metal.RandomUniform(-1, 1, inShape, metal.DTypeFloat32)
	metal.Materialize(base, topKIndices)
	defer metal.Free(base, topKIndices)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*metal.Array, 0, n)
		for i := 0; i < n; i++ {
			x := metal.MulScalar(base, float32(i+1))
			out := metal.GatherQMM(x, gate.Weight, gate.Scales, gate.Biases, nil, topKIndices, true, gate.GroupSize, gate.Bits, gate.QuantizationMode, false)
			metal.Free(x)
			outs = append(outs, out)
		}
		if err := metal.Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(outs...)
	}
}

func BenchmarkGemma4ExpertGatherQMM_Shape5D_Batched32(b *testing.B) {
	benchmarkExpertGatherQMMShape(b, 5, 32)
}
func BenchmarkGemma4ExpertGatherQMM_Shape3D_Batched32(b *testing.B) {
	benchmarkExpertGatherQMMShape(b, 3, 32)
}

// BenchmarkGemma4Router_Decode runs the MoE router (RMSNorm + tiny hidden->experts
// projection + Argpartition top-k + TakeAlongAxis + Softmax) once per token per
// MoE layer. Compute is tiny but it's ~8 small dispatches; this measures whether
// the routing overhead is a real per-token cost on top of the experts. Fresh
// random input per call (RMSNorm is scale-invariant, so scaling would dedup).
func BenchmarkGemma4Router_Decode_Batched32(b *testing.B) {
	const hidden, numExperts, topK, N = 2048, 128, 4, 32
	router := &Gemma4Router{
		Proj:     metal.NewLinear(metal.RandomUniform(-0.05, 0.05, []int32{numExperts, hidden}, metal.DTypeFloat32), nil),
		Scale:    metal.RandomUniform(0.5, 1.5, []int32{hidden}, metal.DTypeFloat32),
		RootSize: 1.0,
		TopK:     topK,
		Eps:      1e-6,
	}
	// Precompute ScaleScaled exactly as the loader does (weights.go:705) so the
	// bench exercises serve's path; the prior shape left it nil, so every call
	// ran a MulScalar the production router never pays.
	router.ScaleScaled = metal.MulScalar(router.Scale, router.RootSize)
	// Pre-generate N distinct inputs OUTSIDE the timed loop. Generating them
	// inside (the prior shape) put a RandomUniform RNG kernel per call into the
	// measurement — that dominated the number, NOT the router. Distinct inputs
	// still stop MLX CSE-ing the forward.
	inputs := make([]*metal.Array, N)
	for i := range inputs {
		inputs[i] = metal.RandomUniform(-1, 1, []int32{1, 1, hidden}, metal.DTypeFloat32)
	}
	warm := append([]*metal.Array{router.Proj.Weight, router.Scale, router.ScaleScaled}, inputs...)
	metal.Materialize(warm...)
	defer metal.FreeLinear(router.Proj)
	defer metal.Free(router.Scale, router.ScaleScaled)
	defer metal.Free(inputs...)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		held := make([]*metal.Array, 0, N*2)
		for i := 0; i < N; i++ {
			idx, w := router.forward(inputs[i])
			held = append(held, idx, w)
		}
		if err := metal.Eval(held...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(held...)
	}
}
