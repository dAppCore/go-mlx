// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// nsa_bench_test.go is the AX-11 allocation instrument for the NSA mixer hot
// paths. The benches drive Mixer.Forward (the three-branch blend) and the
// decode-step path (L=1) on small synthetic shapes so allocs/op reflects the
// kernel composition, not model load. Build with `-tags metal_runtime` and
// MLX_METALLIB_PATH set. The fixtures (identityMixer / constGateProj) are
// shared with the oracle test; identity Q/K/V/O keep the math trivial so the
// allocation profile is the only thing under measurement.
//
//	go test -tags metal_runtime -run='^$' -bench=NSA -benchmem -benchtime=20x ./pkg/metal/model/nsa/

// benchInput builds a [1,L,2] identity-friendly input (hidden = HeadDim = 2,
// matching identityMixer) with deterministic non-degenerate rows.
func benchInput(L int32) *metal.Array {
	data := make([]float32, int(L)*2)
	for t := int32(0); t < L; t++ {
		data[int(t)*2] = float32(t%3) + 1       // varied so attention isn't uniform
		data[int(t)*2+1] = float32((t + 1) % 2) // 0/1 alternating
	}
	return metal.FromValues(data, 1, int(L), 2)
}

// BenchmarkForward_TopN drives the full three-branch path in the TOP-N
// selection regime (0 < SelectBlocks < nBlocks), which is the only regime that
// executes selectionMask's TopK+SliceAxis path (the keep-all regime short-
// circuits to a Clone). BlockSize=2, L=8 → nBlocks=4, SelectBlocks=2. This
// exercises BOTH known SliceAxis sites: branchGate ×3 (nsa.go) + selectionMask
// ×1 (kernels.go). It is the headline allocation bench for the package.
func BenchmarkForward_TopN(b *testing.B) {
	requireMetalRuntime(b)

	const window = 4
	m := identityMixer(constGateProj(0.4, -0.7, 1.1), 2, 2, window, 1.0) // BlockSize=2, SelectBlocks=2
	defer freeMixer(m)

	const L = int32(8) // nBlocks = 8/2 = 4 > SelectBlocks=2 → top-n path live
	x := benchInput(L)
	defer metal.Free(x)
	ctx := &metal.MixerCtx{B: 1, L: L}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ := m.Forward(x, ctx)
		metal.Free(out)
	}
}

// BenchmarkForward_KeepAll drives the full three-branch path in the keep-all
// selection regime (SelectBlocks >= nBlocks): selectionMask short-circuits to a
// Clone, so this isolates the branchGate ×3 SliceAxis cost (nsa.go) from the
// selectionMask site. BlockSize=2, L=4 → nBlocks=2, SelectBlocks=2.
func BenchmarkForward_KeepAll(b *testing.B) {
	requireMetalRuntime(b)

	const window = 2
	m := identityMixer(constGateProj(0.4, -0.7, 1.1), 2, 2, window, 1.0)
	defer freeMixer(m)

	const L = int32(4) // nBlocks = 4/2 = 2 == SelectBlocks → keep-all path
	x := benchInput(L)
	defer metal.Free(x)
	ctx := &metal.MixerCtx{B: 1, L: L}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ := m.Forward(x, ctx)
		metal.Free(out)
	}
}

// BenchmarkForward_Decode drives the decode-step path (L=1 < BlockSize): NSA
// reduces to its sliding-window branch (attend + slidingMask), no compression /
// selection / gate blend. This is the per-token allocation profile — it does
// NOT touch either SliceAxis suspect, but it is the highest-multiplier path
// (fires every emitted token) so its floor matters most for serve.
func BenchmarkForward_Decode(b *testing.B) {
	requireMetalRuntime(b)

	const window = 4
	m := identityMixer(constGateProj(0, 0, 0), 8, 1, window, 1.0) // BlockSize=8 > L=1
	defer freeMixer(m)

	const L = int32(1)
	x := benchInput(L)
	defer metal.Free(x)
	ctx := &metal.MixerCtx{B: 1, L: L}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ := m.Forward(x, ctx)
		metal.Free(out)
	}
}

// BenchmarkSelectionMask_TopN isolates the kernels.go SliceAxis site: the
// top-n threshold slice off the TopK result. blockScores [1,1,L,nBlocks] with
// SelectBlocks in (0, nBlocks) so the TopK+SliceAxis path runs every call.
func BenchmarkSelectionMask_TopN(b *testing.B) {
	requireMetalRuntime(b)

	// L=4 queries, nBlocks=4, all causally valid; keep top-2.
	scores := metal.FromValues([]float32{
		3, 1, 2, 0,
		3, 1, 2, 4,
		5, 1, 2, 3,
		3, 6, 2, 1,
	}, 1, 1, 4, 4)
	causal := metal.FromValues(make([]float32, 16), 1, 1, 4, 4) // all 0 = valid
	defer metal.Free(scores, causal)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := selectionMask(scores, causal, 2)
		metal.Free(out)
	}
}
