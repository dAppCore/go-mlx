// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package retnet

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// retnet_bench_test.go measures the RetNet retention kernel at a representative
// synthetic geometry — no model load (AX-11), the mamba2/rwkv7/deltanet bench
// style. It closes the long-standing measurement gap: retnet had no benchmarks
// at all, so its hot-path allocation behaviour was invisible while its siblings
// (rwkv7's WKV7, mamba2's SSDScan) were swept.
//
// Two shapes matter, and they exercise different code:
//
//   - The chunked prefill path (compute → advanceState, plus crossChunk when a
//     prior state is carried) runs once per chunk. advanceState slices the last
//     row of the [H,L,L] decay matrix — the rank-3 metal.Slice site whose two
//     []int32{…} literals heap-alloc per chunk.
//   - The single-token decode path (decodeStep, L==1) is the steady-state
//     generation kernel: one recurrent update, no decay mask, no cumsum. It is
//     the analogue of the rwkv7/mamba2 sequential decode that gave the big
//     wins — but it composes from different ops (outer-product Matmul + a
//     per-head γ Exp), so its allocation profile is its own question.
//
// decodeStep is benched with a non-nil prior state: that is the only branch
// that runs gammaPerHead + the decay Mul/Add, i.e. the real steady-state decode
// after the first token. Benching it with prev==nil would skip that work and
// measure the unrepresentative first-token case.
//
// Both forms are force-Eval'd inside the timed region: a lazy return measures
// graph build, not compute (the MLA-bench lesson, inherited via deltanet).
//
// Run: go test -tags metal_runtime -bench='Retnet' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/retnet/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	retnetBenchH = 4
	retnetBenchD = 64
	retnetBenchL = 256 // representative prefill chunk
)

func retnetBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// retnetBenchDecay builds the per-head ln(γ) the kernel consumes: γ_h = 1−2^(−5−h),
// the RetNet schedule. Content is irrelevant to timing but kept honest.
func retnetBenchDecay(h int) []float32 {
	ln := make([]float32, h)
	for i := range ln {
		gamma := 1.0 - math.Pow(2, -5-float64(i))
		ln[i] = float32(math.Log(gamma))
	}
	return ln
}

// retnetBenchQKV builds q/k/v [1,H,L,D] with cheap deterministic fills — values
// do not affect allocation counts or op timing.
func retnetBenchQKV(h, l, d int32) (q, k, v *metal.Array, free func()) {
	fill := func(seed float32) []float32 {
		s := make([]float32, h*l*d)
		for i := range s {
			s[i] = seed + 0.01*float32(i%13) - 0.02*float32(i%5)
		}
		return s
	}
	q = metal.FromValues(fill(0.3), 1, int(h), int(l), int(d))
	k = metal.FromValues(fill(-0.2), 1, int(h), int(l), int(d))
	v = metal.FromValues(fill(0.15), 1, int(h), int(l), int(d))
	free = func() { metal.Free(q, k, v) }
	return q, k, v, free
}

// retnetBenchState builds a non-nil prior recurrent state [1,H,Dk,Dv].
func retnetBenchState(h, d int32) *metal.Array {
	s := make([]float32, h*d*d)
	for i := range s {
		s[i] = 0.05 + 0.01*float32(i%7)
	}
	return metal.FromValues(s, 1, int(h), int(d), int(d))
}

// BenchmarkRetnet_ChunkPrefill is the parallel-form prefill: decay-masked Q Kᵀ V
// over an L=256 chunk with no incoming state. Exercises compute → advanceState
// (including the rank-3 decay-row slice).
func BenchmarkRetnet_ChunkPrefill(b *testing.B) {
	retnetBenchGate(b)
	q, k, v, free := retnetBenchQKV(retnetBenchH, retnetBenchL, retnetBenchD)
	defer free()
	decay := retnetBenchDecay(retnetBenchH)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := RetentionChunk(q, k, v, nil, decay, 0)
		if out == nil {
			b.Fatal("RetentionChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}

// BenchmarkRetnet_ChunkCarry is the cross-chunk path: an L=256 chunk fed a
// non-nil prior state, so compute also runs crossChunk + gammaPowL and the
// state-decay Mul/Add. The representative mid-sequence prefill chunk.
func BenchmarkRetnet_ChunkCarry(b *testing.B) {
	retnetBenchGate(b)
	q, k, v, free := retnetBenchQKV(retnetBenchH, retnetBenchL, retnetBenchD)
	defer free()
	prev := retnetBenchState(retnetBenchH, retnetBenchD)
	defer metal.Free(prev)
	decay := retnetBenchDecay(retnetBenchH)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := RetentionChunk(q, k, v, prev, decay, 0)
		if out == nil {
			b.Fatal("RetentionChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}

// BenchmarkRetnet_DecodeStep is the steady-state single-token decode (L==1) with
// a carried state — the per-token generation kernel and the analogue of the
// rwkv7/mamba2 sequential decode benches. The state is fed back in each
// iteration's input (a fresh prev each call is fine; the allocation profile of
// decodeStep is unaffected by whether prev is reused).
func BenchmarkRetnet_DecodeStep(b *testing.B) {
	retnetBenchGate(b)
	q, k, v, free := retnetBenchQKV(retnetBenchH, 1, retnetBenchD)
	defer free()
	prev := retnetBenchState(retnetBenchH, retnetBenchD)
	defer metal.Free(prev)
	decay := retnetBenchDecay(retnetBenchH)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := RetentionChunk(q, k, v, prev, decay, 0)
		if out == nil {
			b.Fatal("RetentionChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}
