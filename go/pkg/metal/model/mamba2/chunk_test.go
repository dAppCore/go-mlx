// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// chunkInputs builds a deterministic ScanInput of the given geometry from a
// simple index-derived pattern so the chunked and sequential kernels run on
// byte-identical data. B is fixed at 1 (the reference is B=1) and the values
// are bounded small so the float32 score matrix stays well-conditioned.
func chunkInputs(L, H, P, N int) (ScanInput, func()) {
	x := make([]float64, L*H*P)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.3) * 0.5
	}
	dt := make([]float64, L*H)
	for i := range dt {
		dt[i] = 0.2 + 0.1*math.Abs(math.Cos(float64(i)*0.7)) // softplus-positive Δ
	}
	a := make([]float64, H)
	for h := 0; h < H; h++ {
		a[h] = -0.5 - 0.3*float64(h) // negative per-head decay
	}
	bProj := make([]float64, L*H*N)
	for i := range bProj {
		bProj[i] = math.Cos(float64(i)*0.21) * 0.4
	}
	cProj := make([]float64, L*H*N)
	for i := range cProj {
		cProj[i] = math.Sin(float64(i)*0.17+0.5) * 0.4
	}
	dSkip := make([]float64, H)
	for h := 0; h < H; h++ {
		dSkip[h] = 0.1 * float64(h+1)
	}

	in := ScanInput{
		X:  metal.FromValues(float64sToFloat32(x), 1, L, H, P),
		Dt: metal.FromValues(float64sToFloat32(dt), 1, L, H),
		A:  metal.FromValues(float64sToFloat32(a), H),
		B:  metal.FromValues(float64sToFloat32(bProj), 1, L, H, N),
		C:  metal.FromValues(float64sToFloat32(cProj), 1, L, H, N),
		D:  metal.FromValues(float64sToFloat32(dSkip), H),
	}
	return in, func() { metal.Free(in.X, in.Dt, in.A, in.B, in.C, in.D) }
}

// TestChunk_SSDScanChunked_Good asserts the chunked-parallel scan produces the
// same output and advanced state as the exact sequential SSDScan on identical
// fixed inputs, run as a SINGLE parallel block (chunkSize >= L). This is the
// core gate: the parallel matmul form must equal the serial recurrence.
func TestChunk_SSDScanChunked_Good(t *testing.T) {
	const L, H, P, N = 8, 2, 3, 4

	in, free := chunkInputs(L, H, P, N)
	defer free()

	seqY, seqState := SSDScan(in, nil)
	defer metal.Free(seqY, seqState)

	// chunkSize >= L ⇒ one parallel block, exercises the diagonal + state math.
	chY, chState := SSDScanChunked(in, nil, int32(L))
	defer metal.Free(chY, chState)

	if got := chY.Shape(); len(got) != 4 || got[0] != 1 || got[1] != L || got[2] != H || got[3] != P {
		t.Fatalf("chunked y shape = %v, want [1 %d %d %d]", got, L, H, P)
	}
	if got := chState.Shape(); len(got) != 4 || got[0] != 1 || got[1] != H || got[2] != P || got[3] != N {
		t.Fatalf("chunked state shape = %v, want [1 %d %d %d]", got, H, P, N)
	}

	assertCloseF32(t, "y chunked-vs-sequential", chY.Floats(), seqY.Floats(), 2e-3)
	assertCloseF32(t, "state chunked-vs-sequential", chState.Floats(), seqState.Floats(), 2e-3)
}

// TestChunk_SSDScanChunked_Bad drives the inter-chunk recurrence: a chunkSize
// SMALLER than L forces the sequence to be split into multiple segments whose
// states must chain (S_entry(c+1) = exp(L_C)·S_entry(c) + S_c). The multi-chunk
// result must still equal the single serial scan.
func TestChunk_SSDScanChunked_Bad(t *testing.T) {
	const L, H, P, N = 10, 2, 2, 3

	in, free := chunkInputs(L, H, P, N)
	defer free()

	seqY, seqState := SSDScan(in, nil)
	defer metal.Free(seqY, seqState)

	// chunkSize = 3 over L = 10 ⇒ chunks of 3,3,3,1 — every inter-chunk carry
	// and the ragged final chunk are exercised.
	chY, chState := SSDScanChunked(in, nil, 3)
	defer metal.Free(chY, chState)

	assertCloseF32(t, "multi-chunk y", chY.Floats(), seqY.Floats(), 2e-3)
	assertCloseF32(t, "multi-chunk state", chState.Floats(), seqState.Floats(), 2e-3)
}

// TestChunk_SSDScanChunked_Ugly carries a non-zero prior state into the chunked
// scan (the decode-continuation path) and checks it matches the sequential scan
// fed the same prior — the Y_off carried-state read-out and the exp(L_C)-decayed
// entry-state in advanceState.
func TestChunk_SSDScanChunked_Ugly(t *testing.T) {
	const L, H, P, N = 6, 1, 2, 2

	in, free := chunkInputs(L, H, P, N)
	defer free()

	priorVals := []float32{0.7, -0.2, 0.4, 0.9} // [1,H,P,N] = [1,1,2,2]
	priorSeq := metal.FromValues(priorVals, 1, H, P, N)
	priorCh := metal.FromValues(priorVals, 1, H, P, N)
	defer metal.Free(priorSeq, priorCh)

	seqY, seqState := SSDScan(in, priorSeq)
	defer metal.Free(seqY, seqState)

	chY, chState := SSDScanChunked(in, priorCh, 4) // chunks 4,2 — carry + prior
	defer metal.Free(chY, chState)

	assertCloseF32(t, "prior-carry y", chY.Floats(), seqY.Floats(), 2e-3)
	assertCloseF32(t, "prior-carry state", chState.Floats(), seqState.Floats(), 2e-3)
}

// assertCloseF32 compares two float32 slices elementwise within tol. Unlike
// assertClose (which takes a float64 reference) this checks the chunked kernel
// against the sequential kernel's own float32 output — both run on MLX, so the
// tolerance absorbs only the reordered-float-op rounding, not a model gap.
func assertCloseF32(t *testing.T, label string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("%s[%d] = %v, want %v (diff %v > tol %v)", label, i, got[i], want[i], diff, tol)
		}
	}
}

// BenchmarkSSDScan_Sequential and BenchmarkSSDScan_Chunked measure the prefill
// speedup at a representative geometry. Bounded with -benchtime=1x in CI; the
// chunked form's win grows with L because it replaces the L serial graph nodes
// with O(L/chunk) chunk steps of batched matmuls.
func BenchmarkSSDScan_Sequential(b *testing.B) {
	const L, H, P, N = 128, 4, 8, 16
	in, free := chunkInputs(L, H, P, N)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y, st := SSDScan(in, nil)
		metal.Free(y, st)
	}
}

func BenchmarkSSDScan_Chunked(b *testing.B) {
	const L, H, P, N = 128, 4, 8, 16
	in, free := chunkInputs(L, H, P, N)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y, st := SSDScanChunked(in, nil, 0)
		metal.Free(y, st)
	}
}
