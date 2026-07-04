// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// chunkStepInputs builds a deterministic StepInput of the given geometry from an
// index-derived pattern so the chunked and sequential kernels run on
// byte-identical data. B is fixed at 1. W is forced ≤ 0 (RWKV-7's W is a
// log-decay), and a/b/k/v/r are bounded small so the float32 attention matrices
// and the triangular inverse stay well-conditioned.
func chunkStepInputs(L, H, K, Vd int) (StepInput, func()) {
	r := make([]float64, L*H*K)
	for i := range r {
		r[i] = math.Sin(float64(i)*0.31) * 0.5
	}
	w := make([]float64, L*H*K)
	for i := range w {
		w[i] = -0.05 - 0.15*math.Abs(math.Cos(float64(i)*0.43)) // log-decay ≤ 0
	}
	k := make([]float64, L*H*K)
	for i := range k {
		k[i] = math.Cos(float64(i)*0.27) * 0.4
	}
	v := make([]float64, L*H*Vd)
	for i := range v {
		v[i] = math.Sin(float64(i)*0.19+0.3) * 0.5
	}
	a := make([]float64, L*H*K)
	for i := range a {
		a[i] = math.Sin(float64(i)*0.11) * 0.2 // in-context learning-rate vectors
	}
	bb := make([]float64, L*H*K)
	for i := range bb {
		bb[i] = math.Cos(float64(i)*0.13+0.7) * 0.2
	}

	in := StepInput{
		R: metal.FromValues(f32(r), 1, L, H, K),
		W: metal.FromValues(f32(w), 1, L, H, K),
		K: metal.FromValues(f32(k), 1, L, H, K),
		V: metal.FromValues(f32(v), 1, L, H, Vd),
		A: metal.FromValues(f32(a), 1, L, H, K),
		B: metal.FromValues(f32(bb), 1, L, H, K),
	}
	return in, func() { metal.Free(in.R, in.W, in.K, in.V, in.A, in.B) }
}

// TestChunk_WKV7Chunked_Good asserts the chunked-parallel DPLR scan produces the
// same output and advanced state as the exact sequential WKV7 on identical
// fixed inputs, run as a SINGLE window (L <= chunkWidth). This is the core gate:
// the parallel matmul + triangular-inverse form must equal the serial
// recurrence to machine precision.
func TestChunk_WKV7Chunked_Good(t *testing.T) {
	const L, H, K, Vd = 8, 2, 3, 4

	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	seqO, seqState := WKV7(in, nil)
	defer metal.Free(seqO, seqState)

	chO, chState := WKV7Chunked(in, nil)
	if chO == nil {
		t.Fatal("WKV7Chunked returned nil (triangular inverse failed)")
	}
	defer metal.Free(chO, chState)

	if got := chO.Shape(); len(got) != 4 || got[0] != 1 || got[1] != L || got[2] != H || got[3] != Vd {
		t.Fatalf("chunked o shape = %v, want [1 %d %d %d]", got, L, H, Vd)
	}
	if got := chState.Shape(); len(got) != 4 || got[0] != 1 || got[1] != H || got[2] != K || got[3] != Vd {
		t.Fatalf("chunked state shape = %v, want [1 %d %d %d]", got, H, K, Vd)
	}

	assertCloseF32(t, "o chunked-vs-sequential", chO.Floats(), seqO.Floats(), 1e-5)
	assertCloseF32(t, "state chunked-vs-sequential", chState.Floats(), seqState.Floats(), 1e-5)
}

// TestChunk_WKV7Chunked_Bad drives the inter-window recurrence by forcing a
// window width smaller than L (via wkv7ChunkedWindowed), so the sequence splits
// into multiple windows whose recurrent states must chain. The multi-window
// result must still equal the single serial scan.
func TestChunk_WKV7Chunked_Bad(t *testing.T) {
	// Use a sequence longer than a deliberately small window so the inter-window
	// state carry (v2 = u + wS, the exp(c_C)-decayed entry state) is exercised.
	const L, H, K, Vd = 10, 2, 2, 3

	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	seqO, seqState := WKV7(in, nil)
	defer metal.Free(seqO, seqState)

	// windowed at 3 over L=10 ⇒ windows 3,3,3,1 — every carry + the ragged
	// final window.
	chO, chState := wkv7ChunkedWindowed(in, nil, 3)
	if chO == nil {
		t.Fatal("WKV7Chunked (windowed) returned nil")
	}
	defer metal.Free(chO, chState)

	assertCloseF32(t, "multi-window o", chO.Floats(), seqO.Floats(), 1e-5)
	assertCloseF32(t, "multi-window state", chState.Floats(), seqState.Floats(), 1e-5)
}

// TestChunk_WKV7Chunked_Ugly carries a non-zero prior state into the chunked
// scan (the decode-continuation path) and checks it matches the sequential scan
// fed the same prior — the q̃ S carried read-out, the w S correction in v2, and
// the exp(c_C)-decayed entry state in the state update.
func TestChunk_WKV7Chunked_Ugly(t *testing.T) {
	const L, H, K, Vd = 6, 1, 2, 2

	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	priorVals := []float32{0.7, -0.2, 0.4, 0.9} // [1,H,K,V] = [1,1,2,2]
	priorSeq := metal.FromValues(priorVals, 1, H, K, Vd)
	priorCh := metal.FromValues(priorVals, 1, H, K, Vd)
	defer metal.Free(priorSeq, priorCh)

	seqO, seqState := WKV7(in, priorSeq)
	defer metal.Free(seqO, seqState)

	chO, chState := wkv7ChunkedWindowed(in, priorCh, 4) // windows 4,2 — carry + prior
	if chO == nil {
		t.Fatal("WKV7Chunked (windowed, prior) returned nil")
	}
	defer metal.Free(chO, chState)

	assertCloseF32(t, "prior-carry o", chO.Floats(), seqO.Floats(), 1e-5)
	assertCloseF32(t, "prior-carry state", chState.Floats(), seqState.Floats(), 1e-5)
}

// assertCloseF32 compares two float32 slices elementwise within tol — the
// chunked kernel against the sequential kernel's own float32 output (both run on
// MLX, so the tolerance absorbs only reordered-float-op rounding plus the f32
// triangular-inverse round-trip).
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

// BenchmarkWKV7_Sequential and BenchmarkWKV7_Chunked measure the prefill
// speedup at a representative geometry. Bounded with -benchtime in CI; the
// chunked form's win grows with L because it replaces the L serial graph nodes
// with O(L/chunkWidth) windows of batched matmuls + one triangular inverse.
func BenchmarkWKV7_Sequential(b *testing.B) {
	const L, H, K, Vd = 128, 4, 8, 16
	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, st := WKV7(in, nil)
		metal.Free(o, st)
	}
}

func BenchmarkWKV7_Chunked(b *testing.B) {
	const L, H, K, Vd = 128, 4, 8, 16
	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, st := WKV7Chunked(in, nil)
		metal.Free(o, st)
	}
}
