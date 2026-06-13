// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// scanReference is an independent, pure-Go float64 implementation of the
// Mamba-2 SSD recurrence for B=1 — the ground truth the MLX kernel is checked
// against. State is [H][P][N]; it mirrors the documented per-step math exactly
// (decay, outer-product input, C contraction, D skip) with no MLX involvement,
// so a divergence pins a bug in the metal.Array composition, not in a shared
// helper.
func scanReference(L, H, P, N int, x, dt, a, b, c, d []float64) (y []float64, state [][][]float64) {
	state = make([][][]float64, H)
	for h := 0; h < H; h++ {
		state[h] = make([][]float64, P)
		for p := 0; p < P; p++ {
			state[h][p] = make([]float64, N)
		}
	}
	y = make([]float64, L*H*P)
	xAt := func(t, h, p int) float64 { return x[(t*H+h)*P+p] }
	bcAt := func(arr []float64, t, h, n int) float64 { return arr[(t*H+h)*N+n] }
	for t := 0; t < L; t++ {
		for h := 0; h < H; h++ {
			dtt := dt[t*H+h]
			dA := math.Exp(dtt * a[h])
			for p := 0; p < P; p++ {
				for n := 0; n < N; n++ {
					state[h][p][n] = state[h][p][n]*dA + xAt(t, h, p)*(dtt*bcAt(b, t, h, n))
				}
			}
			for p := 0; p < P; p++ {
				var acc float64
				for n := 0; n < N; n++ {
					acc += state[h][p][n] * bcAt(c, t, h, n)
				}
				if len(d) == H {
					acc += d[h] * xAt(t, h, p)
				}
				y[(t*H+h)*P+p] = acc
			}
		}
	}
	return y, state
}

func float64sToFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func assertClose(t *testing.T, label string, got []float32, want []float64, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}
	for i := range got {
		diff := got[i] - float32(want[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("%s[%d] = %v, want %v (diff %v > tol %v)", label, i, got[i], want[i], diff, tol)
		}
	}
}

// TestScan_SSDScan_Good runs the kernel on a small fixed B=1,L=2,H=1,P=2,N=2
// chunk from a zero state (prefill) and checks both the output and the advanced
// state against the independent pure-Go reference, plus one fully hand-computed
// anchor value so the reference itself can't drift unnoticed.
func TestScan_SSDScan_Good(t *testing.T) {
	const B, L, H, P, N = 1, 2, 1, 2, 2
	x := []float64{ // [L,H,P]
		1.0, 2.0, // t=0
		0.5, -1.0, // t=1
	}
	dt := []float64{0.5, 1.0} // [L,H]
	a := []float64{-1.0}      // [H] (negative decay)
	bProj := []float64{ // [L,H,N]
		1.0, 0.0, // t=0
		0.0, 1.0, // t=1
	}
	cProj := []float64{ // [L,H,N]
		1.0, 1.0, // t=0
		1.0, 1.0, // t=1
	}
	dSkip := []float64{0.25} // [H]

	wantY, wantState := scanReference(L, H, P, N, x, dt, a, bProj, cProj, dSkip)

	// Hand anchor — t=0 only, channel p=0:
	//   dA0 = exp(0.5 * -1) = 0.60653...
	//   state[0][0] = [ x0p0 * dt0 * b0n0, x0p0 * dt0 * b0n1 ] = [1*0.5*1, 1*0.5*0] = [0.5, 0]
	//   y0p0 = state·C0 + D*x0p0 = (0.5*1 + 0*1) + 0.25*1 = 0.75
	const wantY0P0 = 0.75
	if math.Abs(wantY[0]-wantY0P0) > 1e-9 {
		t.Fatalf("reference self-check failed: y[0]=%v want %v — fix the reference, not the kernel", wantY[0], wantY0P0)
	}

	in := ScanInput{
		X:  metal.FromValues(float64sToFloat32(x), B, L, H, P),
		Dt: metal.FromValues(float64sToFloat32(dt), B, L, H),
		A:  metal.FromValues(float64sToFloat32(a), H),
		B:  metal.FromValues(float64sToFloat32(bProj), B, L, H, N),
		C:  metal.FromValues(float64sToFloat32(cProj), B, L, H, N),
		D:  metal.FromValues(float64sToFloat32(dSkip), H),
	}
	defer metal.Free(in.X, in.Dt, in.A, in.B, in.C, in.D)

	y, state := SSDScan(in, nil)
	defer metal.Free(y, state)

	if got := y.Shape(); len(got) != 4 || got[0] != B || got[1] != L || got[2] != H || got[3] != P {
		t.Fatalf("y shape = %v, want [%d %d %d %d]", got, B, L, H, P)
	}
	if got := state.Shape(); len(got) != 4 || got[0] != B || got[1] != H || got[2] != P || got[3] != N {
		t.Fatalf("state shape = %v, want [%d %d %d %d]", got, B, H, P, N)
	}

	assertClose(t, "y", y.Floats(), wantY, 1e-4)

	// Flatten the reference state [H][P][N] to compare with the kernel's [B,H,P,N].
	flatState := make([]float64, H*P*N)
	for h := 0; h < H; h++ {
		for p := 0; p < P; p++ {
			for n := 0; n < N; n++ {
				flatState[(h*P+p)*N+n] = wantState[h][p][n]
			}
		}
	}
	assertClose(t, "state", state.Floats(), flatState, 1e-4)
}

// TestScan_SSDScan_Bad feeds a decode step (L=1) carrying a non-zero prior
// state and checks the kernel both consumes the prior decay correctly and
// returns a state of the right shape — the path that exercises the carried
// recurrent state rather than the prefill zero-init.
func TestScan_SSDScan_Bad(t *testing.T) {
	const B, L, H, P, N = 1, 1, 1, 2, 2
	// Prior state [B,H,P,N] = [[1, 2],[3, 4]] for the single head.
	priorVals := []float32{1, 2, 3, 4}
	prior := metal.FromValues(priorVals, B, H, P, N)
	defer metal.Free(prior)

	x := []float64{1.0, 1.0} // [L,H,P]
	dt := []float64{1.0}
	a := []float64{-0.5}
	bProj := []float64{1.0, 1.0} // [L,H,N]
	cProj := []float64{1.0, 0.0} // [L,H,N] — reads only the first state column

	// Reference with the same prior pre-loaded.
	dA := math.Exp(dt[0] * a[0])
	wantState := [][]float64{
		{1*dA + x[0]*dt[0]*bProj[0], 2*dA + x[0]*dt[0]*bProj[1]},
		{3*dA + x[1]*dt[0]*bProj[0], 4*dA + x[1]*dt[0]*bProj[1]},
	}
	wantY := []float64{
		wantState[0][0]*cProj[0] + wantState[0][1]*cProj[1],
		wantState[1][0]*cProj[0] + wantState[1][1]*cProj[1],
	}

	in := ScanInput{
		X:  metal.FromValues(float64sToFloat32(x), B, L, H, P),
		Dt: metal.FromValues(float64sToFloat32(dt), B, L, H),
		A:  metal.FromValues(float64sToFloat32(a), H),
		B:  metal.FromValues(float64sToFloat32(bProj), B, L, H, N),
		C:  metal.FromValues(float64sToFloat32(cProj), B, L, H, N),
	}
	defer metal.Free(in.X, in.Dt, in.A, in.B, in.C)

	y, state := SSDScan(in, prior)
	defer metal.Free(y, state)

	flatState := []float64{wantState[0][0], wantState[0][1], wantState[1][0], wantState[1][1]}
	assertClose(t, "decode-state", state.Floats(), flatState, 1e-4)
	assertClose(t, "decode-y", y.Floats(), wantY, 1e-4)
}

// TestScan_Mixer_Ugly asserts the scheme scaffold is wired: the mixer declares
// the mamba2 kind + recurrent state, registers into the catalogue, and resolves
// back through scheme.MixerFor / metal.MixerComputeFor. This is the contract
// surface the decoder loop uses, independent of the scan math.
func TestScan_Mixer_Ugly(t *testing.T) {
	m := &Mixer{}
	if m.Kind() != "mamba2" {
		t.Errorf("Kind() = %q, want %q", m.Kind(), "mamba2")
	}
	if m.State() != scheme.StateRecurrent {
		t.Errorf("State() = %v, want %v", m.State(), scheme.StateRecurrent)
	}
	// register.go's init registered the weightless family seed; it resolves with
	// the right state kind. (The compute surface attaches when the loader
	// registers a built *Mixer — asserted in TestMixer_Ugly, which owns that
	// global-registry mutation so the two tests don't race the catalogue.)
	resolved, ok := scheme.MixerFor("mamba2")
	if !ok {
		t.Fatal("scheme.MixerFor(\"mamba2\") not registered")
	}
	if resolved.State() != scheme.StateRecurrent {
		t.Errorf("resolved State() = %v, want %v", resolved.State(), scheme.StateRecurrent)
	}
}
