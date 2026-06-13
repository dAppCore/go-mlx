// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// wkv7Reference is an independent, pure-Go float64 implementation of the RWKV-7
// recurrence for B=1 — the ground truth the MLX kernel is checked against.
// State is [H][K][V]; it mirrors the documented per-step math exactly (Sa
// contraction, diagonal decay, the two rank-1 updates, the receptance
// read-out) with no MLX involvement, so a divergence pins a bug in the
// metal.Array composition, not in a shared helper.
func wkv7Reference(L, H, K, Vd int, r, w, k, v, a, b []float64) (o []float64, state [][][]float64) {
	state = make([][][]float64, H)
	for h := 0; h < H; h++ {
		state[h] = make([][]float64, K)
		for kk := 0; kk < K; kk++ {
			state[h][kk] = make([]float64, Vd)
		}
	}
	o = make([]float64, L*H*Vd)
	kAt := func(arr []float64, t, h, c int) float64 { return arr[(t*H+h)*K+c] }
	vAt := func(t, h, c int) float64 { return v[(t*H+h)*Vd+c] }
	for t := 0; t < L; t++ {
		for h := 0; h < H; h++ {
			// Sa[vd] = sum_k a[k] * S[k][vd]
			sa := make([]float64, Vd)
			for vd := 0; vd < Vd; vd++ {
				for kk := 0; kk < K; kk++ {
					sa[vd] += kAt(a, t, h, kk) * state[h][kk][vd]
				}
			}
			for kk := 0; kk < K; kk++ {
				ew := math.Exp(kAt(w, t, h, kk))
				for vd := 0; vd < Vd; vd++ {
					state[h][kk][vd] = ew*state[h][kk][vd] +
						kAt(b, t, h, kk)*sa[vd] +
						kAt(k, t, h, kk)*vAt(t, h, vd)
				}
			}
			// o[vd] = sum_k S[k][vd] * r[k]
			for vd := 0; vd < Vd; vd++ {
				var acc float64
				for kk := 0; kk < K; kk++ {
					acc += state[h][kk][vd] * kAt(r, t, h, kk)
				}
				o[(t*H+h)*Vd+vd] = acc
			}
		}
	}
	return o, state
}

func f32(in []float64) []float32 {
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

// TestRecurrence_WKV7_Good runs the kernel on a small fixed B=1,L=2,H=1,K=2,V=2
// chunk from a zero state (prefill) and checks both the output and the advanced
// state against the independent pure-Go reference, plus one fully hand-computed
// anchor so the reference itself can't drift unnoticed.
func TestRecurrence_WKV7_Good(t *testing.T) {
	const B, L, H, K, Vd = 1, 2, 1, 2, 2
	r := []float64{ // [L,H,K]
		1.0, 0.5,
		0.0, 1.0,
	}
	w := []float64{ // [L,H,K] log-decay (≤ 0)
		-0.1, -0.2,
		-0.3, -0.4,
	}
	k := []float64{ // [L,H,K]
		1.0, 0.0,
		0.5, 0.5,
	}
	v := []float64{ // [L,H,V]
		2.0, 1.0,
		-1.0, 3.0,
	}
	a := []float64{ // [L,H,K]
		0.2, 0.1,
		-0.1, 0.3,
	}
	bb := []float64{ // [L,H,K]
		0.5, -0.5,
		0.25, 0.25,
	}

	wantO, wantState := wkv7Reference(L, H, K, Vd, r, w, k, v, a, bb)

	// Hand anchor — t=0 from a zero state: Sa = 0, so S1 = k⊗v and
	//   o0[vd] = (sum_k r[k]·k[k]) · v[vd].
	//   r·k (t=0) = 1*1 + 0.5*0 = 1.0 → o0 = [1*2, 1*1] = [2, 1].
	if math.Abs(wantO[0]-2.0) > 1e-9 || math.Abs(wantO[1]-1.0) > 1e-9 {
		t.Fatalf("reference self-check failed: o[0:2]=%v want [2 1] — fix the reference, not the kernel", wantO[0:2])
	}

	in := StepInput{
		R: metal.FromValues(f32(r), B, L, H, K),
		W: metal.FromValues(f32(w), B, L, H, K),
		K: metal.FromValues(f32(k), B, L, H, K),
		V: metal.FromValues(f32(v), B, L, H, Vd),
		A: metal.FromValues(f32(a), B, L, H, K),
		B: metal.FromValues(f32(bb), B, L, H, K),
	}
	defer metal.Free(in.R, in.W, in.K, in.V, in.A, in.B)

	o, state := WKV7(in, nil)
	defer metal.Free(o, state)

	if got := o.Shape(); len(got) != 4 || got[0] != B || got[1] != L || got[2] != H || got[3] != Vd {
		t.Fatalf("o shape = %v, want [%d %d %d %d]", got, B, L, H, Vd)
	}
	if got := state.Shape(); len(got) != 4 || got[0] != B || got[1] != H || got[2] != K || got[3] != Vd {
		t.Fatalf("state shape = %v, want [%d %d %d %d]", got, B, H, K, Vd)
	}

	assertClose(t, "o", o.Floats(), wantO, 1e-4)

	flatState := make([]float64, H*K*Vd)
	for h := 0; h < H; h++ {
		for kk := 0; kk < K; kk++ {
			for vd := 0; vd < Vd; vd++ {
				flatState[(h*K+kk)*Vd+vd] = wantState[h][kk][vd]
			}
		}
	}
	assertClose(t, "state", state.Floats(), flatState, 1e-4)
}

// TestRecurrence_WKV7_Bad feeds a decode step (L=1) carrying a non-zero prior
// state and checks the kernel consumes the prior decay + the rank-1 transition
// correctly — the path that exercises the carried recurrent state rather than
// the prefill zero-init.
func TestRecurrence_WKV7_Bad(t *testing.T) {
	const B, L, H, K, Vd = 1, 1, 1, 2, 2
	// Prior state [B,H,K,V] = [[1, 2],[3, 4]] for the single head.
	priorVals := []float32{1, 2, 3, 4}
	prior := metal.FromValues(priorVals, B, H, K, Vd)
	defer metal.Free(prior)

	r := []float64{1.0, 1.0}
	w := []float64{-0.5, -1.0}
	k := []float64{1.0, 0.0}
	v := []float64{1.0, 2.0}
	a := []float64{0.5, 0.5}
	bb := []float64{1.0, 1.0}

	// Reference, pre-loading the prior into a single-head [K][V] state.
	state := [][]float64{{1, 2}, {3, 4}}
	sa := make([]float64, Vd)
	for vd := 0; vd < Vd; vd++ {
		for kk := 0; kk < K; kk++ {
			sa[vd] += a[kk] * state[kk][vd]
		}
	}
	for kk := 0; kk < K; kk++ {
		ew := math.Exp(w[kk])
		for vd := 0; vd < Vd; vd++ {
			state[kk][vd] = ew*state[kk][vd] + bb[kk]*sa[vd] + k[kk]*v[vd]
		}
	}
	wantO := make([]float64, Vd)
	for vd := 0; vd < Vd; vd++ {
		for kk := 0; kk < K; kk++ {
			wantO[vd] += state[kk][vd] * r[kk]
		}
	}
	flatState := []float64{state[0][0], state[0][1], state[1][0], state[1][1]}

	in := StepInput{
		R: metal.FromValues(f32(r), B, L, H, K),
		W: metal.FromValues(f32(w), B, L, H, K),
		K: metal.FromValues(f32(k), B, L, H, K),
		V: metal.FromValues(f32(v), B, L, H, Vd),
		A: metal.FromValues(f32(a), B, L, H, K),
		B: metal.FromValues(f32(bb), B, L, H, K),
	}
	defer metal.Free(in.R, in.W, in.K, in.V, in.A, in.B)

	o, st := WKV7(in, prior)
	defer metal.Free(o, st)

	assertClose(t, "decode-state", st.Floats(), flatState, 1e-4)
	assertClose(t, "decode-o", o.Floats(), wantO, 1e-4)
}

// TestRecurrence_Mixer_Ugly asserts the scheme scaffold is wired: the mixer
// declares the rwkv7 kind + recurrent state, registers into the catalogue, and
// resolves back through scheme.MixerFor / metal.MixerComputeFor. This is the
// contract surface the decoder loop uses, independent of the recurrence math.
func TestRecurrence_Mixer_Ugly(t *testing.T) {
	m := &Mixer{}
	if m.Kind() != "rwkv7" {
		t.Errorf("Kind() = %q, want %q", m.Kind(), "rwkv7")
	}
	if m.State() != scheme.StateRecurrent {
		t.Errorf("State() = %v, want %v", m.State(), scheme.StateRecurrent)
	}
	// The weightless family seed (register.go's familyInfo) resolves with the
	// right state but carries no compute; the compute surface attaches when the
	// loader registers a built *Mixer — asserted in TestMixer_Ugly, which owns
	// that global-registry mutation so the two tests don't race the catalogue.
	resolved, ok := scheme.MixerFor("rwkv7")
	if !ok {
		t.Fatal("scheme.MixerFor(\"rwkv7\") not registered")
	}
	if resolved.State() != scheme.StateRecurrent {
		t.Errorf("resolved State() = %v, want %v", resolved.State(), scheme.StateRecurrent)
	}
}
