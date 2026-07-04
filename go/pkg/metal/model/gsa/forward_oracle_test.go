// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_oracle_test.go adds ONE Forward-level ABSOLUTE-VALUE test for
// gsa.Mixer.Forward against a pure-Go scalar slot-recurrence reference that
// INCLUDES the SiLU output gate and the output projection. The recurrence kernel
// is already pinned independently (gsa_test.go), and TestForward_Chunked... is a
// consistency check (single-pass == chunked) that is invariant to wiring/gate
// bugs which cancel across both calls. This test fixes the absolute output, so a
// dropped/transposed gate, a wrong gate operand, or an OProj slip is caught.

const oracleTol = 1e-4

// TestForward_AbsoluteValueWithGate_Good pins the whole Forward path on a fixed
// input. With the all-identity mixer (QProj=KProj=VProj=FProj=GProj=OProj=I,
// 1 head, HeadK=HeadV=Slots=2, GateNorm=1) every projection passes x through, so
// q=k=v=f=gate_logits=x and the math is determined solely by the gated slot
// recurrence + SiLU gate. The oracle recomputes that recurrence and gate in pure
// Go and asserts the exact two-token output.
func TestForward_AbsoluteValueWithGate_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := gsaIdentityMixer()
	defer freeMixer(m)

	// Two tokens of hidden state [B=1, L=2, hidden=2].
	rows := [][]float32{{0.5, -0.3}, {0.1, 0.7}}
	x := metal.FromValues(append(append([]float32{}, rows[0]...), rows[1]...), 1, 2, 2)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	got := out.Floats()
	metal.Free(out)
	rc.Reset()

	want := gsaForwardOracle(rows)
	if len(got) != len(want) {
		t.Fatalf("output len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > oracleTol {
			t.Errorf("Forward[%d] = %f, want %f (diff %g)", i, got[i], want[i], diff)
		}
	}
}

// gsaForwardOracle reproduces the full GSA Forward in pure Go for the
// all-identity, single-head, HeadK=HeadV=Slots=2, GateNorm=1 configuration:
// q=k=v=f=gate=x. Per timestep it runs the gated slot recurrence
//
//	g    = logsigmoid(f) = log(sigmoid(x))
//	s    = 1 - exp(g)
//	Sk  := Sk * exp(g)[broadcast over HeadK] + (k ⊗ s)   [HeadK x Slots]
//	Sv  := Sv * exp(g)[broadcast over HeadV] + (s ⊗ v)   [Slots x HeadV]
//	a    = softmax_slots(q · Sk)
//	o    = a · Sv
//
// then applies the SiLU output gate (gated = o ⊙ SiLU(x)) and the identity OProj.
// Returns the flat [L*HeadV] output in row order.
func gsaForwardOracle(rows [][]float32) []float32 {
	const headK, headV, slots = 2, 2, 2
	L := len(rows)

	// Slot memory: Sk[dk][slot], Sv[slot][dv], zero-initialised.
	var sk [headK][slots]float64
	var sv [slots][headV]float64

	out := make([]float32, L*headV)
	for t := 0; t < L; t++ {
		x := make([]float64, len(rows[t]))
		for j := range rows[t] {
			x[j] = float64(rows[t][j])
		}
		// All identity projections: q=k=v=f=gate_logits=x.
		q, k, v, f := x, x, x, x

		// Gate decay g = log(sigmoid(f)); slot-write s = 1 - exp(g); eg = exp(g).
		var g, s, eg [slots]float64
		for sl := 0; sl < slots; sl++ {
			sig := 1.0 / (1.0 + math.Exp(-f[sl]))
			g[sl] = math.Log(sig)
			eg[sl] = math.Exp(g[sl])
			s[sl] = 1 - eg[sl]
		}

		// Sk = Sk*eg + k ⊗ s  (eg broadcasts over the HeadK axis).
		for dk := 0; dk < headK; dk++ {
			for sl := 0; sl < slots; sl++ {
				sk[dk][sl] = sk[dk][sl]*eg[sl] + k[dk]*s[sl]
			}
		}
		// Sv = Sv*eg + s ⊗ v  (eg broadcasts over the HeadV axis; row = slot).
		for sl := 0; sl < slots; sl++ {
			for dv := 0; dv < headV; dv++ {
				sv[sl][dv] = sv[sl][dv]*eg[sl] + s[sl]*v[dv]
			}
		}

		// scores = q · Sk over slots; a = softmax_slots(scores).
		var scores [slots]float64
		for sl := 0; sl < slots; sl++ {
			var acc float64
			for dk := 0; dk < headK; dk++ {
				acc += q[dk] * sk[dk][sl]
			}
			scores[sl] = acc
		}
		maxS := math.Inf(-1)
		for _, sc := range scores {
			if sc > maxS {
				maxS = sc
			}
		}
		var sum float64
		var a [slots]float64
		for sl := 0; sl < slots; sl++ {
			a[sl] = math.Exp(scores[sl] - maxS)
			sum += a[sl]
		}
		for sl := 0; sl < slots; sl++ {
			a[sl] /= sum
		}

		// o = a · Sv over slots → [HeadV].
		var o [headV]float64
		for dv := 0; dv < headV; dv++ {
			var acc float64
			for sl := 0; sl < slots; sl++ {
				acc += a[sl] * sv[sl][dv]
			}
			o[dv] = acc
		}

		// Output gate: gated = o ⊙ SiLU(gate_logits), gate_logits = x. SiLU(z) =
		// z·sigmoid(z). OProj = identity → output is gated directly.
		for dv := 0; dv < headV; dv++ {
			z := x[dv]
			silu := z / (1.0 + math.Exp(-z))
			out[t*headV+dv] = float32(o[dv] * silu)
		}
	}
	return out
}
