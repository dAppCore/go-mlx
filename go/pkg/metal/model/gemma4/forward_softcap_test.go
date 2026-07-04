// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestForward_logitSoftcap_Good covers the logit soft-cap helper, which squashes
// each logit through softcap*tanh(x/softcap). The closed form is checked at
// representative points: 0 maps to 0, a value far above the cap saturates toward
// +softcap, and a value far below saturates toward -softcap. A NEW Test (the
// existing logit_softcap_bench_test.go is a Benchmark, which contributes no
// statement coverage without -bench).
func TestForward_logitSoftcap_Good(t *testing.T) {
	requireMetalRuntime(t)

	const softcap float32 = 30.0
	in := []float32{0, 15, -15, 300, -300}
	x := metal.FromValues(in, 1, len(in))
	metal.Materialize(x)
	defer metal.Free(x)

	out := logitSoftcap(x, softcap)
	if out == nil || !out.Valid() {
		t.Fatal("logitSoftcap returned invalid array")
	}
	defer metal.Free(out)
	metal.Materialize(out)

	got := out.Floats()
	if len(got) != len(in) {
		t.Fatalf("len(out) = %d, want %d", len(got), len(in))
	}
	for i, v := range in {
		want := softcap * float32(math.Tanh(float64(v/softcap)))
		if math.Abs(float64(got[i]-want)) > 1e-2 {
			t.Fatalf("logitSoftcap[%d](%g) = %g, want %g", i, v, got[i], want)
		}
	}
	// Saturation sanity: large-magnitude inputs are pinned inside the cap and
	// never beyond it. At 10x the cap, float32 tanh saturates to exactly 1, so
	// the bound is inclusive of +/-softcap; the sign must still match the input.
	if got[3] <= 0 || got[3] > softcap {
		t.Fatalf("saturated positive = %g, want 0 < v <= %g", got[3], softcap)
	}
	if got[4] >= 0 || got[4] < -softcap {
		t.Fatalf("saturated negative = %g, want -%g <= v < 0", got[4], softcap)
	}
}
