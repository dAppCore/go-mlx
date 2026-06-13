// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// requireMetalRuntime skips the body unless the suite was built with
// -tags metal_runtime on a Metal-capable host. The test file itself stays
// un-tagged so it always compiles (catching kernel signature regressions);
// only the compute body needs the live runtime. Mirrors the gate every model
// package uses (qwen3/qwen3_test.go).
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

const mlaTol = 1e-4

// causalMask2 builds the additive [1,1,2,2] causal mask used by the kernel
// math test: position 0 sees only key 0; position 1 sees keys 0 and 1.
func causalMask2() *metal.Array {
	negInf := float32(math.Inf(-1))
	return metal.FromValues([]float32{
		0, negInf,
		0, 0,
	}, 1, 1, 2, 2)
}

// TestMixer_KindState_Good pins the scheme identity: MLA answers to "mla" and
// declares a growing KV-cache (of the compressed latent). This is the contract
// the registry and scheme.Compatible rely on; it needs no Metal runtime.
func TestMixer_KindState_Good(t *testing.T) {
	if MixerKind != "mla" {
		t.Fatalf("MixerKind = %q, want %q", MixerKind, "mla")
	}
	m := &Mixer{}
	if got := m.Kind(); got != MixerKind {
		t.Fatalf("Kind() = %q, want %q", got, MixerKind)
	}
	if got := m.State(); got != scheme.StateKVCache {
		t.Fatalf("State() = %v, want StateKVCache", got)
	}
}

// TestMixer_Register_Good proves the init() side-effect registers a
// compute-bearing mixer that scheme.MixerFor resolves and metal.MixerComputeFor
// can assert the Forward surface on.
func TestMixer_Register_Good(t *testing.T) {
	m, ok := scheme.MixerFor(MixerKind)
	if !ok {
		t.Fatalf("scheme.MixerFor(%q) not registered", MixerKind)
	}
	if m.State() != scheme.StateKVCache {
		t.Fatalf("registered mixer State() = %v, want StateKVCache", m.State())
	}
	if _, ok := metal.MixerComputeFor(MixerKind); !ok {
		t.Fatalf("metal.MixerComputeFor(%q) = false, want a compute-bearing mixer", MixerKind)
	}
}

// TestAttendLatent_Math_Good pins the MLA attention kernel against a
// hand-derived reference. B=1,H=1,L=2,D=2, identity Q/K, V=[[1,2],[3,4]],
// scale=1, causal mask. Position 0 attends only to itself → V0=[1,2].
// Position 1 softmaxes scores [0,1] → weights [1/(1+e), e/(1+e)] over V.
func TestAttendLatent_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	mask := causalMask2()
	defer metal.Free(q, k, v, mask)

	out := attendLatent(q, k, v, mask, 1.0)
	defer metal.Free(out)

	got := out.Floats()
	w0 := 1.0 / (1.0 + math.E)
	w1 := math.E / (1.0 + math.E)
	want := []float32{
		1, 2, // position 0: only attends to V0
		float32(w0*1 + w1*3), float32(w0*2 + w1*4), // position 1
	}
	if len(got) != len(want) {
		t.Fatalf("output len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > mlaTol {
			t.Errorf("out[%d] = %f, want %f (diff %g)", i, got[i], want[i], diff)
		}
	}
}

// TestAttendLatent_NoMask_Good checks the unmasked (full-attention) path: with
// identity Q/K and scale=1 the scores are the identity matrix, softmax over
// [1,0] and [0,1] both yield weights [e/(1+e), 1/(1+e)] biased to the diagonal.
// Confirms the kernel runs the nil-mask branch (mask == nil) cleanly.
func TestAttendLatent_NoMask_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	defer metal.Free(q, k, v)

	out := attendLatent(q, k, v, nil, 1.0)
	defer metal.Free(out)

	got := out.Floats()
	hi := math.E / (1.0 + math.E) // weight on the matching position
	lo := 1.0 / (1.0 + math.E)
	want := []float32{
		float32(hi*1 + lo*3), float32(hi*2 + lo*4), // pos0 weights [hi,lo]
		float32(lo*1 + hi*3), float32(lo*2 + hi*4), // pos1 weights [lo,hi]
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > mlaTol {
			t.Errorf("out[%d] = %f, want %f (diff %g)", i, got[i], want[i], diff)
		}
	}
}

// TestAttendLatent_Scale_Good confirms the scale multiplier is applied to the
// scores before softmax: scale=0 flattens every score to 0, so softmax is
// uniform and the output is the mean of the values, regardless of Q/K.
func TestAttendLatent_Scale_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{2, -1, 5, 7}, 1, 1, 2, 2)
	k := metal.FromValues([]float32{3, 1, -2, 4}, 1, 1, 2, 2)
	v := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	defer metal.Free(q, k, v)

	out := attendLatent(q, k, v, nil, 0.0)
	defer metal.Free(out)

	got := out.Floats()
	// Uniform softmax → mean of V rows = [(1+3)/2, (2+4)/2] = [2,3] for both positions.
	want := []float32{2, 3, 2, 3}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > mlaTol {
			t.Errorf("out[%d] = %f, want %f (scale=0 should give uniform mean)", i, got[i], want[i])
		}
	}
}
