// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// deltanet_coverage_test.go reaches the defensive/branch statements the oracle
// and layout tests in deltanet_test.go / builder_test.go don't exercise: the
// builder's missing-projection and headDim-fallback arms, the public kernels'
// resolve-failure early-outs, the gated path's alpha-shape guard, Forward's
// nil-Beta and nil-output arms, and the bf16 cast bookkeeping inside the
// chunked-parallel solve window. No model load — all synthetic geometry.
//
// The solve-failure cluster (computeParallel's outW==nil cleanup, solveWindow's
// metal.Solve error branch, and DeltaRuleChunk's sequential fallback — eight
// statements) is NOT reachable through the public surface: metal.Solve errors
// only on a non-square first argument (TestLinalgOp_SolveNonSquare_Bad), and the
// within-window matrix M is [B,H,wl,wl] — square by construction — so NaN, Inf
// and singular inputs return a (NaN-filled) result with rc=0 rather than an
// error. The three blocks are one coupled cluster (computeParallel returns nil
// iff solveWindow's solve fails iff DeltaRuleChunk falls back) and stay dead
// without a production seam, which is out of scope for a tests-only change.

// TestDeltanet_BuildMixer_MissingProjection_Bad covers builder.go's
// missing-q/k/v/o guard: a resolver that yields nil for q_proj makes buildMixer
// surface a clear error instead of a half-built mixer.
func TestDeltanet_BuildMixer_MissingProjection_Bad(t *testing.T) {
	noQ := func(name string) *metal.Linear {
		if name == "q_proj" {
			return nil
		}
		return builderLinear(builderDModel)(name)
	}
	if _, err := buildMixer(metal.MixerBuildCtx{Linear: noQ, Cfg: builderCfg()}); err == nil {
		t.Fatal("builder accepted a layer missing q_proj")
	}
}

// TestDeltanet_HeadGeometry_DerivedHeadDim_Good covers builder.go's headDim
// fallback: when the config states no HeadDim but gives HiddenSize and a head
// count, the builder derives HeadDim = HiddenSize / NumHeads. The built mixer
// must then run a forward at that derived per-head width.
func TestDeltanet_HeadGeometry_DerivedHeadDim_Good(t *testing.T) {
	const (
		heads  = 2
		dModel = 8 // → derived HeadDim = 8/2 = 4
	)
	cfg := metal.TransformerConfig{
		HiddenSize:        dModel,
		NumAttentionHeads: heads,
		// HeadDim deliberately omitted (0) → fallback path.
	}
	resolver := func(name string) *metal.Linear {
		if name == "b_proj" {
			vals := make([]float32, heads*dModel)
			for i := range vals {
				vals[i] = 0.02 + float32(i%5)*0.01
			}
			return metal.NewLinear(metal.FromValues(vals, heads, dModel), nil)
		}
		vals := make([]float32, dModel*dModel)
		for i := 0; i < dModel; i++ {
			vals[i*dModel+i] = 1
		}
		return metal.NewLinear(metal.FromValues(vals, dModel, dModel), nil)
	}
	m, err := buildMixer(metal.MixerBuildCtx{Linear: resolver, Cfg: cfg})
	if err != nil {
		t.Fatalf("buildMixer with derived headDim: %v", err)
	}
	dn, ok := m.(*Mixer)
	if !ok {
		t.Fatalf("buildMixer returned %T, want *Mixer", m)
	}
	if dn.W.HeadDim != dModel/heads {
		t.Fatalf("derived HeadDim = %d, want %d", dn.W.HeadDim, dModel/heads)
	}

	vals := make([]float32, 2*dModel)
	for i := range vals {
		vals[i] = 0.1 + float32(i%6)*0.05
	}
	x := metal.FromValues(vals, 1, 2, dModel)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: metal.NewRecurrentCache(), B: 1, L: 2})
	if out == nil || !out.Valid() {
		t.Fatal("Forward with derived headDim returned no output")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if got := out.Dim(2); got != dModel {
		t.Fatalf("output D_model = %d, want %d", got, dModel)
	}
}

// TestDeltanet_SequentialResolveFail_Bad covers DeltaRuleChunkSequential's
// resolve-failure early-out: a nil query makes resolve() reject the input and
// the function surface (nil, nil) rather than dereference it.
func TestDeltanet_SequentialResolveFail_Bad(t *testing.T) {
	beta := metal.FromValues([]float32{0.3, 0.4}, 1, 1, 2, 1)
	defer metal.Free(beta)
	if out, st := DeltaRuleChunkSequential(nil, nil, nil, beta, nil, 1, 0); out != nil || st != nil {
		t.Fatal("DeltaRuleChunkSequential accepted a nil query")
	}
}

// TestDeltanet_GatedResolveFail_Bad covers GatedDeltaRuleChunkSequential's
// resolve-failure early-out with a nil query.
func TestDeltanet_GatedResolveFail_Bad(t *testing.T) {
	beta := metal.FromValues([]float32{0.3, 0.4}, 1, 1, 2, 1)
	alpha := metal.FromValues([]float32{0.9, 0.9}, 1, 1, 2, 1)
	defer metal.Free(beta, alpha)
	if out, st := GatedDeltaRuleChunkSequential(nil, nil, nil, beta, alpha, nil, 1, 0); out != nil || st != nil {
		t.Fatal("GatedDeltaRuleChunkSequential accepted a nil query")
	}
}

// TestDeltanet_GatedAlphaShapeMismatch_Bad covers resolve()'s alpha-shape guard:
// q/k/v/beta are well-formed but alpha carries the wrong last dim, so the gated
// kernel rejects it (alpha must be [B,H,L,1]).
func TestDeltanet_GatedAlphaShapeMismatch_Bad(t *testing.T) {
	f := deltaFixture{h: 1, l: 2, d: 2}
	q, _ := f.tensor(0.1)
	k, _ := f.tensor(0.1)
	v, _ := f.tensor(0.1)
	beta, _ := f.beta()
	// alpha with a non-1 last dim → fails the [B,H,L,1] check at deltanet.go:176.
	badAlpha := metal.FromValues(make([]float32, int(f.h*f.l)*2), 1, int(f.h), int(f.l), 2)
	defer metal.Free(q, k, v, beta, badAlpha)
	if out, st := GatedDeltaRuleChunkSequential(q, k, v, beta, badAlpha, nil, 1, 0); out != nil || st != nil {
		t.Fatal("GatedDeltaRuleChunkSequential accepted a mis-shaped alpha")
	}
}

// TestDeltanet_Forward_NilBeta_Bad covers mixer.go's nil-Beta arm: a Mixer whose
// β projection was never wired surfaces a clean nil output rather than fabricate
// a write strength.
func TestDeltanet_Forward_NilBeta_Bad(t *testing.T) {
	m := &Mixer{
		W: &Weights{
			QProj:    identityLinear(t, 4),
			KProj:    identityLinear(t, 4),
			VProj:    identityLinear(t, 4),
			Output:   identityLinear(t, 4),
			NumHeads: 2,
			HeadDim:  2,
		},
		Beta: nil, // not wired
	}
	x := tokenInput(1, 4, 0.2)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: 1})
	if out != nil {
		t.Fatal("Forward with nil Beta returned a non-nil output")
	}
}

// TestDeltanet_Forward_NilKernelOutput_Bad covers mixer.go's out==nil arm: a Beta
// provider that emits a mis-shaped β makes the DeltaRuleChunk kernel's resolve()
// reject the input and return nil, so Forward must surface nil (not merge/project
// a nil read-out). Drives the path between projectHeads and the output projection.
func TestDeltanet_Forward_NilKernelOutput_Bad(t *testing.T) {
	m := &Mixer{
		W: &Weights{
			QProj:    identityLinear(t, 4),
			KProj:    identityLinear(t, 4),
			VProj:    identityLinear(t, 4),
			Output:   identityLinear(t, 4),
			NumHeads: 2,
			HeadDim:  2,
		},
		// β with last dim 2 (kernel needs [B,H,L,1]) → DeltaRuleChunk.resolve fails.
		Beta: func(_ *metal.Array, b, l int32) *metal.Array {
			return metal.FromValues(make([]float32, int(b)*2*int(l)*2), int(b), 2, int(l), 2)
		},
	}
	x := tokenInput(1, 4, 0.2)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: 1})
	if out != nil {
		t.Fatal("Forward returned non-nil when the kernel rejected the write strength")
	}
}

// TestDeltanet_ChunkedParallel_BF16_Good covers the bf16 cast bookkeeping in the
// chunked-parallel solve window: LAPACK has no bf16 path, so a bf16 right-hand
// side is cast up to float32 for the solve (asFloat32 with cast=true → rCast
// free) and the solution cast back to bf16 (castBack non-float32 path). L>1 so
// the parallel window runs; the result must match the sequential recurrence to
// tolerance, proving the round-trip is numerically faithful, not just executed.
func TestDeltanet_ChunkedParallel_BF16_Good(t *testing.T) {
	f := deltaFixture{h: 2, l: 8, d: 4}
	q32, _ := f.longTensor(0.3)
	k32, _ := f.longTensor(-0.2)
	v32, _ := f.longTensor(0.15)
	beta32, _ := f.longBeta()
	defer metal.Free(q32, k32, v32, beta32)

	q := metal.AsType(q32, metal.DTypeBFloat16)
	k := metal.AsType(k32, metal.DTypeBFloat16)
	v := metal.AsType(v32, metal.DTypeBFloat16)
	beta := metal.AsType(beta32, metal.DTypeBFloat16)
	defer metal.Free(q, k, v, beta)

	par, parS := DeltaRuleChunk(q, k, v, beta, nil, 0.5, 0)
	if par == nil {
		t.Fatal("bf16 parallel DeltaRuleChunk returned nil")
	}
	defer metal.Free(par, parS)
	// Reference: the same inputs through the exact sequential recurrence (also
	// bf16) — the WY solve is exact, so the two must agree within bf16 tolerance.
	seq, seqS := DeltaRuleChunkSequential(q, k, v, beta, nil, 0.5, 0)
	if seq == nil {
		t.Fatal("bf16 sequential DeltaRuleChunkSequential returned nil")
	}
	defer metal.Free(seq, seqS)

	gotPar := evalFloats(t, "bf16 parallel out", par)
	gotSeq := evalFloats(t, "bf16 sequential out", seq)
	if len(gotPar) != len(gotSeq) {
		t.Fatalf("length mismatch: parallel %d, sequential %d", len(gotPar), len(gotSeq))
	}
	const tol = 5e-2 // bf16 has ~3 significant decimal digits
	for i := range gotSeq {
		if d := gotPar[i] - gotSeq[i]; d > tol || d < -tol {
			t.Fatalf("bf16 out[%d]: parallel %g sequential %g (Δ %g)", i, gotPar[i], gotSeq[i], d)
		}
	}
}

// TestDeltanet_ChunkedParallel_Float64_NoCrash_Ugly drives the chunked-parallel
// window with float64 inputs. It asserts one real robustness property: an
// unusual-but-valid input dtype does not crash the kernel — DeltaRuleChunk
// returns rather than panicking. As a by-product it reaches the within-window
// matrix M as float64 (the masks force M float32 for bf16/float32 inputs, so the
// asFloat32(M)-casts-M arm only fires when the keys are float64), exercising the
// cast bookkeeping from a side the bf16 test cannot. Metal's LAPACK float64 solve
// does not materialise a usable array on every build, so no numerical claim is
// made — only that the call completes and frees cleanly.
func TestDeltanet_ChunkedParallel_Float64_NoCrash_Ugly(t *testing.T) {
	const (
		h, l, d = 1, 8, 3
	)
	mk := func(seed float64) *metal.Array {
		vals := make([]float64, h*l*d)
		for i := range vals {
			vals[i] = seed + float64(i%7)*0.05 - float64(i%3)*0.03
		}
		return metal.FromValues(vals, 1, h, l, d)
	}
	q := mk(0.3)
	k := mk(-0.2)
	v := mk(0.15)
	bvals := make([]float64, h*l)
	for i := range bvals {
		bvals[i] = 0.3 + float64(i%3)*0.2
	}
	beta := metal.FromValues(bvals, 1, h, l, 1)
	defer metal.Free(q, k, v, beta)

	// The only assertion: this returns without panicking. par/parS may be nil or
	// (on a build where Metal's f64 solve yields an invalid array) non-nil but
	// invalid — either way the kernel must not crash and must hand back arrays the
	// caller can free.
	par, parS := DeltaRuleChunk(q, k, v, beta, nil, 0.5, 0)
	metal.Free(par, parS)
}
