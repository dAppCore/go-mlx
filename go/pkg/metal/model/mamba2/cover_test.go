// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// cover_test.go closes the remaining reachable-statement gaps the behavioural
// tests leave open: the loader's missing-tensor + bad-geometry guards, the
// mixer's nil-weights early-out, the G==H (per-head B/C) group-expansion no-op,
// the advanceState fresh-state (nil entry) arm of the chunked scan, and Forward's
// no-holder free path. Each test is named for the exact production branch it
// pins so a coverage regression maps straight back to the code it guards.
//
// The defensive eval-failure logs (chunk.go / scan.go / the holder write in
// mixer.go) and the y==nil early-out are NOT exercised here: the real scan
// kernels never return a nil output and a passing metal.Eval never reports a
// non-zero rc, so those branches are only reachable by fault injection — which
// "tests only, no production-code change" rules out and which the sibling mixer
// packages leave uncovered for the same reason.

// TestLoader_MissingInProj_Bad: a missing in_proj is a loud build error.
func TestLoader_MissingInProj_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "in_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildMamba2(ctx); err == nil {
		t.Error("expected error for missing in_proj, got nil")
	}
}

// TestLoader_MissingOutProj_Bad: a missing out_proj is a loud build error.
func TestLoader_MissingOutProj_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "out_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildMamba2(ctx); err == nil {
		t.Error("expected error for missing out_proj, got nil")
	}
}

// TestLoader_MissingALog_Bad: a missing A_log is a loud build error (the SSD
// scan cannot derive its per-head decay without it).
func TestLoader_MissingALog_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Weight
	ctx.Weight = func(n string) *metal.Array {
		if n == "A_log" {
			return nil
		}
		return base(n)
	}
	if _, err := buildMamba2(ctx); err == nil {
		t.Error("expected error for missing A_log, got nil")
	}
}

// TestLoader_BadGeometry_Bad routes an nGroups=1-inconsistent shape set through
// buildMamba2 so configFromShapes rejects it AND the wrapper surfaces the
// "derive geometry" error (covers both the guard in configFromShapes and the
// error return in buildMamba2). Here A_log has an odd head count that makes
// dInner = projOut − convDim − H non-divisible, the loud-error case the doc
// promises instead of a silent mis-split.
func TestLoader_BadGeometry_Bad(t *testing.T) {
	const D = 4
	// Keep in_proj/out_proj/conv shapes from the good fixture (projOut=14,
	// convDim=8) but give A_log H=3 → dInner = 14 − 8 − 3 = 3, and 3%3==0, so
	// nudge with H=5 → dInner = 14 − 8 − 5 = 1, 1%5 != 0 ⇒ inconsistent.
	linears := map[string]*metal.Linear{
		"in_proj":  metal.NewLinear(metal.FromValues(make([]float32, 14*D), 14, D), nil),
		"out_proj": metal.NewLinear(metal.FromValues(make([]float32, D*4), D, 4), nil),
	}
	weights := map[string]*metal.Array{
		"conv1d.weight": metal.FromValues(make([]float32, 8*1*3), 8, 1, 3),
		"A_log":         metal.FromValues(make([]float32, 5), 5),
	}
	ctx := metal.MixerBuildCtx{
		Linear: func(p string) *metal.Linear { return linears[p] },
		Weight: func(n string) *metal.Array { return weights[n] },
		Cfg:    metal.TransformerConfig{RMSNormEps: 1e-5},
	}
	if _, err := buildMamba2(ctx); err == nil {
		t.Error("expected geometry error for nGroups=1-inconsistent shapes, got nil")
	}
}

// TestConfigFromShapes_NilWeight_Bad pins the in_proj-has-no-weight guard with a
// direct call (a Linear whose dense Weight was never set — the quant-factory
// invariant the loader assumes, asserted loudly rather than nil-deref'd).
func TestConfigFromShapes_NilWeight_Bad(t *testing.T) {
	conv := metal.FromValues(make([]float32, 8*1*3), 8, 1, 3)
	aLog := metal.FromValues(make([]float32, 2), 2)
	defer metal.Free(conv, aLog)
	if _, err := configFromShapes(&metal.Linear{}, conv, aLog, 1e-5); err == nil {
		t.Error("expected error for in_proj with nil weight, got nil")
	}
}

// TestConfigFromShapes_EmptyConvShape_Bad pins the scalar-conv guard: a conv
// weight with a rank-0 (empty) shape has no convDim to read, so geometry
// derivation must error rather than index a zero-length shape.
func TestConfigFromShapes_EmptyConvShape_Bad(t *testing.T) {
	inProj := metal.NewLinear(metal.FromValues(make([]float32, 14*4), 14, 4), nil)
	conv := metal.FromValue(float32(1.0)) // rank-0 scalar ⇒ Shape() is empty
	aLog := metal.FromValues(make([]float32, 2), 2)
	defer metal.Free(inProj.Weight, conv, aLog)
	if _, err := configFromShapes(inProj, conv, aLog, 1e-5); err == nil {
		t.Error("expected error for conv1d.weight with empty shape, got nil")
	}
}

// TestMixer_Forward_NilWeights_Ugly pins the nil-weights / nil-config early-out:
// a zero-value Mixer (and one with weights but no config) returns (nil, zero)
// from Forward rather than nil-deref'ing the geometry — the seed-before-load
// shape where a Mixer exists but the loader has not attached weights yet.
func TestMixer_Forward_NilWeights_Ugly(t *testing.T) {
	x := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 1, 4)
	defer metal.Free(x)

	out, kv := (&Mixer{}).Forward(x, &metal.MixerCtx{B: 1, L: 1})
	if out != nil {
		t.Errorf("Forward on nil-weights Mixer = %v, want nil", out)
	}
	if kv.HasState() {
		t.Error("Forward on nil-weights Mixer returned a SharedKV with state")
	}

	// Weights present but no Config attached (NewMixer with a nil cfg) — same
	// early-out via the cfg == nil arm.
	out2, _ := (&Mixer{W: &Weights{}}).Forward(x, &metal.MixerCtx{B: 1, L: 1})
	if out2 != nil {
		t.Errorf("Forward on nil-config Mixer = %v, want nil", out2)
	}
}

// ghMixer builds a tiny Mamba-2 mixer whose group count EQUALS its head count
// (nGroups == H): the per-head B/C layout where groupToHeads is a no-op reshape
// (no broadcast-repeat), the arm tinyMixer's G==1 and the oracle's 1<G<H configs
// never reach. H=2, P=2, N=2, G=2 ⇒ dInner=4, convDim=12, projOut=18, D=4.
func ghMixer() *Mixer {
	const D, H, P, N, G, K = 4, 2, 2, 2, 2, 3
	cfg := &Config{NumHeads: H, HeadDim: P, StateDim: N, NumGroups: G, ConvKernel: K, RMSNormEps: 1e-5}
	dInner := int(cfg.dInner())   // 4
	convDim := int(cfg.convDim()) // 4 + 2*2*2 = 12
	projOut := 2*dInner + 2*G*N + H

	inW := make([]float32, projOut*D)
	for i := range inW {
		inW[i] = 0.03 * float32((i%5)-2)
	}
	convW := make([]float32, convDim*1*K)
	for i := range convW {
		convW[i] = 0.05 * float32((i%3)+1)
	}
	w := &Weights{
		InProj:     metal.NewLinear(metal.FromValues(inW, projOut, D), nil),
		ConvWeight: metal.FromValues(convW, convDim, 1, K),
		ALog:       metal.FromValues([]float32{0.0, -0.3}, H),
		D:          metal.FromValues([]float32{0.1, 0.2}, H),
		DtBias:     metal.FromValues([]float32{-0.5, -0.4}, H),
		Norm:       metal.FromValues([]float32{1.0, 1.1, 0.9, 1.05}, dInner),
		OutProj:    metal.NewLinear(metal.FromValues(make([]float32, D*dInner), D, dInner), nil),
	}
	return NewMixer(w, cfg)
}

// TestMixer_Forward_GroupsEqualHeads_Good drives the nGroups==H group-expansion
// no-op through a real prefill: each head owns its own B/C, so groupToHeads
// returns the [B,L,G,N] reshape unchanged. The output must be finite and the
// right shape — the per-head case is a valid Mamba-2 layout the loader can infer.
func TestMixer_Forward_GroupsEqualHeads_Good(t *testing.T) {
	m := ghMixer()
	defer freeMixer(m)
	const B, L, D = 1, 3, 4

	xVals := make([]float32, B*L*D)
	for i := range xVals {
		xVals[i] = 0.1 * float32((i%4)-1)
	}
	x := metal.FromValues(xVals, B, L, D)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: L})
	if out == nil {
		t.Fatal("Forward (G==H) returned nil output")
	}
	defer metal.Free(out)

	if got := out.Shape(); len(got) != 3 || got[0] != B || got[1] != L || got[2] != D {
		t.Fatalf("out shape = %v, want [%d %d %d]", got, B, L, D)
	}
	for i, v := range out.Floats() {
		if v != v { // NaN
			t.Fatalf("out[%d] is NaN", i)
		}
	}
}

// TestMixer_Forward_NoHolder_Ugly drives Forward with a bare MixerCtx (nil Cache
// ⇒ Recurrent() reports false): the self-contained zero-state path the doc
// describes for a unit-test caller. It exercises the else-arm that frees the
// advanced conv/ssm state instead of writing it to a holder, and must still
// return a finite output of the right shape.
func TestMixer_Forward_NoHolder_Ugly(t *testing.T) {
	m := tinyMixer(t)
	defer freeMixer(m)
	const B, L, D = 1, 2, 4

	x := metal.FromValues([]float32{0.5, -0.2, 0.1, 0.3, -0.1, 0.4, -0.3, 0.2}, B, L, D)
	defer metal.Free(x)

	out, _ := m.Forward(x, &metal.MixerCtx{B: B, L: L}) // no Cache ⇒ no holder
	if out == nil {
		t.Fatal("Forward with no holder returned nil output")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != B || got[1] != L || got[2] != D {
		t.Fatalf("no-holder out shape = %v, want [%d %d %d]", got, B, L, D)
	}
	for i, v := range out.Floats() {
		if v != v {
			t.Fatalf("no-holder out[%d] is NaN", i)
		}
	}
}

// TestChunk_AdvanceState_FreshState_Ugly pins the entry==nil arm of advanceState
// (the chunk-end state when there is no carried entry state: state_exit = S_c,
// no exp(L_C)·entry term). The chunked scan's public entry always hands compute
// a valid zero state, so this fresh-segment arm is reached by calling compute
// directly with a nil entry — idiomatic here as the loader-level helpers are
// already unit-tested white-box. The single-segment compute(nil) result must
// equal the sequential SSDScan from a zero state.
func TestChunk_AdvanceState_FreshState_Ugly(t *testing.T) {
	const L, H, P, N = 4, 1, 2, 2
	in, free := chunkInputs(L, H, P, N)
	defer free()

	// Reference: the exact sequential scan from a fresh (zero) state.
	seqY, seqState := SSDScan(in, nil)
	defer metal.Free(seqY, seqState)

	// One segment spanning the whole sequence, computed against a NIL entry
	// state — the advanceState fresh-state arm. dtX = Δ·X, a = Δ·A.
	dtCol := metal.Reshape(in.Dt, 1, L, H, 1)
	dtX := metal.Mul(in.X, dtCol)
	metal.Free(dtCol)
	aSeq := metal.Mul(in.Dt, in.A)
	defer metal.Free(dtX, aSeq)

	seg := segment{
		B: 1, H: H, P: P, N: N, cLen: L, dtype: in.X.Dtype(),
		dtX:   dtX,
		a:     aSeq,
		bProj: in.B,
		cProj: in.C,
	}
	y, state := seg.compute(nil) // nil entry ⇒ advanceState returns contrib directly
	defer metal.Free(y, state)

	if err := metal.Eval(y, state); err != nil {
		t.Fatalf("compute eval: %v", err)
	}

	// compute does NOT apply the D skip (that is folded once over the whole
	// concatenated output in SSDScanChunked), so compare against a no-D sequential
	// scan for the output, and the state directly (state carries no D term).
	inNoD := in
	inNoD.D = nil
	seqYnoD, seqStateNoD := SSDScan(inNoD, nil)
	defer metal.Free(seqYnoD, seqStateNoD)

	assertCloseF32(t, "compute(nil) y vs sequential (no D)", y.Floats(), seqYnoD.Floats(), 2e-3)
	assertCloseF32(t, "compute(nil) state vs sequential", state.Floats(), seqStateNoD.Floats(), 2e-3)
}
