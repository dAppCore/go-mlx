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

// TestMixer_CacheMode_Good pins MLA's bespoke cache declaration: it names the
// "mla-latent" scheme so the KV factory builds its compressed-latent store, and
// the factory's resolver (metal.CacheModeForMixer) returns that mode for MLA.
func TestMixer_CacheMode_Good(t *testing.T) {
	m := &Mixer{}
	if got := m.CacheMode(); got != string(metal.KVCacheModeMLALatent) {
		t.Fatalf("CacheMode() = %q, want %q", got, metal.KVCacheModeMLALatent)
	}
	if got := metal.CacheModeForMixer(m); got != string(metal.KVCacheModeMLALatent) {
		t.Fatalf("CacheModeForMixer(MLA) = %q, want %q", got, metal.KVCacheModeMLALatent)
	}
}

// causalMaskN builds the additive [1,1,n,n] causal mask: position i sees keys
// 0..i, future keys are -inf. The general form of causalMask2 for the
// decode-oracle test (which prefills n-1 then decodes the nth token).
func causalMaskN(n int32) *metal.Array {
	neg := float32(math.Inf(-1))
	vals := make([]float32, n*n)
	for i := int32(0); i < n; i++ {
		for j := int32(0); j < n; j++ {
			if j > i {
				vals[i*n+j] = neg
			}
		}
	}
	return metal.FromValues(vals, 1, 1, int(n), int(n))
}

// mlaLin builds a deterministic, varied, bounded Linear [out,in] for the Forward
// oracle — distinct per projection (via base) so the reconstructed Q/K/V differ
// across positions and the softmax genuinely discriminates between keys (a
// concat / ordering bug then changes the last-token output, not just round-off).
func mlaLin(out, in int32, base float32) *metal.Linear {
	n := out * in
	w := make([]float32, n)
	for i := int32(0); i < n; i++ {
		w[i] = base + 0.13*float32(i%5) - 0.07*float32(i%3)
	}
	return metal.NewLinear(metal.FromValues(w, int(out), int(in)), nil)
}

// TestForward_DecodeMatchesPrefill_Good is the decode-caching contract for #1:
// a single-token decode that attends through the latent cache must reproduce the
// full-prefill's last-token output. Because this MLA carries no RoPE, the cached
// latent is position-independent, so up-projecting the concatenated latent is
// per-row identical to up-projecting a one-shot latent — chunked decode must
// reconstruct full-prefill K/V to round-off.
//
// Reference: one-shot full prefill of 3 tokens, NO cache (proves the nil-cache
// path too), last token = oracle. Then a fresh latent cache: prefill 2 tokens
// (causal mask), decode the 3rd alone (mask nil — the last position legitimately
// sees every key). The decode output must equal the reference within mlaTol.
//
// Scope: this pins prefill + single-token decode. L>1 attention over prior
// cached history needs an [L,totalL] mask MLA does not yet build — the named
// next piece, not covered here.
func TestForward_DecodeMatchesPrefill_Good(t *testing.T) {
	requireMetalRuntime(t)

	// hidden=4, kvLatentDim=4, qLatentDim=4, heads=2, HeadDim=2.
	// WUK out = heads*2*HeadDim = 8 (per-head interleaved K|V); WUQ out =
	// heads*HeadDim = 4; OProj 4 → hidden.
	m := &Mixer{
		WDKV:     mlaLin(4, 4, 0.10),
		WUK:      mlaLin(8, 4, 0.05),
		WDQ:      mlaLin(4, 4, 0.12),
		WUQ:      mlaLin(4, 4, 0.08),
		OProj:    mlaLin(4, 4, 0.09),
		NumHeads: 2,
		HeadDim:  2,
		Scale:    0.5,
	}
	defer metal.FreeLinear(m.WDKV)
	defer metal.FreeLinear(m.WUK)
	defer metal.FreeLinear(m.WDQ)
	defer metal.FreeLinear(m.WUQ)
	defer metal.FreeLinear(m.OProj)

	// Three distinct token rows [B=1, L=3, hidden=4].
	x := metal.FromValues([]float32{
		0.5, -0.2, 0.9, 0.1,
		-0.3, 0.7, 0.2, -0.6,
		0.8, 0.4, -0.5, 0.3,
	}, 1, 3, 4)
	defer metal.Free(x)

	// Reference: full prefill, no cache. Extract the last token's output as a
	// materialised Go slice BEFORE the decode steps mutate/free shared graph state.
	mask3 := causalMaskN(3)
	full, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: 3, Mask: mask3})
	last := metal.Slice(full, []int32{0, 2, 0}, []int32{1, 3, 4}) // [1,1,4]
	metal.Materialize(last)
	ref := last.Floats()
	metal.Free(full, last, mask3)

	// Decode path: fresh latent cache, prefill 2 tokens then decode the 3rd.
	c := metal.NewLatentKVCache()
	defer c.Reset()

	x01 := metal.Slice(x, []int32{0, 0, 0}, []int32{1, 2, 4})
	mask2 := causalMaskN(2)
	pre, _ := m.Forward(x01, &metal.MixerCtx{Cache: c, B: 1, L: 2, Mask: mask2})
	metal.Free(x01, mask2, pre)

	x2 := metal.Slice(x, []int32{0, 2, 0}, []int32{1, 3, 4})
	dec, _ := m.Forward(x2, &metal.MixerCtx{Cache: c, B: 1, L: 1, Mask: nil})
	metal.Free(x2)
	metal.Materialize(dec)
	got := dec.Floats()
	metal.Free(dec)

	if c.Offset() != 3 {
		t.Fatalf("cache offset after prefill+decode = %d, want 3", c.Offset())
	}
	if len(got) != len(ref) {
		t.Fatalf("decode output len = %d, want %d (prefill last token)", len(got), len(ref))
	}
	for i := range ref {
		if diff := math.Abs(float64(got[i] - ref[i])); diff > mlaTol {
			t.Errorf("decode[%d] = %f, prefill-last[%d] = %f (diff %g) — cache reconstruction wrong",
				i, got[i], i, ref[i], diff)
		}
	}
}

// materialiseRows slices [start:stop) along the sequence axis of a [1,L,hidden]
// tensor and returns the values as a Go slice, materialised so they survive the
// later graph mutation the chunked forwards cause.
func materialiseRows(out *metal.Array, start, stop int32) []float32 {
	h := int32(out.Dim(2))
	s := metal.Slice(out, []int32{0, start, 0}, []int32{1, stop, h})
	metal.Materialize(s)
	vals := s.Floats()
	metal.Free(s)
	return vals
}

// assertRowsClose fails t when got and want differ by more than mlaTol.
func assertRowsClose(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > mlaTol {
			t.Errorf("%s[%d] = %f, want %f (diff %g)", label, i, got[i], want[i], diff)
		}
	}
}

// TestForward_ChunkedPrefillMatchesOneShot_Good is the chunked-prefill contract:
// prefilling a prompt in several L>1 chunks (each appended to the warm cache)
// must reproduce a one-shot prefill of the whole prompt, token for token. This
// is what the internal offset-causal mask (MultiTokenCausalMask, histLen>0)
// buys — without it the second chunk would attend its tokens over the full
// history with no within-chunk causality. Both paths pass a nil mask so Forward
// builds the causal mask itself: the production (composed) path.
//
// chunk1 == one-shot[0:2] proves chunk-1 prefill is causal; chunk2 == one-shot
// [2:4] is where the [L,totalL] offset-causal mask earns its keep (a wrong
// offset only shows there). Both are exact: masked softmax weights underflow to
// zero in float32, so a one-shot token gets literally nothing from its
// future-masked keys — identical to a chunk that never held them.
func TestForward_ChunkedPrefillMatchesOneShot_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := &Mixer{
		WDKV:     mlaLin(4, 4, 0.10),
		WUK:      mlaLin(8, 4, 0.05),
		WDQ:      mlaLin(4, 4, 0.12),
		WUQ:      mlaLin(4, 4, 0.08),
		OProj:    mlaLin(4, 4, 0.09),
		NumHeads: 2,
		HeadDim:  2,
		Scale:    0.5,
	}
	defer metal.FreeLinear(m.WDKV)
	defer metal.FreeLinear(m.WUK)
	defer metal.FreeLinear(m.WDQ)
	defer metal.FreeLinear(m.WUQ)
	defer metal.FreeLinear(m.OProj)

	// Four distinct token rows [B=1, L=4, hidden=4].
	x := metal.FromValues([]float32{
		0.5, -0.2, 0.9, 0.1,
		-0.3, 0.7, 0.2, -0.6,
		0.8, 0.4, -0.5, 0.3,
		-0.1, -0.4, 0.6, 0.2,
	}, 1, 4, 4)
	defer metal.Free(x)

	// Reference: one-shot prefill of all 4 tokens, nil mask (internal causal).
	// Materialise the per-chunk slices to Go BEFORE the chunked forwards run —
	// they mutate/free shared graph state.
	oneShot, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: 4, Mask: nil})
	refA := materialiseRows(oneShot, 0, 2) // tokens 0,1
	refB := materialiseRows(oneShot, 2, 4) // tokens 2,3
	metal.Free(oneShot)

	// Chunked: a fresh cache, prefilled in two L=2 chunks (offsets 0 then 2).
	c := metal.NewLatentKVCache()
	defer c.Reset()

	x01 := metal.Slice(x, []int32{0, 0, 0}, []int32{1, 2, 4})
	c1, _ := m.Forward(x01, &metal.MixerCtx{Cache: c, B: 1, L: 2, Mask: nil})
	metal.Materialize(c1)
	gotA := c1.Floats()
	metal.Free(x01, c1)

	x23 := metal.Slice(x, []int32{0, 2, 0}, []int32{1, 4, 4})
	c2, _ := m.Forward(x23, &metal.MixerCtx{Cache: c, B: 1, L: 2, Mask: nil})
	metal.Materialize(c2)
	gotB := c2.Floats()
	metal.Free(x23, c2)

	if c.Offset() != 4 {
		t.Fatalf("cache offset after two L=2 chunks = %d, want 4", c.Offset())
	}
	assertRowsClose(t, "chunk1", gotA, refA) // chunk-1 prefill is causal
	assertRowsClose(t, "chunk2", gotB, refB) // the [L,totalL] offset-causal mask
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

// TestUpProjectKV_PerHeadInterleaved_Good pins the K/V split LAYOUT, not just
// its shape — the bug a shape-only test masks. DeepSeek-V2's kv_b_proj output is
// per-head interleaved ([h0_K, h0_V, h1_K, h1_V, …]), split along the per-head
// last axis — NOT block-concatenated ([all-K | all-V]). With an identity WUK so
// kv == cKV, heads=2, HeadDim=1, cKV = [1, 2, 3, 4] views as head0=[1,2],
// head1=[3,4]; the per-head split must yield K = [head0_K, head1_K] = [1, 3] and
// V = [head0_V, head1_V] = [2, 4]. A block split would (wrongly) give K=[1,2],
// V=[3,4] — this test fails loudly on that layout.
func TestUpProjectKV_PerHeadInterleaved_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Identity WUK [4,4] so WUK.Forward(cKV) == cKV (y = cKV @ Iᵀ).
	id := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	m := &Mixer{WUK: metal.NewLinear(metal.FromValues(id, 4, 4), nil), NumHeads: 2, HeadDim: 1}
	defer metal.FreeLinear(m.WUK)

	cKV := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 4) // [B=1,L=1, heads*2*HeadDim=4]
	defer metal.Free(cKV)

	kFlat, vFlat := m.upProjectKV(cKV, 1, 1)
	defer metal.Free(kFlat, vFlat)

	// Force evaluation of the slice-over-reshape views before reading. Reading
	// the two views back-to-back without an explicit Materialize can surface a
	// stale lazy result for the second; materialising both first is the reliable
	// read (the Forward path forces eval naturally via the downstream matmuls).
	metal.Materialize(kFlat, vFlat)
	gotK := kFlat.Floats()
	gotV := vFlat.Floats()
	if len(gotK) != 2 || gotK[0] != 1 || gotK[1] != 3 {
		t.Errorf("K = %v, want [1 3] (per-head K: head0_K, head1_K)", gotK)
	}
	if len(gotV) != 2 || gotV[0] != 2 || gotV[1] != 4 {
		t.Errorf("V = %v, want [2 4] (per-head V: head0_V, head1_V)", gotV)
	}
}
