// SPDX-Licence-Identifier: EUPL-1.2
//
// Perf invariants — falsifiable properties, not aspirational targets.
//
// "Make it faster" is unfalsifiable, and "N tok/s" invites a physics argument
// (the bandwidth ceiling rebuttal). These tests instead encode properties
// that can only go red when something is actually wrong:
//
//  1. ROUTING   — each quant decodes through the kernel measured fastest for
//                 it (AffineQuantPrefersGemm). Pure logic, no timing.
//  2. ORDERING  — quant decode cost must track bytes-per-weight. q8 beating
//                 q6 is bandwidth-impossible; an inversion is always a kernel
//                 or routing defect (this is exactly how the 2026-06-09 q6
//                 319 GB/s bitstream-kernel bug was caught).
//  3. ZERO-GARBAGE — per-token ops must not allocate on the Go heap. Normal
//                 Go hygiene; regressions here are GC pressure on the decode
//                 loop.
//  4. FLATNESS  — steady-state cache work must not get slower the longer it
//                 runs. Cumulative degradation is a leak or pool pathology.
//
// A red here is a bug hunt with a narrow scope, never a tuning argument.
package metal

import (
	"testing"
	"time"
)

// --- 1. ROUTING -------------------------------------------------------------

func TestPerfInvariant_AffineQuantRouting(t *testing.T) {
	requireMetalRuntime(t)

	cases := []struct {
		name       string
		outDim     int
		inDim      int
		groupSize  int
		bits       int
		wantGemm   bool
		whyNotGemm string
	}{
		{name: "q4_gs64", outDim: 8, inDim: 256, groupSize: 64, bits: 4, wantGemm: true},
		{name: "q8_gs64", outDim: 8, inDim: 256, groupSize: 64, bits: 8, wantGemm: true},
		{name: "q6_bitstream_gs64", outDim: 8, inDim: 256, groupSize: 64, bits: 6, wantGemm: true},
		// MLX ships qmv kernels only for groups 32/64/128 — gs=4 dies at Eval
		// with "Unable to load kernel affine_qmv_float_gs_4_…".
		{name: "q4_gs4_unsupported_group", outDim: 6, inDim: 8, groupSize: 4, bits: 4, wantGemm: false,
			whyNotGemm: "MLX has no qmv kernel for group size 4"},
		{name: "q6_gs5_unsupported_group", outDim: 4, inDim: 10, groupSize: 5, bits: 6, wantGemm: false,
			whyNotGemm: "MLX has no qmv kernel for group size 5"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fixture := quantizedLinearDenseMatVecFixture(t, tc.outDim, tc.inDim, tc.groupSize, tc.bits, 11)
			defer FreeLinear(fixture.linear)
			if got := AffineQuantPrefersGemm(fixture.linear); got != tc.wantGemm {
				t.Fatalf("AffineQuantPrefersGemm(%s) = %v, want %v %s", tc.name, got, tc.wantGemm, tc.whyNotGemm)
			}
		})
	}

	// Legacy-packed q6 (packedIn×5 == inDim, pre-bitstream layout) must stay
	// on the native kernel — MLX's gemm cannot read that layout at all.
	t.Run("q6_legacy_packed", func(t *testing.T) {
		const outDim, inDim, groupSize = 4, 320, 64
		packedIn := inDim / 5
		words := make([]uint32, outDim*packedIn)
		groups := inDim / groupSize
		scales := make([]float32, outDim*groups)
		biases := make([]float32, outDim*groups)
		for i := range scales {
			scales[i] = 0.01
		}
		linear := NewQuantizedLinear(
			FromValues(words, outDim, packedIn),
			FromValues(scales, outDim, groups),
			FromValues(biases, outDim, groups),
			nil, groupSize, 6,
		)
		defer FreeLinear(linear)
		if AffineQuantPrefersGemm(linear) {
			t.Fatal("AffineQuantPrefersGemm(legacy q6) = true; gemm cannot read legacy packing — wrong results, not just slow")
		}
	})
}

// --- 2. ORDERING ------------------------------------------------------------

// perfInvariantTimeQuantForward times the SERVE-ROUTED decode path (real
// Linear.Forward with the production gates on) for one quant: chained
// single-token calls into one Eval, min-of-rounds to reject noise.
func perfInvariantTimeQuantForward(t *testing.T, bits, dim int) time.Duration {
	t.Helper()
	const chain, itersPerRound, rounds = 64, 6, 3

	fixture := quantizedLinearDenseMatVecFixture(t, dim, dim, 64, bits, 41)
	lin := fixture.linear
	defer FreeLinear(lin)
	x0 := RandomUniform(-1, 1, []int32{1, 1, int32(dim)}, DTypeFloat32)
	Materialize(x0, lin.Weight, lin.Scales, lin.Biases)
	defer Free(x0)

	runChain := func() {
		outs := make([]*Array, 0, chain)
		x := x0
		for range chain {
			y := lin.Forward(x)
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			t.Fatalf("Eval(q%d dim%d): %v", bits, dim, err)
		}
		Free(outs...)
	}

	// JIT-compile kernels outside the timed window (the 3x-vs-100x trap:
	// cold kernel compilation once cost a 4.6x misread of the fused router).
	runChain()

	best := time.Duration(1<<62 - 1)
	for range rounds {
		start := time.Now()
		for range itersPerRound {
			runChain()
		}
		if d := time.Since(start); d < best {
			best = d
		}
	}
	return best
}

func TestPerfInvariant_QuantDecodeOrdering(t *testing.T) {
	requireMetalRuntime(t)
	if testing.Short() {
		t.Skip("timing invariant skipped in -short")
	}

	restoreNative := SetRuntimeGate(GateNativeLinearMatVec, true)
	defer restoreNative()
	restoreQ6 := SetRuntimeGate(GateNativeQ6BitstreamMatVec, true)
	defer restoreQ6()

	// dim must be big enough that the weight read dominates dispatch —
	// at 4096² the q6 read is ~12.6 MB/call, solidly bandwidth-bound.
	const dim = 4096
	q4 := perfInvariantTimeQuantForward(t, 4, dim)
	q6 := perfInvariantTimeQuantForward(t, 6, dim)
	q8 := perfInvariantTimeQuantForward(t, 8, dim)
	t.Logf("min-of-rounds: q4=%v q6=%v q8=%v (ratios q6/q4=%.2f q8/q6=%.2f)",
		q4, q6, q8, float64(q6)/float64(q4), float64(q8)/float64(q6))

	// HARD: q4 reads ~0.69x of q6's bytes and ~0.53x of q8's — it must win.
	if q4 >= q6 {
		t.Errorf("ORDERING INVERSION: q4 (%v) not faster than q6 (%v) — q4 reads fewer bytes; this is a kernel or routing defect", q4, q6)
	}
	if q4 >= q8 {
		t.Errorf("ORDERING INVERSION: q4 (%v) not faster than q8 (%v) — q4 reads half the bytes; this is a kernel or routing defect", q4, q8)
	}
	// SOFT CAP: by bytes q6 should BEAT q8 (~0.78x). Today MLX's own q6 qmv
	// kernel runs ~1.1x of q8 at this shape (upstream kernel cost, measured
	// 2026-06-09: gemm q6 39.4us vs q8 35.5us at dim 6144) — so the cap locks
	// in "no worse than the best known" with noise headroom. The 2026-06-09
	// custom-bitstream-kernel bug would score ~2.1x here and go red.
	// Tighten toward <1.0 when the upstream q6 kernel improves or is beaten.
	const q6VsQ8Cap = 1.25
	if ratio := float64(q6) / float64(q8); ratio > q6VsQ8Cap {
		t.Errorf("q6/q8 = %.2f exceeds %.2f: q6 reads FEWER bytes than q8 — a ratio this far above 1.0 means q6 is off its best kernel (see AffineQuantPrefersGemm)", ratio, q6VsQ8Cap)
	}
}

// --- 3. ZERO-GARBAGE --------------------------------------------------------

func TestPerfInvariant_PerTokenOpsAllocBudget(t *testing.T) {
	requireMetalRuntime(t)

	flat := Zeros([]int32{1024}, DTypeFloat32)
	defer Free(flat)
	four := Zeros([]int32{1, 8, 1, 128}, DTypeFloat32)
	defer Free(four)
	shape4 := []int32{1, 8, 1, 128}
	strides := []int64{1024, 128, 1024, 1}
	bshape := []int32{2, 8, 1, 128}

	// Budget is ≤1 Go heap alloc per op (the *Array wrapper); the pooled
	// shape-scratch work (commits d3de0a1f, 2d92e0ce, 1a181648) got the op
	// internals to zero. AllocsPerRun averages, so 1.5 absorbs rounding.
	const budget = 1.5
	cases := []struct {
		name string
		op   func()
	}{
		{"AsStrided_4D", func() { Free(AsStrided(flat, shape4, strides, 0)) }},
		{"Reshape_4D", func() { Free(Reshape(flat, shape4...)) }},
		{"Transpose_4D", func() { Free(Transpose(four, 0, 2, 1, 3)) }},
		{"BroadcastTo_4D", func() { Free(BroadcastTo(four, bshape)) }},
		{"AddScalar", func() { Free(AddScalar(four, 1.0)) }},
		{"Zeros_4D", func() { Free(Zeros(shape4, DTypeFloat32)) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.op() // warm any lazy init outside the measured window
			if avg := testing.AllocsPerRun(200, tc.op); avg > budget {
				t.Errorf("%s allocates %.1f/op (budget %.1f) — per-token garbage feeds the GC on the decode loop; pool it (see the W11 escape-pool pattern)", tc.name, avg, budget)
			}
		})
	}
}

// --- 4. FLATNESS ------------------------------------------------------------

func TestPerfInvariant_RotatingCacheSteadyStateFlat(t *testing.T) {
	requireMetalRuntime(t)
	if testing.Short() {
		t.Skip("timing invariant skipped in -short")
	}

	// Correct-usage steady state: one cache, Eval + Free every update (the
	// per-token serve pattern). The pre-2026-06-09 bench discarded Update's
	// returned views without Free and showed 17x cross-iteration growth —
	// that was the leak compounding, and this test pins the distinction:
	// under correct usage, round N must cost what round 1 cost.
	const (
		cap     = 256
		perRound = 384 // past cap from round 1's second half onward
		rounds  = 4
	)
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	cache := NewRotatingKVCache(cap)
	defer cache.Reset()

	// Reach steady state (past cap) before timing.
	for range cap + 32 {
		ck, cv := cache.Update(k, v, 1)
		if err := Eval(ck, cv); err != nil {
			t.Fatalf("Eval(warmup): %v", err)
		}
		Free(ck, cv)
	}

	durations := make([]time.Duration, 0, rounds)
	for range rounds {
		start := time.Now()
		for range perRound {
			ck, cv := cache.Update(k, v, 1)
			if err := Eval(ck, cv); err != nil {
				t.Fatalf("Eval: %v", err)
			}
			Free(ck, cv)
		}
		durations = append(durations, time.Since(start))
	}
	t.Logf("steady-state rounds: %v (cache_mb=%d)", durations, GetCacheMemory()/(1024*1024))

	first, last := durations[0], durations[len(durations)-1]
	const growthCap = 2.0
	if ratio := float64(last) / float64(first); ratio > growthCap {
		t.Errorf("CUMULATIVE DEGRADATION: round %d took %.1fx round 1 (%v vs %v) — steady-state cache work got slower the longer it ran; suspect a handle leak or allocator-pool pathology, not load", rounds, ratio, last, first)
	}
}
