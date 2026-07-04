// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// makeDistinctKV builds a [1,H,seqLen,D] K/V pair where each (head, token) row
// is a distinct constant vector, so a compaction that keeps a subset of tokens
// produces a state that is byte-identifiable against the original rows.
func makeDistinctKV(heads, seqLen, dim int) (*Array, *Array) {
	size := 1 * heads * seqLen * dim
	kData := make([]float32, size)
	vData := make([]float32, size)
	for h := 0; h < heads; h++ {
		for t := 0; t < seqLen; t++ {
			for d := 0; d < dim; d++ {
				idx := ((h*seqLen)+t)*dim + d
				kData[idx] = float32(t) + float32(h)*0.001
				vData[idx] = float32(t)*10 + float32(h)*0.001
			}
		}
	}
	k := FromValues(kData, 1, heads, seqLen, dim)
	v := FromValues(vData, 1, heads, seqLen, dim)
	return k, v
}

// --- scheme resolution (the task gate) ---

// TestCacheScheme_TurboQuantResolves_Good — the turboquant scheme resolves from
// the registry and carries the metal compute surface serving a KV cache.
func TestCacheScheme_TurboQuantResolves_Good(t *testing.T) {
	cc, ok := CacheComputeFor("turboquant")
	if !ok {
		t.Fatal("turboquant cache compute did not resolve")
	}
	if cc.Mode() != "turboquant" {
		t.Errorf("mode = %q, want turboquant", cc.Mode())
	}
	if cc.Serves() != scheme.StateKVCache {
		t.Errorf("serves = %v, want kv-cache", cc.Serves())
	}
	c := cc.NewCache(CacheParams{MaxSize: 4096, PageSize: 2048})
	if c == nil {
		t.Fatal("NewCache returned nil")
	}
	if _, isTurbo := c.(*TurboQuantKVCache); !isTurbo {
		t.Errorf("NewCache returned %T, want *TurboQuantKVCache", c)
	}
}

// TestCacheScheme_CompactionResolves_Good — the compaction scheme resolves and
// carries the metal compute surface serving a KV cache.
func TestCacheScheme_CompactionResolves_Good(t *testing.T) {
	cc, ok := CacheComputeFor("compaction")
	if !ok {
		t.Fatal("compaction cache compute did not resolve")
	}
	if cc.Mode() != "compaction" {
		t.Errorf("mode = %q, want compaction", cc.Mode())
	}
	if cc.Serves() != scheme.StateKVCache {
		t.Errorf("serves = %v, want kv-cache", cc.Serves())
	}
	c := cc.NewCache(CacheParams{MaxSize: 64})
	if c == nil {
		t.Fatal("NewCache returned nil")
	}
	comp, isComp := c.(*CompactionKVCache)
	if !isComp {
		t.Fatalf("NewCache returned %T, want *CompactionKVCache", c)
	}
	if comp.Budget() != 64 {
		t.Errorf("budget = %d, want 64", comp.Budget())
	}
}

// --- TurboQuant round-trip ---

// TestTurboQuantScheme_RoundTripPreservesShape_Good — a TurboQuant cache built
// through the scheme accepts an Update and reports state of the expected shape.
func TestTurboQuantScheme_RoundTripPreservesShape_Good(t *testing.T) {
	cc, _ := CacheComputeFor("turboquant")
	c := cc.NewCache(CacheParams{MaxSize: 0, PageSize: 8})
	k, v := makeKV(4) // [1,2,4,4]
	defer Free(k, v)

	outK, outV := c.Update(k, v, 4)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval turboquant update: %v", err)
	}
	if c.Len() != 4 {
		t.Errorf("len = %d, want 4", c.Len())
	}
	state := c.State()
	if len(state) < 2 || state[0] == nil {
		t.Fatalf("state = %v, want non-nil K/V", state)
	}
	if got := state[0].Dim(2); got != 4 {
		t.Errorf("state K seq dim = %d, want 4", got)
	}
}

// --- compaction round-trip + size reduction (the task gate) ---

// TestCompactionScheme_ReducesCacheSize_Good — once the held window crosses the
// budget, the compaction cache holds exactly `budget` tokens (≈10% of the
// original) while still returning a usable, correctly-shaped K/V.
func TestCompactionScheme_ReducesCacheSize_Good(t *testing.T) {
	const heads, dim, budget = 2, 8, 8
	cc, _ := CacheComputeFor("compaction")
	c := cc.NewCache(CacheParams{MaxSize: budget})

	// Prompt of 80 tokens — 10x the budget.
	k, v := makeDistinctKV(heads, 80, dim)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 80)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval compaction update: %v", err)
	}
	if got := c.(*CompactionKVCache).Err(); got != nil {
		t.Fatalf("compaction error: %v", got)
	}

	// Offset is the monotonic token count; Len is the compacted size.
	if c.Offset() != 80 {
		t.Errorf("offset = %d, want 80", c.Offset())
	}
	if c.Len() != budget {
		t.Errorf("len = %d, want compacted to %d", c.Len(), budget)
	}

	state := c.State()
	if len(state) < 2 || state[0] == nil || state[1] == nil {
		t.Fatalf("state = %v, want non-nil K/V", state)
	}
	if err := Eval(state[0], state[1]); err != nil {
		t.Fatalf("Eval compacted state: %v", err)
	}
	if got := state[0].Dim(2); got != budget {
		t.Errorf("compacted K seq dim = %d, want %d", got, budget)
	}
	if got := state[1].Dim(2); got != budget {
		t.Errorf("compacted V seq dim = %d, want %d", got, budget)
	}
	// Heads and head-dim are preserved by compaction (only the seq axis shrinks).
	if state[0].Dim(1) != heads || state[0].Dim(3) != dim {
		t.Errorf("compacted K shape = [%d,%d,%d,%d], want heads=%d dim=%d",
			state[0].Dim(0), state[0].Dim(1), state[0].Dim(2), state[0].Dim(3), heads, dim)
	}
}

// TestCompactionScheme_BelowBudgetRoundTripsExactly_Good — a window that never
// crosses the budget is held uncompacted, so Len tracks the token count exactly.
func TestCompactionScheme_BelowBudgetRoundTripsExactly_Good(t *testing.T) {
	cc, _ := CacheComputeFor("compaction")
	c := cc.NewCache(CacheParams{MaxSize: 64})

	k, v := makeKV(5)
	defer Free(k, v)
	outK, outV := c.Update(k, v, 5)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	// Single-token decode step.
	k2, v2 := makeSingleTokenKV(1.0)
	defer Free(k2, v2)
	o2K, o2V := c.Update(k2, v2, 1)
	if err := Eval(o2K, o2V); err != nil {
		t.Fatalf("Eval decode: %v", err)
	}

	if c.Len() != 6 {
		t.Errorf("len = %d, want 6 (below budget, uncompacted)", c.Len())
	}
	state := c.State()
	if got := state[0].Dim(2); got != 6 {
		t.Errorf("state K seq dim = %d, want 6", got)
	}
}

// TestCompactionScheme_ForceCompact_Good — Compact() reclaims the window down to
// the budget on demand, and reports false when there is nothing to compact.
func TestCompactionScheme_ForceCompact_Good(t *testing.T) {
	c := NewCompactionKVCache(8, 0)
	k, v := makeDistinctKV(1, 40, 8)
	defer Free(k, v)
	outK, outV := c.Update(k, v, 40)
	_ = Eval(outK, outV)

	// The 40-token window already crossed the budget during Update.
	if c.Len() != 8 {
		t.Fatalf("len after over-budget update = %d, want 8", c.Len())
	}
	// A second Compact with nothing over budget reports false.
	if c.Compact() {
		t.Error("Compact() = true on an at-budget cache, want false")
	}
	if c.Err() != nil {
		t.Errorf("unexpected error: %v", c.Err())
	}
}

// TestCompactionScheme_Reset_Good — Reset clears the cache for reuse.
func TestCompactionScheme_Reset_Good(t *testing.T) {
	c := NewCompactionKVCache(8, 0)
	k, v := makeDistinctKV(1, 20, 4)
	defer Free(k, v)
	c.Update(k, v, 20)

	c.Reset()
	if c.Offset() != 0 || c.Len() != 0 {
		t.Errorf("after reset offset=%d len=%d, want 0/0", c.Offset(), c.Len())
	}
	if c.State() != nil {
		t.Error("state should be nil after reset")
	}
}

// --- Bad / Ugly ---

// TestCompactionScheme_InvalidArrays_Bad — invalid K/V is rejected via Err and
// the cache stays empty rather than panicking.
func TestCompactionScheme_InvalidArrays_Bad(t *testing.T) {
	c := NewCompactionKVCache(8, 0)
	var nilArray *Array
	c.Update(nilArray, nilArray, 1)
	if c.Err() == nil {
		t.Error("expected error for nil arrays, got nil")
	}
	if c.Len() != 0 {
		t.Errorf("len = %d after invalid update, want 0", c.Len())
	}
}

// TestCompactionScheme_BudgetOfOne_Ugly — a degenerate budget of one still
// produces a valid single-token compacted state from a long window.
func TestCompactionScheme_BudgetOfOne_Ugly(t *testing.T) {
	c := NewCompactionKVCache(1, 0)
	k, v := makeDistinctKV(2, 50, 8)
	defer Free(k, v)
	outK, outV := c.Update(k, v, 50)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval budget-1 update: %v", err)
	}
	if c.Err() != nil {
		t.Fatalf("error: %v", c.Err())
	}
	if c.Len() != 1 {
		t.Errorf("len = %d, want compacted to 1", c.Len())
	}
	state := c.State()
	if got := state[0].Dim(2); got != 1 {
		t.Errorf("compacted K seq dim = %d, want 1", got)
	}
}

// TestCompactionScheme_MetadataModeStillResolves_Ugly — the scheme catalogue in
// pkg/scheme also lists compaction as metadata; resolving the metal compute must
// win (the same Mode overwrites). A round-trip through both confirms they agree
// on the StateKind.
func TestCompactionScheme_MetadataModeStillResolves_Ugly(t *testing.T) {
	meta, ok := scheme.CacheFor("compaction")
	if !ok {
		t.Fatal("compaction not in scheme catalogue")
	}
	cc, ok := CacheComputeFor("compaction")
	if !ok {
		t.Fatal("compaction has no metal compute")
	}
	if meta.Serves() != cc.Serves() {
		t.Errorf("catalogue serves %v but compute serves %v", meta.Serves(), cc.Serves())
	}
}

// --- QA-eval parity (the reassigned gate) ---

// hostAttention computes single-head attention output softmax(q·Kᵀ/√D)·V on the
// host from a [1,1,L,D] cache and a [1,1,n,D] query batch, returning the n×D
// output rows. It is an independent reference — it does NOT call the MLX path
// the cache uses — so comparing it against the compacted-cache output is a true
// parity check rather than a tautology.
func hostAttention(qBatch, kBatch, vBatch *Array) [][]float32 {
	n, d := qBatch.Dim(2), qBatch.Dim(3)
	L := kBatch.Dim(2)
	q := qBatch.Floats()
	k := kBatch.Floats()
	v := vBatch.Floats()
	invSqrtD := 1.0 / math.Sqrt(float64(d))
	out := make([][]float32, n)
	for qi := 0; qi < n; qi++ {
		scores := make([]float64, L)
		maxScore := math.Inf(-1)
		for ki := 0; ki < L; ki++ {
			var dot float64
			for di := 0; di < d; di++ {
				dot += float64(q[qi*d+di]) * float64(k[ki*d+di])
			}
			dot *= invSqrtD
			scores[ki] = dot
			if dot > maxScore {
				maxScore = dot
			}
		}
		var sumExp float64
		for ki := 0; ki < L; ki++ {
			scores[ki] = math.Exp(scores[ki] - maxScore)
			sumExp += scores[ki]
		}
		row := make([]float32, d)
		for ki := 0; ki < L; ki++ {
			w := scores[ki] / sumExp
			for di := 0; di < d; di++ {
				row[di] += float32(w * float64(v[ki*d+di]))
			}
		}
		out[qi] = row
	}
	return out
}

func relL2(ref, got []float32) float64 {
	var diffSq, refSq float64
	for i := range ref {
		d := float64(ref[i] - got[i])
		diffSq += d * d
		refSq += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(diffSq) / (math.Sqrt(refSq) + 1e-10)
}

// TestCompactionScheme_AttentionParity_Good — the QA-eval parity property: a
// compacted cache reproduces the original attention OUTPUT, not just its shape.
// The window is built so attention mass concentrates on the `hot` keys; the
// compacted output is then compared (independent host computation, hostAttention)
// against the full-cache output for probe queries aligned to the hot keys, and
// must match within a relative-L2 tolerance.
//
// Self-attention-proxy caveat: with no separate query stream, selection scores
// keys by softmax(K·Kᵀ) — keys attending to keys — so a key that strongly
// self-attends can be picked even if probe queries ignore it. Here the filler
// keys self-attend (shared repel direction), so the top-`budget` set is the 6
// hot keys plus ~2 filler keys. That is fine for parity because those filler
// keys carry ~0 weight under the probe queries (they repel) and benign values,
// so the reproduced output still tracks the full output. The tolerance reflects
// this real fidelity rather than an idealised exact-match.
func TestCompactionScheme_AttentionParity_Good(t *testing.T) {
	// dims 0..5 are the "hot" query/key directions; dim 6 is a shared positive
	// bias every probe query carries, and dim 7 is the filler-repel axis. Hot
	// keys lean into dim 6 (so they score positively under every query); filler
	// keys lean hard into dim 7 with a LARGE NEGATIVE coefficient while every
	// query carries a small POSITIVE dim-7 lean — so filler scores are strongly
	// negative and exp() drives their softmax mass to ~0. This is the regime
	// compaction targets: attention concentrated on a small key subset. (The
	// earlier "filler at score 0" window was incompressible — 60 keys at exp(0)
	// out-massed the hot keys; that is a property of the window, not the
	// algorithm, so the window, not the selector, was the bug.)
	const dim, hot, filler, budget = 8, 6, 58, 8
	L := hot + filler
	const biasDim, repelDim = 6, 7

	kData := make([]float32, L*dim)
	vData := make([]float32, L*dim)
	for ki := 0; ki < L; ki++ {
		if ki < hot {
			kData[ki*dim+(ki%biasDim)] = 6.0 // strong key in one query direction
			kData[ki*dim+biasDim] = 2.0      // shared positive lean → always in-play
			for di := 0; di < dim; di++ {    // distinctive value per hot key
				vData[ki*dim+di] = float32(ki+1) + float32(di)*0.01
			}
		} else {
			kData[ki*dim+repelDim] = -20.0 // strong repel: query's +dim7 lean → big negative score
			for di := 0; di < dim; di++ {
				vData[ki*dim+di] = 0.5 // filler values near the value range, not outliers
			}
		}
	}
	k := FromValues(kData, 1, 1, L, dim)
	v := FromValues(vData, 1, 1, L, dim)
	defer Free(k, v)

	// Probe queries: aligned to each hot key direction, plus the shared dim-6
	// bias and a small dim-7 lean that makes the filler-repel bite.
	qData := make([]float32, hot*dim)
	for qi := 0; qi < hot; qi++ {
		qData[qi*dim+(qi%biasDim)] = 1.0
		qData[qi*dim+biasDim] = 1.0
		qData[qi*dim+repelDim] = 1.0
	}
	qBatch := FromValues(qData, 1, 1, hot, dim)
	defer Free(qBatch)

	// Reference: full-cache attention output (independent host computation).
	refOut := hostAttention(qBatch, k, v)

	// Compact the window, then compute attention against the compacted K/V.
	cmpK, cmpV, err := compactAttentionMatching(k, v, budget)
	if err != nil {
		t.Fatalf("compactAttentionMatching: %v", err)
	}
	defer Free(cmpK, cmpV)
	if err := Eval(cmpK, cmpV); err != nil {
		t.Fatalf("Eval compacted: %v", err)
	}
	if cmpK.Dim(2) != budget {
		t.Fatalf("compacted seq dim = %d, want %d", cmpK.Dim(2), budget)
	}
	compOut := hostAttention(qBatch, cmpK, cmpV)

	// Parity: every probe query's output reproduces the full-cache output.
	const tol = 0.02 // 2% relative L2 — hot keys carry ~all the mass
	var worst float64
	for qi := range refOut {
		e := relL2(refOut[qi], compOut[qi])
		if e > worst {
			worst = e
		}
	}
	if worst > tol {
		t.Errorf("attention parity: worst relative L2 = %.4f, want ≤ %.2f", worst, tol)
	}
}

// --- full-AM variant (task #42) ---

// TestCacheScheme_CompactionFullResolves_Good — the compaction-full scheme
// resolves from the registry, carries the metal compute surface, and builds a
// CompactionKVCache configured for the full lsq variant.
func TestCacheScheme_CompactionFullResolves_Good(t *testing.T) {
	cc, ok := CacheComputeFor("compaction-full")
	if !ok {
		t.Fatal("compaction-full cache compute did not resolve")
	}
	if cc.Mode() != "compaction-full" {
		t.Errorf("mode = %q, want compaction-full", cc.Mode())
	}
	if cc.Serves() != scheme.StateKVCache {
		t.Errorf("serves = %v, want kv-cache", cc.Serves())
	}
	c := cc.NewCache(CacheParams{MaxSize: 8})
	comp, isComp := c.(*CompactionKVCache)
	if !isComp {
		t.Fatalf("NewCache returned %T, want *CompactionKVCache", c)
	}
	if comp.variant != CompactionFull {
		t.Errorf("variant = %v, want CompactionFull", comp.variant)
	}
}

// buildLossyAMWindow builds a single-head [1,1,L,D] K/V window whose attention
// spreads across MORE distinct-value key groups than the budget, so the direct
// value-take (keep budget keys, drop the rest) loses real output mass while the
// lsq fit can fold the dropped groups' contribution into the kept values. Each
// of `groups` key clusters points along its own one-hot direction with a shared
// positive bias, and carries a group-distinct value; the probe queries align to
// each group direction. Returns K, V (the cache) and the probe-query batch.
func buildLossyAMWindow(groups, perGroup, dim int) (k, v, q *Array) {
	L := groups * perGroup
	const biasDim = 0 // shared lean so every key stays in play under every query
	kData := make([]float32, L*dim)
	vData := make([]float32, L*dim)
	for g := 0; g < groups; g++ {
		dir := 1 + (g % (dim - 1)) // 1..dim-1, leaving dim 0 as the shared bias
		for p := 0; p < perGroup; p++ {
			ki := g*perGroup + p
			kData[ki*dim+dir] = 4.0
			kData[ki*dim+biasDim] = 1.0
			for d := 0; d < dim; d++ { // group-distinct value vector
				vData[ki*dim+d] = float32(g+1) + float32(d)*0.1
			}
		}
	}
	qData := make([]float32, groups*dim)
	for g := 0; g < groups; g++ {
		dir := 1 + (g % (dim - 1))
		qData[g*dim+dir] = 1.0
		qData[g*dim+biasDim] = 1.0
	}
	return FromValues(kData, 1, 1, L, dim),
		FromValues(vData, 1, 1, L, dim),
		FromValues(qData, 1, 1, groups, dim)
}

// worstRelL2 reduces the per-query reconstruction error of compacted (K,V)
// against the full-cache reference output to its worst case across queries.
func worstRelL2(refOut [][]float32, cmpK, cmpV, qBatch *Array) float64 {
	compOut := hostAttention(qBatch, cmpK, cmpV)
	var worst float64
	for qi := range refOut {
		if e := relL2(refOut[qi], compOut[qi]); e > worst {
			worst = e
		}
	}
	return worst
}

// TestCompactionFull_BeatsDirect_Good — the task gate: on a window whose
// attention spreads across more distinct-value key groups than the budget, the
// full lsq variant reconstructs the full-cache attention OUTPUT measurably
// better than the direct value-take variant. Both keep the SAME selected keys;
// the only difference is the value reconstruction, so a lower error isolates the
// lsq fit's contribution.
func TestCompactionFull_BeatsDirect_Good(t *testing.T) {
	// 10 groups, but a budget of 8 — selection must drop ≥2 value-distinct
	// groups, which the direct take loses outright.
	const groups, perGroup, dim, budget = 10, 3, 12, 8
	k, v, qBatch := buildLossyAMWindow(groups, perGroup, dim)
	defer Free(k, v, qBatch)

	// Reference: the FULL cache's attention output (independent host compute).
	refOut := hostAttention(qBatch, k, v)

	directK, directV, err := compactAttentionMatching(k, v, budget)
	if err != nil {
		t.Fatalf("direct compaction: %v", err)
	}
	defer Free(directK, directV)
	fullK, fullV, err := compactAttentionMatchingFull(k, v, budget)
	if err != nil {
		t.Fatalf("full compaction: %v", err)
	}
	defer Free(fullK, fullV)
	if err := Eval(directK, directV, fullK, fullV); err != nil {
		t.Fatalf("Eval compacted: %v", err)
	}

	// Both variants keep exactly `budget` tokens (same selection).
	if fullK.Dim(2) != budget || fullV.Dim(2) != budget {
		t.Fatalf("full compacted shape K=%v V=%v, want seq=%d", fullK.Shape(), fullV.Shape(), budget)
	}

	directErr := worstRelL2(refOut, directK, directV, qBatch)
	fullErr := worstRelL2(refOut, fullK, fullV, qBatch)

	t.Logf("worst relative-L2 — direct: %.4f, full: %.4f", directErr, fullErr)
	if fullErr >= directErr {
		t.Errorf("full-AM error %.4f not better than direct %.4f", fullErr, directErr)
	}
	// The improvement should be substantial, not a rounding-noise win: the
	// dropped groups carry real mass the lsq fit recovers.
	if directErr-fullErr < 0.01 {
		t.Errorf("full-AM improvement %.4f below the 0.01 floor (direct %.4f, full %.4f)",
			directErr-fullErr, directErr, fullErr)
	}
}

// TestCompactionFull_RoundTripsThroughCache_Good — a CompactionKVCache built for
// the full variant compacts an over-budget window and returns a usable,
// correctly-shaped K/V through the Cache contract (not just the bare function).
func TestCompactionFull_RoundTripsThroughCache_Good(t *testing.T) {
	const budget = 8
	c := NewCompactionKVCacheVariant(budget, 0, CompactionFull)
	k, v := makeDistinctKV(2, 60, 12)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 60)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval full-variant update: %v", err)
	}
	if c.Err() != nil {
		t.Fatalf("compaction error: %v", c.Err())
	}
	if c.Len() != budget {
		t.Errorf("len = %d, want compacted to %d", c.Len(), budget)
	}
	state := c.State()
	if len(state) < 2 || state[0] == nil || state[1] == nil {
		t.Fatalf("state = %v, want non-nil K/V", state)
	}
	if got := state[0].Dim(2); got != budget {
		t.Errorf("compacted K seq dim = %d, want %d", got, budget)
	}
	if state[0].Dim(1) != 2 || state[0].Dim(3) != 12 {
		t.Errorf("compacted K shape = %v, want heads=2 dim=12", state[0].Shape())
	}
}
