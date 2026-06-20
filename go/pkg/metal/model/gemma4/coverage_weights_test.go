// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the pure weight-shaping helpers in weights.go. These operate over
// crafted metal arrays / weight maps and need no checkpoint. Helpers prefixed
// cov4w.

// TestWeights_RotatedDims_Good tables gemma4RotatedDims through each clamp: the
// default factor, an over-one factor clamped to headDim, an odd result rounded
// down, and a zero-collapse fallback.
func TestWeights_RotatedDims_Good(t *testing.T) {
	cases := []struct {
		name    string
		headDim int32
		factor  float32
		want    int32
	}{
		{"factor unset → full headDim", 256, 0, 256},
		{"half factor", 256, 0.5, 128},
		{"over-one factor clamps to headDim", 64, 2.0, 64},
		{"odd result rounds down to even", 10, 0.5, 4}, // round(5)=5 → 4
		{"tiny factor collapses to headDim", 8, 0.01, 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := gemma4RotatedDims(tc.headDim, RopeParams{PartialRotaryFactor: tc.factor})
			if got != tc.want {
				t.Fatalf("gemma4RotatedDims(%d, %v) = %d, want %d", tc.headDim, tc.factor, got, tc.want)
			}
		})
	}
}

// TestWeights_ProportionalFreqs_Good covers gemma4ProportionalFreqs: the zero
// guard, the full-headDim case, the factor-scaled case, and the partial-rotary
// case that pads with +Inf.
func TestWeights_ProportionalFreqs_Good(t *testing.T) {
	requireMetalRuntime(t)

	if gemma4ProportionalFreqs(8, 0, 10000, 1) != nil {
		t.Fatal("rotatedDims 0 should return nil")
	}

	full := gemma4ProportionalFreqs(8, 8, 10000, 1)
	if full == nil {
		t.Fatal("full rotary should return freqs")
	}
	metal.Materialize(full)
	if full.Shape()[0] != 4 { // headDim/2 entries
		t.Fatalf("full freqs len = %d, want 4", full.Shape()[0])
	}
	metal.Free(full)

	scaled := gemma4ProportionalFreqs(8, 8, 10000, 0.5)
	if scaled == nil {
		t.Fatal("factor-scaled freqs should be non-nil")
	}
	metal.Free(scaled)

	// partial rotary (rotatedDims < headDim) pads with +Inf to headDim/2.
	partial := gemma4ProportionalFreqs(8, 4, 10000, 1)
	if partial == nil {
		t.Fatal("partial rotary freqs should be non-nil")
	}
	metal.Materialize(partial)
	host := partial.Floats()
	if len(host) != 4 {
		t.Fatalf("partial freqs len = %d, want 4 (headDim/2)", len(host))
	}
	if !math.IsInf(float64(host[len(host)-1]), 1) {
		t.Fatalf("partial freqs tail = %v, want +Inf padding", host[len(host)-1])
	}
	metal.Free(partial)
}

// TestWeights_SplitGateUpArray_Good covers splitGemma4GateUpArray: nil, the
// rank-2 expert split, the odd-dim reject, and the empty-shape guard.
func TestWeights_SplitGateUpArray_Good(t *testing.T) {
	requireMetalRuntime(t)

	if _, _, ok := splitGemma4GateUpArray(nil); ok {
		t.Fatal("nil should not split")
	}

	// rank-2 [experts, 2*hidden] splits the feature axis.
	gu := seqArray(0.1, 3, 8)
	defer metal.Free(gu)
	l, r, ok := splitGemma4GateUpArray(gu)
	if !ok {
		t.Fatal("even rank-2 should split")
	}
	defer metal.Free(l, r)
	if l.Dim(1) != 4 || r.Dim(1) != 4 {
		t.Fatalf("split halves = %d/%d, want 4/4", l.Dim(1), r.Dim(1))
	}

	// odd feature dim rejected.
	odd := seqArray(0.1, 3, 7)
	defer metal.Free(odd)
	if _, _, ok := splitGemma4GateUpArray(odd); ok {
		t.Fatal("odd feature dim should not split")
	}
}

// TestWeights_InferPerLayerInputSize_Good covers inferGemma4PerLayerInputSize's
// resolution order: the projection weight wins, then the gate fallback, then the
// embedding 2D/3D layouts, and the numHiddenLayers<=0 guard.
func TestWeights_InferPerLayerInputSize_Good(t *testing.T) {
	requireMetalRuntime(t)

	if got := inferGemma4PerLayerInputSize(map[string]*metal.Array{}, 0); got != 0 {
		t.Fatalf("numHiddenLayers 0 should return 0, got %d", got)
	}

	// projection weight [layers*size, hidden] wins: out=8, layers=2 → 4.
	proj := map[string]*metal.Array{
		"model.per_layer_model_projection.weight": seqArray(0.1, 8, 16),
	}
	defer freeWeightMap(proj)
	if got := inferGemma4PerLayerInputSize(proj, 2); got != 4 {
		t.Fatalf("projection infer = %d, want 4", got)
	}

	// gate weight fallback: rows = size.
	gate := map[string]*metal.Array{
		"model.layers.0.per_layer_input_gate.weight": seqArray(0.1, 5, 8),
	}
	defer freeWeightMap(gate)
	if got := inferGemma4PerLayerInputSize(gate, 2); got != 5 {
		t.Fatalf("gate infer = %d, want 5", got)
	}

	// embedding 2D [vocab, layers*size]: cols=8, layers=2 → 4.
	emb2 := map[string]*metal.Array{
		"model.embed_tokens_per_layer.weight": seqArray(0.1, 10, 8),
	}
	defer freeWeightMap(emb2)
	if got := inferGemma4PerLayerInputSize(emb2, 2); got != 4 {
		t.Fatalf("embed 2D infer = %d, want 4", got)
	}

	// embedding 3D [vocab, layers, size].
	emb3 := map[string]*metal.Array{
		"model.embed_tokens_per_layer.weight": seqArray(0.1, 10, 2, 6),
	}
	defer freeWeightMap(emb3)
	if got := inferGemma4PerLayerInputSize(emb3, 2); got != 6 {
		t.Fatalf("embed 3D infer = %d, want 6", got)
	}
}

// TestWeights_Embedding_Good covers gemma4Embedding: a plain (unquantised) table,
// a quantised table carrying scales/biases, and the absent-weight nil return.
func TestWeights_Embedding_Good(t *testing.T) {
	requireMetalRuntime(t)

	if e := gemma4Embedding(map[string]*metal.Array{}, "missing", nil); e != nil {
		t.Fatal("absent weight should give nil embedding")
	}

	plain := map[string]*metal.Array{"embed.weight": seqArray(0.1, 10, 8)}
	defer freeWeightMap(plain)
	e := gemma4Embedding(plain, "embed", nil)
	if e == nil || e.Weight == nil {
		t.Fatal("plain embedding should load")
	}
	if e.Scales != nil {
		t.Fatal("plain embedding should carry no scales")
	}

	// quantised table: pack a [10,32] embedding at gs=32/bits=4.
	dense := seqArray(0.1, 10, 32)
	wq, s, b, err := metal.Quantize(dense, 32, 4, "affine")
	if err != nil {
		t.Fatalf("Quantize embedding: %v", err)
	}
	metal.Materialize(wq, s, b)
	metal.Free(dense)
	q := map[string]*metal.Array{"embed.weight": wq, "embed.scales": s, "embed.biases": b}
	defer freeWeightMap(q)
	qe := gemma4Embedding(q, "embed", &metal.QuantizationConfig{GroupSize: 32, Bits: 4, Mode: "affine"})
	if qe == nil || qe.Scales == nil || qe.Bits != 4 {
		t.Fatalf("quantised embedding not wired: %+v", qe)
	}
}

// TestWeights_OutputLinear_Good covers gemma4OutputLinear: an explicit lm_head,
// the tied-embedding fallback, and the untied-missing-lm_head error.
func TestWeights_OutputLinear_Good(t *testing.T) {
	requireMetalRuntime(t)

	// explicit lm_head wins.
	headW := map[string]*metal.Array{"lm_head.weight": seqArray(0.1, 10, 8)}
	defer freeWeightMap(headW)
	out, err := gemma4OutputLinear(headW, &Gemma4TextConfig{}, nil)
	if err != nil || out == nil {
		t.Fatalf("explicit lm_head should load: %v", err)
	}

	// tied fallback uses the embedding.
	embW := seqArray(0.2, 10, 8)
	defer metal.Free(embW)
	embed := &metal.Embedding{Weight: embW}
	tied, err := gemma4OutputLinear(map[string]*metal.Array{}, &Gemma4TextConfig{TieWordEmbeddings: true}, embed)
	if err != nil || tied == nil {
		t.Fatalf("tied output should fall back to embedding: %v", err)
	}

	// untied + missing lm_head → error.
	if _, err := gemma4OutputLinear(map[string]*metal.Array{}, &Gemma4TextConfig{TieWordEmbeddings: false}, embed); err == nil {
		t.Fatal("untied output with no lm_head should error")
	}
}

// TestWeights_FreeUnusedWeights_Good covers gemma4FreeUnusedWeights: retained
// arrays survive, unretained are freed, and an array appearing twice is freed
// once (the dedup map).
func TestWeights_FreeUnusedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	keep := seqArray(0.1, 4)
	drop := seqArray(0.2, 4)
	weights := map[string]*metal.Array{
		"keep":      keep,
		"drop":      drop,
		"drop_dup":  drop, // same handle twice → freed once
		"keep_also": keep,
	}
	retained := map[*metal.Array]struct{}{keep: {}}
	gemma4FreeUnusedWeights(weights, retained)
	if !keep.Valid() {
		t.Fatal("retained array should survive")
	}
	if drop.Valid() {
		t.Fatal("unretained array should be freed")
	}
	metal.Free(keep)
}

// TestWeights_RetainedMapHelpers_Good covers gemma4LazyRetainedWeights and
// gemma4MaterializableRetainedWeights: the nil-model guard, lazy exclusion, and
// the invalid-array skip.
func TestWeights_RetainedMapHelpers_Good(t *testing.T) {
	requireMetalRuntime(t)

	// nil model → empty lazy set.
	if got := gemma4LazyRetainedWeights(nil); len(got) != 0 {
		t.Fatal("nil model lazy set should be empty")
	}

	// a model with a per-layer embedding tracks it as lazy.
	embW := seqArray(0.1, 10, 4)
	defer metal.Free(embW)
	m := &Gemma4Model{EmbedTokensPerLayer: &metal.Embedding{Weight: embW}}
	lazy := gemma4LazyRetainedWeights(m)
	if _, ok := lazy[embW]; !ok {
		t.Fatal("per-layer embedding weight should be lazy-tracked")
	}

	// materializable excludes lazy entries.
	a := seqArray(0.2, 4)
	defer metal.Free(a)
	retained := map[*metal.Array]struct{}{a: {}, embW: {}}
	mat := gemma4MaterializableRetainedWeights(retained, lazy)
	if !arraySliceContains(mat, a) {
		t.Fatal("non-lazy retained array should be materializable")
	}
	if arraySliceContains(mat, embW) {
		t.Fatal("lazy array should be excluded from materializable set")
	}
}
