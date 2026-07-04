// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the pure assistant (MTP drafter) validators and small accessors.
// These operate over synthetic Gemma4AssistantModel / Gemma4Model structs and
// crafted metal.Linear shapes — no checkpoint load. Helpers prefixed cov4av.

// TestAssistant_LinearShape_Good walks validateGemma4AssistantLinearShape: a nil
// linear is accepted, a rank-1 weight is rejected, a wrong output dim is named,
// and a wrong input dim is named; the happy match passes.
func TestAssistant_LinearShape_Good(t *testing.T) {
	requireMetalRuntime(t)

	// nil linear → no error.
	if err := validateGemma4AssistantLinearShape("p", nil, 8, 8); err != nil {
		t.Fatalf("nil linear should pass, got %v", err)
	}

	// rank-1 weight → invalid rank.
	bad := &metal.Linear{Weight: seqArray(0.1, 8)}
	defer metal.FreeLinear(bad)
	if err := validateGemma4AssistantLinearShape("p", bad, 8, 8); err == nil {
		t.Fatal("rank-1 weight should be rejected")
	}

	// exact [out,in] match passes.
	good := &metal.Linear{Weight: seqArray(0.1, 8, 4)}
	defer metal.FreeLinear(good)
	if err := validateGemma4AssistantLinearShape("p", good, 8, 4); err != nil {
		t.Fatalf("matching shape should pass, got %v", err)
	}

	// wrong output dim named.
	if err := validateGemma4AssistantLinearShape("p", good, 16, 4); err == nil {
		t.Fatal("wrong output dim should be rejected")
	}

	// wrong input dim named.
	if err := validateGemma4AssistantLinearShape("p", good, 8, 16); err == nil {
		t.Fatal("wrong input dim should be rejected")
	}
}

// TestAssistant_ProjectionShapes_Good drives validateGemma4AssistantProjectionShapes
// across the pre/post projections and the ordered-embedding centroids branch.
func TestAssistant_ProjectionShapes_Good(t *testing.T) {
	requireMetalRuntime(t)

	// nil model / cfg → no error.
	if err := validateGemma4AssistantProjectionShapes(nil); err != nil {
		t.Fatalf("nil model should pass, got %v", err)
	}

	const hidden, backbone = 16, 8
	m := &Gemma4AssistantModel{
		Cfg:                &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: hidden}},
		BackboneHiddenSize: backbone,
		PreProjection:      &metal.Linear{Weight: seqArray(0.1, hidden, backbone*2)},
		PostProjection:     &metal.Linear{Weight: seqArray(0.2, backbone, hidden)},
	}
	defer metal.FreeLinear(m.PreProjection)
	defer metal.FreeLinear(m.PostProjection)
	if err := validateGemma4AssistantProjectionShapes(m); err != nil {
		t.Fatalf("valid projections should pass, got %v", err)
	}

	// ordered-embedding centroids checked when enabled.
	m.UseOrderedEmbeddings = true
	m.NumCentroids = 4
	m.MaskedCentroids = &metal.Linear{Weight: seqArray(0.3, 4, hidden)}
	defer metal.FreeLinear(m.MaskedCentroids)
	if err := validateGemma4AssistantProjectionShapes(m); err != nil {
		t.Fatalf("valid centroids should pass, got %v", err)
	}

	// a wrong pre-projection output dim fails.
	bad := *m
	bad.PreProjection = &metal.Linear{Weight: seqArray(0.1, hidden+1, backbone*2)}
	defer metal.FreeLinear(bad.PreProjection)
	if err := validateGemma4AssistantProjectionShapes(&bad); err == nil {
		t.Fatal("wrong pre-projection shape should fail")
	}
}

// TestAssistant_OrderedEmbeddingShape_Good covers validateGemma4AssistantOrderedEmbeddingShape:
// the disabled fast-return, a wrong dtype, a non-divisible vocab, the [vocab] and
// [centroids,perCentroid] valid layouts, and a wrong shape.
func TestAssistant_OrderedEmbeddingShape_Good(t *testing.T) {
	requireMetalRuntime(t)

	// disabled (no ordered embeddings) passes.
	if err := validateGemma4AssistantOrderedEmbeddingShape(&Gemma4AssistantModel{}); err != nil {
		t.Fatalf("disabled should pass, got %v", err)
	}

	base := func(ordering *metal.Array) *Gemma4AssistantModel {
		return &Gemma4AssistantModel{
			Cfg:                  &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 8}},
			NumCentroids:         4,
			UseOrderedEmbeddings: true,
			TokenOrdering:        ordering,
		}
	}

	// wrong dtype (float) rejected.
	fl := seqArray(0.1, 8)
	defer metal.Free(fl)
	if err := validateGemma4AssistantOrderedEmbeddingShape(base(fl)); err == nil {
		t.Fatal("float token_ordering dtype should be rejected")
	}

	// flat [vocab] int32 passes.
	flat := metal.FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7}, 8)
	defer metal.Free(flat)
	if err := validateGemma4AssistantOrderedEmbeddingShape(base(flat)); err != nil {
		t.Fatalf("flat [vocab] ordering should pass, got %v", err)
	}

	// [centroids, perCentroid] = [4,2] passes.
	grid := metal.FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7}, 4, 2)
	defer metal.Free(grid)
	if err := validateGemma4AssistantOrderedEmbeddingShape(base(grid)); err != nil {
		t.Fatalf("[centroids,perCentroid] ordering should pass, got %v", err)
	}

	// a wrong shape ([3]) is rejected.
	wrong := metal.FromValues([]int32{0, 1, 2}, 3)
	defer metal.Free(wrong)
	if err := validateGemma4AssistantOrderedEmbeddingShape(base(wrong)); err == nil {
		t.Fatal("wrong-shaped ordering should be rejected")
	}

	// non-divisible vocab/centroids rejected.
	nd := base(flat)
	nd.NumCentroids = 3 // 8 % 3 != 0
	if err := validateGemma4AssistantOrderedEmbeddingShape(nd); err == nil {
		t.Fatal("non-divisible vocab/centroids should be rejected")
	}
}

// TestAssistant_PairValidators_Good walks validateGemma4AssistantPair's error
// ladder plus the target-layer-type / head-dim / slice-equal helpers.
func TestAssistant_PairValidators_Good(t *testing.T) {
	requireMetalRuntime(t)

	// nil target.
	if err := validateGemma4AssistantPair(nil, &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{}}); err == nil {
		t.Fatal("nil target should fail")
	}
	// nil assistant.
	tgt := &Gemma4Model{Cfg: &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: 8, VocabSize: 10}}}
	if err := validateGemma4AssistantPair(tgt, nil); err == nil {
		t.Fatal("nil assistant should fail")
	}
	// invalid target hidden size.
	badHidden := &Gemma4Model{Cfg: &Gemma4TextConfig{}}
	if err := validateGemma4AssistantPair(badHidden, &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{}}); err == nil {
		t.Fatal("zero target hidden_size should fail")
	}
	// backbone mismatch.
	asst := &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 10}}, BackboneHiddenSize: 4}
	if err := validateGemma4AssistantPair(tgt, asst); err == nil {
		t.Fatal("backbone/hidden mismatch should fail")
	}
	// vocab mismatch.
	asst2 := &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 99}}, BackboneHiddenSize: 8}
	if err := validateGemma4AssistantPair(tgt, asst2); err == nil {
		t.Fatal("vocab mismatch should fail")
	}
	// nil tokenizers (matched dims but no Tok).
	asst3 := &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 10}}, BackboneHiddenSize: 8}
	if err := validateGemma4AssistantPair(tgt, asst3); err == nil {
		t.Fatal("nil tokenizers should fail")
	}

	// gemma4TargetLayerTypes from cfg + layers.
	tgt.Cfg.LayerTypes = []string{"sliding_attention", "full_attention"}
	types := gemma4TargetLayerTypes(tgt)
	if !types["sliding_attention"] || !types["full_attention"] {
		t.Fatalf("target layer types = %v, want both", types)
	}
	if got := gemma4TargetLayerTypes(nil); len(got) != 0 {
		t.Fatal("nil target should give empty type set")
	}

	// gemma4TargetHeadDimForLayerType: global head dim for full, head dim for sliding.
	tgt.Cfg.GlobalHeadDim = 512
	tgt.Cfg.HeadDim = 256
	if got := gemma4TargetHeadDimForLayerType(tgt.Cfg, "full_attention"); got != 512 {
		t.Fatalf("full head dim = %d, want 512", got)
	}
	if got := gemma4TargetHeadDimForLayerType(tgt.Cfg, "sliding_attention"); got != 256 {
		t.Fatalf("sliding head dim = %d, want 256", got)
	}
	if got := gemma4TargetHeadDimForLayerType(nil, "full_attention"); got != 0 {
		t.Fatal("nil cfg head dim should be 0")
	}

	// gemma4AssistantInt32SlicesEqual.
	if !gemma4AssistantInt32SlicesEqual([]int32{1, 2, 3}, []int32{1, 2, 3}) {
		t.Fatal("equal slices should compare equal")
	}
	if gemma4AssistantInt32SlicesEqual([]int32{1, 2}, []int32{1, 2, 3}) {
		t.Fatal("different lengths should not be equal")
	}
	if gemma4AssistantInt32SlicesEqual([]int32{1, 2, 3}, []int32{1, 9, 3}) {
		t.Fatal("differing elements should not be equal")
	}
}

// TestAssistant_Accessors_Good covers embeddingWeight, NumLayers and Tokenizer,
// including the nil-receiver branches.
func TestAssistant_Accessors_Good(t *testing.T) {
	requireMetalRuntime(t)

	// embeddingWeight: nil and non-nil.
	if embeddingWeight(nil) != nil {
		t.Fatal("nil embedding should give nil weight")
	}
	w := seqArray(0.1, 4, 4)
	defer metal.Free(w)
	if embeddingWeight(&metal.Embedding{Weight: w}) != w {
		t.Fatal("embeddingWeight should return the table weight")
	}

	// nil-receiver accessors.
	var nilM *Gemma4AssistantModel
	if nilM.NumLayers() != 0 {
		t.Fatal("nil model NumLayers should be 0")
	}
	if nilM.Tokenizer() != nil {
		t.Fatal("nil model Tokenizer should be nil")
	}

	// populated accessors.
	m := &Gemma4AssistantModel{Layers: make([]*Gemma4AssistantLayer, 3)}
	if m.NumLayers() != 3 {
		t.Fatalf("NumLayers = %d, want 3", m.NumLayers())
	}
	if m.Tokenizer() != nil {
		t.Fatal("Tokenizer with no Tok should be nil")
	}
}
