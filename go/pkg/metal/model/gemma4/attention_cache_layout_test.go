// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "testing"

// These tests pin Gemma4Model.AttentionCacheLayout — the architecture-specific
// layer→cache-index mapping for Gemma 4's shared local/global windows (shared
// owners, promoted owner). They moved here from package metal's generate_test.go
// (TestAttentionCacheIndexByLayer_Gemma4*) with the model type. The metal-side
// dispatch (attentionCacheIndexByLayer → AttentionCacheLayouter) is pinned by
// metal's model_dispatch_test.go.

func TestAttentionCacheLayout_Gemma4SharedOwners_Good(t *testing.T) {
	coverageTokens := "Gemma4SharedOwners"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
	}

	got := model.AttentionCacheLayout(len(model.Layers), 2)
	want := []int{0, 1, 0, 1}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheLayout_Gemma4PromotedOwner_Good(t *testing.T) {
	coverageTokens := "Gemma4PromotedOwner"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
		},
	}

	got := model.AttentionCacheLayout(len(model.Layers), 5)
	want := []int{0, 1, 2, 3, 4, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}
