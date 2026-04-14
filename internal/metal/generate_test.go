//go:build darwin && arm64

package metal

import "testing"

func TestAttentionCacheIndexByLayer_DefaultModel_Good(t *testing.T) {
	got := attentionCacheIndexByLayer(&fakeModel{numLayers: 4}, 4, 4)
	want := []int{0, 1, 2, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4SharedOwners_Good(t *testing.T) {
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

	got := attentionCacheIndexByLayer(model, len(model.Layers), 2)
	want := []int{0, 1, 0, 1}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4PromotedOwner_Good(t *testing.T) {
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

	got := attentionCacheIndexByLayer(model, len(model.Layers), 5)
	want := []int{0, 1, 2, 3, 4, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}
