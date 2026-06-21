// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	model "dappco.re/go/mlx/pkg/model"
)

// TestDeriveLayersParity gates the declarative arch (pkg/model/gemma4, part three):
// the backend-agnostic DeriveLayers must reproduce the metal loader's
// buildGemma4CacheLayout — the per-layer KV-cache-sharing map — plus the layer-type
// rule, exactly. Driven by cheap layer stubs (just LayerType set), so it needs no
// model load. Covers an E2B-like pattern (35 layers, a global layer every 6th, 20
// shared) and the edges (all-sliding, all-global, no sharing, shared > layers).
func TestDeriveLayersParity(t *testing.T) {
	slide := func(n int, isSliding func(i int) bool) []string {
		ts := make([]string, n)
		for i := range ts {
			if isSliding(i) {
				ts[i] = "sliding_attention"
			} else {
				ts[i] = "full_attention"
			}
		}
		return ts
	}
	cases := []struct {
		name   string
		types  []string
		shared int
	}{
		{"e2b-like 35L global-every-6 shared-20", slide(35, func(i int) bool { return (i+1)%6 != 0 }), 20},
		{"all sliding shared-3", slide(8, func(int) bool { return true }), 3},
		{"all global shared-2", slide(6, func(int) bool { return false }), 2},
		{"mixed shared-0 (all own)", slide(10, func(i int) bool { return (i+1)%3 != 0 }), 0},
		{"shared > layers (toy)", slide(4, func(int) bool { return true }), 10},
	}
	for _, tc := range cases {
		// metal reference: stubs carrying only LayerType
		layers := make([]*Gemma4DecoderLayer, len(tc.types))
		for i := range layers {
			layers[i] = &Gemma4DecoderLayer{LayerType: tc.types[i]}
		}
		prevM, ciM := buildGemma4CacheLayout(layers, int32(tc.shared))

		specs := model.DeriveLayers(tc.types, tc.shared)
		if len(specs) != len(tc.types) {
			t.Fatalf("%s: DeriveLayers len %d != %d", tc.name, len(specs), len(tc.types))
		}
		for i := range tc.types {
			if int32(specs[i].KVShareFrom) != prevM[i] {
				t.Fatalf("%s layer %d: KVShareFrom %d != metal previous %d", tc.name, i, specs[i].KVShareFrom, prevM[i])
			}
			if int32(specs[i].CacheIndex) != ciM[i] {
				t.Fatalf("%s layer %d: CacheIndex %d != metal %d", tc.name, i, specs[i].CacheIndex, ciM[i])
			}
			if (specs[i].Attention == model.SlidingAttention) != (tc.types[i] == "sliding_attention") {
				t.Fatalf("%s layer %d: Attention %v mismatches layer_type %q", tc.name, i, specs[i].Attention, tc.types[i])
			}
		}
		t.Logf("%s: DeriveLayers ≡ buildGemma4CacheLayout (%d layers, %d shared)", tc.name, len(tc.types), tc.shared)
	}
}
