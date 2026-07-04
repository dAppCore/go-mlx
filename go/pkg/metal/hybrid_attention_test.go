// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestBuildHybridAttentionCachePlan_ExpandsPattern_Good(t *testing.T) {
	plan, err := BuildHybridAttentionCachePlan(6, []string{"linear-attention", "full_attention"}, 1024)
	if err != nil {
		t.Fatalf("BuildHybridAttentionCachePlan() error = %v", err)
	}
	if len(plan.Layers) != 6 || plan.CachelessLayers != 3 || plan.GlobalLayers != 3 {
		t.Fatalf("plan = %+v, want 3 linear and 3 full layers", plan)
	}
	wantCacheIndex := []int{-1, 0, -1, 1, -1, 2}
	for i, layer := range plan.Layers {
		wantKind := HybridAttentionLinear
		wantKV := false
		wantWindow := 0
		wantLayerCacheIndex := -1
		if i%2 == 1 {
			wantKind = HybridAttentionFull
			wantKV = true
			wantWindow = 1024
			wantLayerCacheIndex = i / 2
		}
		if layer.Layer != i || layer.Kind != wantKind || layer.RequiresKV != wantKV || layer.Window != wantWindow || layer.CacheIndex != wantLayerCacheIndex {
			t.Fatalf("layer[%d] = %+v, want kind=%s kv=%v window=%d cache=%d", i, layer, wantKind, wantKV, wantWindow, wantLayerCacheIndex)
		}
		if plan.CacheIndexByLayer[i] != wantCacheIndex[i] {
			t.Fatalf("CacheIndexByLayer[%d] = %d, want %d", i, plan.CacheIndexByLayer[i], wantCacheIndex[i])
		}
	}
}

func TestBuildHybridAttentionCachePlan_Validation_Bad(t *testing.T) {
	cases := []struct {
		name       string
		numLayers  int
		layerTypes []string
		want       string
	}{
		{name: "missing-layers", numLayers: 0, layerTypes: []string{"linear_attention", "full_attention"}, want: "positive layer count"},
		{name: "missing-layer-types", numLayers: 2, want: "linear_attention"},
		{name: "missing-linear", numLayers: 2, layerTypes: []string{"full_attention"}, want: "linear_attention"},
		{name: "missing-full", numLayers: 2, layerTypes: []string{"linear_attention"}, want: "full_attention"},
		{name: "unknown", numLayers: 2, layerTypes: []string{"linear_attention", "mystery_attention"}, want: "unsupported layer type"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildHybridAttentionCachePlan(tc.numLayers, tc.layerTypes, 0)
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestParseHybridAttentionKind_Ugly(t *testing.T) {
	cases := map[string]string{
		"linear":           HybridAttentionLinear,
		"linear.attention": HybridAttentionLinear,
		"global-attention": HybridAttentionFull,
		"full":             HybridAttentionFull,
	}
	for input, want := range cases {
		got, ok := ParseHybridAttentionKind(input)
		if !ok || got != want {
			t.Fatalf("ParseHybridAttentionKind(%q) = %q ok=%v, want %q", input, got, ok, want)
		}
	}
	if got, ok := ParseHybridAttentionKind("banana"); ok || got != "" {
		t.Fatalf("ParseHybridAttentionKind(banana) = %q ok=%v, want unsupported", got, ok)
	}
}
