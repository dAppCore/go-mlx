// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestQwen36HybridAttentionPlan_ExpandsPattern_Good(t *testing.T) {
	coverageTokens := "Qwen36 HybridAttentionPlan ExpandsPattern"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	plan, err := buildQwen36HybridAttentionPlan(6, []string{"linear-attention", "full_attention"}, 4096)
	if err != nil {
		t.Fatalf("buildQwen36HybridAttentionPlan() error = %v", err)
	}
	if len(plan.Layers) != 6 || plan.LinearLayers != 3 || plan.FullLayers != 3 || plan.LocalWindow != 4096 {
		t.Fatalf("plan = %+v, want 6 layers with 3 linear and 3 full", plan)
	}
	wantCacheIndex := []int{-1, 0, -1, 1, -1, 2}
	for i, layer := range plan.Layers {
		wantKind := qwen36AttentionLinear
		wantKV := false
		wantWindow := 0
		wantLayerCacheIndex := -1
		if i%2 == 1 {
			wantKind = qwen36AttentionFull
			wantKV = true
			wantWindow = 4096
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

func TestQwen36HybridAttentionPlan_Bad(t *testing.T) {
	coverageTokens := "Qwen36 HybridAttentionPlan Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cases := []struct {
		name       string
		layerTypes []string
		want       string
	}{
		{name: "missing-linear", layerTypes: []string{"full_attention"}, want: "linear_attention"},
		{name: "missing-full", layerTypes: []string{"linear_attention"}, want: "full_attention"},
		{name: "unknown", layerTypes: []string{"linear_attention", "mystery_attention"}, want: "unsupported layer type"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := buildQwen36HybridAttentionPlan(2, tc.layerTypes, 0)
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestModel_LoadModel_Qwen36StagedLoaderBuildsHybridPlan_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_5",
		"text_config": {
			"model_type": "qwen3_5_text",
			"hidden_size": 16,
			"intermediate_size": 32,
			"num_hidden_layers": 4,
			"num_attention_heads": 4,
			"num_key_value_heads": 2,
			"head_dim": 4,
			"vocab_size": 128,
			"max_position_embeddings": 4096,
			"sliding_window": 1024,
			"layer_types": ["linear_attention", "full_attention"]
		}
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(qwen3_6) error = %v", err)
	}
	staged, ok := model.(*qwen36StagedModel)
	if !ok {
		t.Fatalf("model type = %T, want *qwen36StagedModel", model)
	}
	if staged.plan.LinearLayers != 2 || staged.plan.FullLayers != 2 || len(staged.plan.Layers) != 4 {
		t.Fatalf("plan = %+v, want 2 linear and 2 full layers", staged.plan)
	}
	if !staged.plan.Layers[1].RequiresKV || staged.plan.Layers[1].Window != 1024 {
		t.Fatalf("full layer plan = %+v, want KV with window 1024", staged.plan.Layers[1])
	}
	if staged.plan.Layers[2].RequiresKV || staged.plan.Layers[2].Window != 0 {
		t.Fatalf("linear layer plan = %+v, want no KV and zero window", staged.plan.Layers[2])
	}
	caches := staged.NewCache()
	defer FreeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCache() length = %d, want full-attention layer count 2", len(caches))
	}
	for i, cache := range caches {
		if _, ok := cache.(*KVCache); !ok {
			t.Fatalf("cache[%d] = %T, want *KVCache for full-attention layer", i, cache)
		}
	}
	if got := attentionCacheIndexByLayer(staged, staged.NumLayers(), len(caches)); got[0] != -1 || got[1] != 0 || got[2] != -1 || got[3] != 1 {
		t.Fatalf("attentionCacheIndexByLayer(qwen3_6) = %+v, want [-1 0 -1 1]", got)
	}
}

func TestModel_LoadModel_Qwen36MoEStagedLoaderBuildsHybridPlan_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["Qwen3_6MoeForConditionalGeneration"],
		"model_type": "qwen3_6_moe",
		"hidden_size": 16,
		"num_hidden_layers": 4,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"num_experts": 8,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 32,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(qwen3_6_moe) error = %v", err)
	}
	staged, ok := model.(*qwen36MoEStagedModel)
	if !ok {
		t.Fatalf("model type = %T, want *qwen36MoEStagedModel", model)
	}
	if staged.plan.LinearLayers != 2 || staged.plan.FullLayers != 2 || len(staged.plan.Layers) != 4 {
		t.Fatalf("plan = %+v, want 2 linear and 2 full layers", staged.plan)
	}
	caches := staged.NewCache()
	defer FreeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCache() length = %d, want full-attention layer count 2", len(caches))
	}
	if got := attentionCacheIndexByLayer(staged, staged.NumLayers(), len(caches)); got[0] != -1 || got[1] != 0 || got[2] != -1 || got[3] != 1 {
		t.Fatalf("attentionCacheIndexByLayer(qwen3_6_moe) = %+v, want [-1 0 -1 1]", got)
	}
}
