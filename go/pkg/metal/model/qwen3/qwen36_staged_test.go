// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func TestQwen36HybridAttentionPlan_ExpandsPattern_Good(t *testing.T) {
	coverageTokens := "Qwen36 HybridAttentionPlan ExpandsPattern"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	plan, err := metal.BuildHybridAttentionCachePlan(6, []string{"linear-attention", "full_attention"}, 4096)
	if err != nil {
		t.Fatalf("BuildHybridAttentionCachePlan() error = %v", err)
	}
	if len(plan.Layers) != 6 || plan.CachelessLayers != 3 || plan.GlobalLayers != 3 {
		t.Fatalf("plan = %+v, want 6 layers with 3 linear and 3 full", plan)
	}
	wantCacheIndex := []int{-1, 0, -1, 1, -1, 2}
	for i, layer := range plan.Layers {
		wantKind := metal.HybridAttentionLinear
		wantKV := false
		wantWindow := 0
		wantLayerCacheIndex := -1
		if i%2 == 1 {
			wantKind = metal.HybridAttentionFull
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
			_, err := metal.BuildHybridAttentionCachePlan(2, tc.layerTypes, 0)
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

	model, err := loadQwen36StagedModel(dir, []byte(`{
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
	}`))
	if err != nil {
		t.Fatalf("loadQwen36StagedModel(qwen3_6) error = %v", err)
	}
	if model.plan.CachelessLayers != 2 || model.plan.GlobalLayers != 2 || len(model.plan.Layers) != 4 {
		t.Fatalf("plan = %+v, want 2 linear and 2 full layers", model.plan)
	}
	if !model.plan.Layers[1].RequiresKV || model.plan.Layers[1].Window != 1024 {
		t.Fatalf("full layer plan = %+v, want KV with window 1024", model.plan.Layers[1])
	}
	if model.plan.Layers[2].RequiresKV || model.plan.Layers[2].Window != 0 {
		t.Fatalf("linear layer plan = %+v, want no KV and zero window", model.plan.Layers[2])
	}
	caches := model.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCache() length = %d, want full-attention layer count 2", len(caches))
	}
	for i, cache := range caches {
		if _, ok := cache.(*metal.KVCache); !ok {
			t.Fatalf("cache[%d] = %T, want *KVCache for full-attention layer", i, cache)
		}
	}
	plan, ok := model.HybridAttentionCachePlan()
	if !ok || len(plan.CacheIndexByLayer) != 4 || plan.CacheIndexByLayer[0] != -1 || plan.CacheIndexByLayer[1] != 0 || plan.CacheIndexByLayer[2] != -1 || plan.CacheIndexByLayer[3] != 1 {
		t.Fatalf("HybridAttentionCachePlan(qwen3_6) = %+v ok=%v, want [-1 0 -1 1]", plan, ok)
	}
}

func TestModel_LoadAndInit_Qwen36StagedLoader_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_5",
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"text_config": {
			"model_type": "qwen3_5_text",
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"num_hidden_layers": 64,
			"num_attention_heads": 24,
			"num_key_value_heads": 4,
			"head_dim": 256,
			"vocab_size": 248320,
			"max_position_embeddings": 262144,
			"layer_types": ["linear_attention", "full_attention"],
			"quantization": {"bits": 4, "group_size": 64}
		}
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3_6 staged fixture) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "qwen3_6" {
		t.Fatalf("ModelType() = %q, want qwen3_6", model.ModelType())
	}
	info := model.Info()
	if info.Architecture != "qwen3_6" || info.VocabSize != 248320 || info.HiddenSize != 5120 || info.NumLayers != 64 || info.ContextLength != 262144 {
		t.Fatalf("Info() = %+v, want Qwen3.6 config metadata", info)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
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

	model, err := loadQwen36MoEStagedModel(dir, []byte(`{
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
	}`))
	if err != nil {
		t.Fatalf("loadQwen36MoEStagedModel(qwen3_6_moe) error = %v", err)
	}
	if model.plan.CachelessLayers != 2 || model.plan.GlobalLayers != 2 || len(model.plan.Layers) != 4 {
		t.Fatalf("plan = %+v, want 2 linear and 2 full layers", model.plan)
	}
	caches := model.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCache() length = %d, want full-attention layer count 2", len(caches))
	}
	plan, ok := model.HybridAttentionCachePlan()
	if !ok || len(plan.CacheIndexByLayer) != 4 || plan.CacheIndexByLayer[0] != -1 || plan.CacheIndexByLayer[1] != 0 || plan.CacheIndexByLayer[2] != -1 || plan.CacheIndexByLayer[3] != 1 {
		t.Fatalf("HybridAttentionCachePlan(qwen3_6_moe) = %+v ok=%v, want [-1 0 -1 1]", plan, ok)
	}
}

func TestModel_LoadAndInit_Qwen36MoEStagedLoader_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["Qwen3_6MoeForConditionalGeneration"],
		"model_type": "qwen3_6_moe",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 16,
		"num_key_value_heads": 2,
		"vocab_size": 248320,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 512,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3_6_moe staged fixture) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "qwen3_6_moe" {
		t.Fatalf("ModelType() = %q, want qwen3_6_moe", model.ModelType())
	}
	info := model.Info()
	if info.VocabSize != 248320 || info.HiddenSize != 1024 || info.NumLayers != 2 {
		t.Fatalf("Info() = %+v, want vocab=248320 hidden=1024 layers=2", info)
	}
}
