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

// TestQwen36_NativeGuardMessage_Good covers both diagnostic strings the dense
// and MoE loaders emit when a hybrid Qwen 3.6 config reaches the native trunk:
// the MoE variant names sparse expert routing, the dense variant does not.
func TestQwen36_NativeGuardMessage_Good(t *testing.T) {
	moe := qwen36NativeGuardMessage("qwen3_6_moe")
	if !core.Contains(moe, "qwen3_6_moe") || !core.Contains(moe, "sparse expert") {
		t.Fatalf("MoE guard message = %q, want qwen3_6_moe + sparse expert", moe)
	}
	dense := qwen36NativeGuardMessage("qwen3_6")
	if !core.Contains(dense, "qwen3_6") || core.Contains(dense, "sparse expert") {
		t.Fatalf("dense guard message = %q, want qwen3_6 without sparse expert", dense)
	}
}

// TestQwen36_IsHybridConfig_Good covers the three positive paths of
// isQwen36HybridConfig: the qwen3_6 model_type, a linear_attention layer type,
// and a fractional partial-rotary factor; _Bad covers a plain dense config and
// the nil guard.
func TestQwen36_IsHybridConfig_Good(t *testing.T) {
	byType := &metal.DenseConfig{}
	byType.ModelType = "qwen3_6"
	if !isQwen36HybridConfig(byType) {
		t.Error("qwen3_6 model_type not detected as hybrid")
	}

	byLayer := &metal.DenseConfig{LayerTypes: []string{"linear_attention", "full_attention"}}
	if !isQwen36HybridConfig(byLayer) {
		t.Error("linear_attention layer type not detected as hybrid")
	}

	byRotary := &metal.DenseConfig{PartialRotaryFactor: 0.25}
	if !isQwen36HybridConfig(byRotary) {
		t.Error("fractional partial_rotary_factor not detected as hybrid")
	}
}

func TestQwen36_IsHybridConfig_Bad(t *testing.T) {
	if isQwen36HybridConfig(nil) {
		t.Error("nil config detected as hybrid, want false")
	}
	plain := &metal.DenseConfig{LayerTypes: []string{"full_attention"}}
	plain.ModelType = "qwen3"
	if isQwen36HybridConfig(plain) {
		t.Error("plain qwen3 dense config detected as hybrid, want false")
	}
}

// TestQwen36_LoadQwen3_HybridGuard_Bad confirms the dense native loader rejects
// a Qwen 3.6 hybrid config with the guard message rather than attempting a load.
func TestQwen36_LoadQwen3_HybridGuard_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_6",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 100,
		"max_position_embeddings": 4096,
		"layer_types": ["linear_attention", "full_attention"]
	}`)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected hybrid-guard error from LoadQwen3 on a qwen3_6 config")
	}
	if !core.Contains(err.Error(), "qwen3_6") {
		t.Fatalf("error = %v, want qwen3_6 hybrid guard", err)
	}
}

// TestQwen36_StagedModel_DecodeUnavailableError_Good confirms the staged dense
// loader's decode-unavailable error names the operation and the missing native
// hybrid linear-attention kernels (a real formatted error, not a nil sentinel).
func TestQwen36_StagedModel_DecodeUnavailableError_Good(t *testing.T) {
	m := &qwen36StagedModel{}
	err := m.DecodeUnavailableError("generate")
	if err == nil {
		t.Fatal("DecodeUnavailableError = nil, want a formatted error")
	}
	if !core.Contains(err.Error(), "generate") || !core.Contains(err.Error(), "linear-attention") {
		t.Fatalf("error = %v, want operation + linear-attention", err)
	}
}

// TestQwen36_StagedModel_FillModelInfo_Good copies the staged dense config's
// metadata into a ModelInfo, including the sliding-window fallback for context
// length and the quantization fields.
func TestQwen36_StagedModel_FillModelInfo_Good(t *testing.T) {
	m := &qwen36StagedModel{config: qwen36StagedConfig{
		VocabSize:     128,
		HiddenSize:    16,
		SlidingWindow: 1024, // ContextLength falls back to this when max pos = 0
		Quantization:  metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.VocabSize != 128 || info.HiddenSize != 16 {
		t.Fatalf("FillModelInfo = %+v, want vocab=128 hidden=16", info)
	}
	if info.ContextLength != 1024 {
		t.Fatalf("ContextLength = %d, want 1024 sliding-window fallback", info.ContextLength)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("FillModelInfo quant = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
	}
}

// TestQwen36_MoEStagedModel_DecodeUnavailableError_Good confirms the staged MoE
// loader's decode-unavailable error names the operation and the missing native
// hybrid + sparse-expert kernels.
func TestQwen36_MoEStagedModel_DecodeUnavailableError_Good(t *testing.T) {
	m := &qwen36MoEStagedModel{}
	err := m.DecodeUnavailableError("generate")
	if err == nil {
		t.Fatal("DecodeUnavailableError = nil, want a formatted error")
	}
	if !core.Contains(err.Error(), "generate") || !core.Contains(err.Error(), "sparse-expert") {
		t.Fatalf("error = %v, want operation + sparse-expert", err)
	}
}

// TestQwen36_MoEStagedModel_FillModelInfo_Good copies the staged MoE config's
// metadata into a ModelInfo including the quantization fields.
func TestQwen36_MoEStagedModel_FillModelInfo_Good(t *testing.T) {
	cfg := &metal.DenseConfig{Quantization: &metal.QuantizationConfig{Bits: 4, GroupSize: 32}}
	cfg.VocabSize = 256
	cfg.HiddenSize = 32
	cfg.MaxPositionEmbeddings = 8192
	m := &qwen36MoEStagedModel{config: cfg}

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.VocabSize != 256 || info.HiddenSize != 32 || info.ContextLength != 8192 {
		t.Fatalf("FillModelInfo = %+v, want vocab=256 hidden=32 ctx=8192", info)
	}
	if info.QuantBits != 4 || info.QuantGroup != 32 {
		t.Fatalf("FillModelInfo quant = %d/%d, want 4/32", info.QuantBits, info.QuantGroup)
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
