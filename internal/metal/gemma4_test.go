//go:build darwin && arm64

package metal

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/core"

	coreio "dappco.re/go/core/io"
)

func requireMetalRuntime(t *testing.T) {
	t.Helper()
	if os.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func TestGemma4_ParseConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.GlobalHeadDim != 256 {
		t.Errorf("GlobalHeadDim = %d, want 256", cfg.GlobalHeadDim)
	}
	if cfg.SlidingWindow != 512 {
		t.Errorf("SlidingWindow = %d, want 512", cfg.SlidingWindow)
	}
	if cfg.NumKVSharedLayers != 20 {
		t.Errorf("NumKVSharedLayers = %d, want 20", cfg.NumKVSharedLayers)
	}
	if cfg.FinalLogitSoftcapping != 30 {
		t.Errorf("FinalLogitSoftcapping = %f, want 30", cfg.FinalLogitSoftcapping)
	}
	if len(cfg.LayerTypes) != 6 {
		t.Fatalf("LayerTypes len = %d, want 6", len(cfg.LayerTypes))
	}
	want := []string{
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"sliding_attention",
	}
	for i, got := range cfg.LayerTypes {
		if got != want[i] {
			t.Fatalf("LayerTypes[%d] = %q, want %q", i, got, want[i])
		}
	}
	if cfg.RopeParameters["full_attention"].RopeType != "proportional" {
		t.Errorf("full attention rope type = %q, want proportional", cfg.RopeParameters["full_attention"].RopeType)
	}
	if cfg.RopeParameters["sliding_attention"].RopeTheta != 10000 {
		t.Errorf("sliding attention rope theta = %f, want 10000", cfg.RopeParameters["sliding_attention"].RopeTheta)
	}
}

func TestGemma4_ParseConfig_ExplicitZeroSharedKV_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"num_kv_shared_layers": 0
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumKVSharedLayers != 0 {
		t.Fatalf("NumKVSharedLayers = %d, want 0", cfg.NumKVSharedLayers)
	}
}

func TestGemma4_ParseConfig_PartialRopeParameters_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"rope_parameters": {
			"full_attention": {
				"rope_theta": 123456
			}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	full := cfg.RopeParameters["full_attention"]
	if full.RopeTheta != 123456 {
		t.Fatalf("full rope theta = %f, want 123456", full.RopeTheta)
	}
	if full.PartialRotaryFactor != 0.25 {
		t.Fatalf("full partial rotary factor = %f, want 0.25", full.PartialRotaryFactor)
	}
	if full.RopeType != "proportional" {
		t.Fatalf("full rope type = %q, want proportional", full.RopeType)
	}
	if full.Factor != 1.0 {
		t.Fatalf("full factor = %f, want 1.0", full.Factor)
	}

	sliding := cfg.RopeParameters["sliding_attention"]
	if sliding.RopeTheta != 10000 {
		t.Fatalf("sliding rope theta = %f, want 10000", sliding.RopeTheta)
	}
	if sliding.PartialRotaryFactor != 1.0 {
		t.Fatalf("sliding partial rotary factor = %f, want 1.0", sliding.PartialRotaryFactor)
	}
	if sliding.RopeType != "default" {
		t.Fatalf("sliding rope type = %q, want default", sliding.RopeType)
	}
}

func TestGemma4_ParseConfig_MoEDefaults_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"enable_moe_block": true
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumExperts == nil || *cfg.NumExperts != 128 {
		t.Fatalf("NumExperts = %v, want 128", cfg.NumExperts)
	}
	if cfg.TopKExperts == nil || *cfg.TopKExperts != 8 {
		t.Fatalf("TopKExperts = %v, want 8", cfg.TopKExperts)
	}
}

func TestGemma4_ParseConfig_NestedQuantization_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"text_config": {
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"layer_types": ["sliding_attention", "full_attention"],
			"quantization": {"group_size": 64, "bits": 4}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ModelType != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", cfg.ModelType)
	}
	if cfg.Quantization == nil || cfg.Quantization.GroupSize != 64 || cfg.Quantization.Bits != 4 {
		t.Fatalf("Quantization = %+v, want group_size=64 bits=4", cfg.Quantization)
	}
	if got := cfg.LayerTypes; len(got) != 2 || got[0] != "sliding_attention" || got[1] != "full_attention" {
		t.Fatalf("LayerTypes = %v, want explicit nested layer types", got)
	}
}

func TestGemma4_OutputLinear_TiedFallback_Good(t *testing.T) {
	embed := &Embedding{}
	output, err := gemma4OutputLinear(map[string]*Array{}, &Gemma4TextConfig{
		TieWordEmbeddings: true,
	}, embed)
	if err != nil {
		t.Fatalf("gemma4OutputLinear: %v", err)
	}
	if output == nil {
		t.Fatal("expected tied output linear")
	}
	if output.Weight != embed.Weight || output.Scales != embed.Scales || output.Biases != embed.Biases {
		t.Fatal("tied output should reuse embedding weights")
	}
}

func TestGemma4_OutputLinear_UntiedMissingLMHead_Bad(t *testing.T) {
	_, err := gemma4OutputLinear(map[string]*Array{}, &Gemma4TextConfig{}, &Embedding{})
	if err == nil {
		t.Fatal("expected error when untied Gemma4 model lacks lm_head.weight")
	}
	if !core.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("expected lm_head.weight error, got: %v", err)
	}
}

func TestGemma4_AttentionScale_Good(t *testing.T) {
	got := gemma4AttentionScale(512)
	want := float32(1.0 / math.Sqrt(512))
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("gemma4AttentionScale(512) = %f, want %f", got, want)
	}
}

func TestGemma4_SanitizeWeights_GateUpProj_Good(t *testing.T) {
	requireMetalRuntime(t)

	gateUp := FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 1, 4, 2)
	Materialize(gateUp)
	vision := FromValues([]float32{1}, 1)
	rotary := FromValues([]float32{1}, 1)

	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.layers.0.experts.gate_up_proj.weight": gateUp,
		"model.vision_tower.block.weight":            vision,
		"model.layers.0.self_attn.rotary_emb.inv":    rotary,
	})

	gate := sanitized["model.layers.0.experts.gate_proj.weight"]
	up := sanitized["model.layers.0.experts.up_proj.weight"]
	if gate == nil || up == nil {
		t.Fatal("expected split gate_proj and up_proj weights")
	}
	if _, ok := sanitized["model.layers.0.experts.gate_up_proj.weight"]; ok {
		t.Fatal("gate_up_proj should be replaced by split weights")
	}
	if _, ok := sanitized["model.vision_tower.block.weight"]; ok {
		t.Fatal("vision tower weights should be stripped")
	}
	if _, ok := sanitized["model.layers.0.self_attn.rotary_emb.inv"]; ok {
		t.Fatal("rotary embedding weights should be stripped")
	}
	if got := gate.Shape(); len(got) != 3 || got[1] != 2 {
		t.Fatalf("gate split shape = %v, want [1 2 2]", got)
	}
	if got := up.Shape(); len(got) != 3 || got[1] != 2 {
		t.Fatalf("up split shape = %v, want [1 2 2]", got)
	}
	if gateUp.Valid() {
		t.Fatal("gate_up source tensor should be freed after split sanitization")
	}
	if vision.Valid() {
		t.Fatal("vision tower tensor should be freed after sanitization")
	}
	if rotary.Valid() {
		t.Fatal("rotary embedding tensor should be freed after sanitization")
	}
}

func TestGemma4_SanitizeWeights_GateUpProjBias2D_Good(t *testing.T) {
	requireMetalRuntime(t)

	biases := FromValues([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 2, 4)
	Materialize(biases)

	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.layers.0.experts.gate_up_proj.biases": biases,
	})

	gate := sanitized["model.layers.0.experts.gate_proj.biases"]
	up := sanitized["model.layers.0.experts.up_proj.biases"]
	if gate == nil || up == nil {
		t.Fatal("expected split gate_proj and up_proj biases")
	}
	if got := gate.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("gate bias split shape = %v, want [2 2]", got)
	}
	if got := up.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("up bias split shape = %v, want [2 2]", got)
	}
}

func TestGemma4_SanitizeWeights_LanguageModelPrefix_Good(t *testing.T) {
	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"language_model.model.embed_tokens.weight":       nil,
		"language_model.model.norm.weight":               nil,
		"language_model.model.vision_tower.block.weight": nil,
		"language_model.multi_modal_projector.weight":    nil,
	})

	if _, ok := sanitized["model.embed_tokens.weight"]; !ok {
		t.Fatal("expected embed_tokens weight to be normalised to model.*")
	}
	if _, ok := sanitized["model.norm.weight"]; !ok {
		t.Fatal("expected norm weight to be normalised to model.*")
	}
	if _, ok := sanitized["language_model.model.embed_tokens.weight"]; ok {
		t.Fatal("expected language_model.model prefix to be stripped")
	}
	if _, ok := sanitized["language_model.model.vision_tower.block.weight"]; ok {
		t.Fatal("vision tower weights should be stripped even under language_model.model")
	}
	if _, ok := sanitized["language_model.multi_modal_projector.weight"]; ok {
		t.Fatal("multimodal projector weights should be stripped even under language_model")
	}
}

func TestGemma4_BuildPreviousKVs_Good(t *testing.T) {
	layers := []*Gemma4DecoderLayer{
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
	}
	got := buildGemma4PreviousKVs(layers, 2)
	want := []int32{0, 1, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("PreviousKVs[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestGemma4_BuildCacheLayout_PromotesMissingOwner_Good(t *testing.T) {
	layers := []*Gemma4DecoderLayer{
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
		{LayerType: "sliding_attention"},
	}

	previous, cacheIndexByLayer := buildGemma4CacheLayout(layers, 2)

	wantPrevious := []int32{0, 1, 2, 3, 4, 3}
	for i, want := range wantPrevious {
		if previous[i] != want {
			t.Fatalf("PreviousKVs[%d] = %d, want %d", i, previous[i], want)
		}
	}

	wantCacheIndex := []int32{0, 1, 2, 3, 4, -1}
	for i, want := range wantCacheIndex {
		if cacheIndexByLayer[i] != want {
			t.Fatalf("CacheIndexByLayer[%d] = %d, want %d", i, cacheIndexByLayer[i], want)
		}
	}
}

func TestGemma4_NewCache_SharedLayers_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumHiddenLayers:   4,
			NumKVSharedLayers: 2,
			SlidingWindow:     32,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
	}
	caches := model.NewCache()
	if len(caches) != 2 {
		t.Fatalf("len(caches) = %d, want 2", len(caches))
	}
	if _, ok := caches[0].(*RotatingKVCache); !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if _, ok := caches[1].(*KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *KVCache", caches[1])
	}
}

func TestGemma4_NewCache_PromotedOwner_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumHiddenLayers:   6,
			NumKVSharedLayers: 2,
			SlidingWindow:     32,
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

	caches := model.NewCache()
	if len(caches) != 5 {
		t.Fatalf("len(caches) = %d, want 5", len(caches))
	}
	if _, ok := caches[4].(*KVCache); !ok {
		t.Fatalf("cache[4] = %T, want *KVCache for promoted full-attention owner", caches[4])
	}
	if got := model.PreviousKVs[4]; got != 4 {
		t.Fatalf("PreviousKVs[4] = %d, want 4", got)
	}
	if got := model.CacheIndexByLayer[4]; got != 4 {
		t.Fatalf("CacheIndexByLayer[4] = %d, want 4", got)
	}
}

func TestGemma4_LoadModel_Dispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"hidden_size_per_layer_input": 0
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected tokenizer error, proving dispatch reached Gemma4 loader")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "gemma4") {
		t.Fatalf("expected gemma4 loader error, got: %v", err)
	}
}

func TestGemma4_LoadAndForwardDenseModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"tie_word_embeddings": true,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func TestGemma4_LoadAndForwardDenseModelFromGGUF_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"tie_word_embeddings": true,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveGGUF(core.JoinPath(dir, "model.gguf"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveGGUF: %v", err)
	}

	model, err := LoadGemma4(core.JoinPath(dir, "model.gguf"))
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func gemma4TinyWeights() map[string]*Array {
	weights := map[string]*Array{
		"model.embed_tokens.weight": seqArray(0.01, 10, 8),
		"model.norm.weight":         seqArray(0.02, 8),
	}

	addLayer := func(idx int, sliding bool) {
		prefix := core.Sprintf("model.layers.%d", idx)
		headDim := 4
		oIn := 4
		if !sliding {
			headDim = 8
			oIn = 8
		}
		weights[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx), 8)
		weights[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx), 8)
		weights[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx), 8)
		weights[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx), 8)
		weights[prefix+".layer_scalar"] = FromValues([]float32{1}, 1)

		weights[prefix+".self_attn.q_proj.weight"] = seqArray(0.10+float32(idx), headDim, 8)
		weights[prefix+".self_attn.k_proj.weight"] = seqArray(0.20+float32(idx), headDim, 8)
		weights[prefix+".self_attn.v_proj.weight"] = seqArray(0.30+float32(idx), headDim, 8)
		weights[prefix+".self_attn.o_proj.weight"] = seqArray(0.40+float32(idx), 8, oIn)
		weights[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx), headDim)
		weights[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx), headDim)

		weights[prefix+".mlp.gate_proj.weight"] = seqArray(0.70+float32(idx), 16, 8)
		weights[prefix+".mlp.up_proj.weight"] = seqArray(0.80+float32(idx), 16, 8)
		weights[prefix+".mlp.down_proj.weight"] = seqArray(0.90+float32(idx), 8, 16)
	}

	addLayer(0, true)
	addLayer(1, false)
	return weights
}

func seqArray(start float32, shape ...int) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range size {
		data[i] = start + 0.01*float32(i)
	}
	return FromValues(data, shape...)
}
