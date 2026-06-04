// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"encoding/binary"
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// --- loadModel dispatch ---

func TestModel_LoadModel_MissingConfigJSON_Bad(t *testing.T) {
	coverageTokens := "LoadModel MissingConfigJSON"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	dir := t.TempDir()
	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error for missing config.json")
	}
	if !core.Contains(err.Error(), "config") {
		t.Errorf("error should mention config, got: %v", err)
	}
}

func TestModel_LoadModel_InvalidConfigJSON_Bad(t *testing.T) {
	coverageTokens := "LoadModel InvalidConfigJSON"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "{invalid")

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestModel_LoadModel_UnsupportedArchitecture_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{"model_type": "gpt99"}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error for unsupported architecture")
	}
	if !core.Contains(err.Error(), "gpt99") {
		t.Errorf("error should mention architecture name, got: %v", err)
	}
}

func TestModel_LoadModel_Gemma3TextType_Good(t *testing.T) {
	// "gemma3_text" should route to Gemma3 loader (will fail on missing tokenizer, but
	// that proves the dispatch happened).
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma3_text",
		"hidden_size": 1152,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"vocab_size": 1000
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error (missing tokenizer), but dispatch should have reached gemma3")
	}
	// If the error mentions "tokenizer" or "gemma3", dispatch worked correctly.
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "gemma3") {
		t.Errorf("expected gemma3 loader error, got: %v", err)
	}
}

func TestModel_LoadModel_Gemma4NestedTextConfig_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 1152,
			"num_hidden_layers": 2,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"vocab_size": 1000
		}
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error (missing tokenizer), but dispatch should have reached gemma4")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "gemma4") {
		t.Errorf("expected gemma4 loader error, got: %v", err)
	}
}

func TestModel_LoadModel_Gemma4AssistantStandaloneBoundary_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma4_assistant",
		"architectures": ["Gemma4AssistantForCausalLM"],
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 256,
			"num_hidden_layers": 4,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"vocab_size": 262144
		}
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected assistant loader boundary error")
	}
	if !core.Contains(err.Error(), "attached MTP drafter") ||
		!core.Contains(err.Error(), "LoadSpeculativePair") ||
		!core.Contains(err.Error(), "LoadGemma4AssistantPair") {
		t.Errorf("expected assistant attached-loader boundary error, got: %v", err)
	}
}

func TestModel_LoadModel_ArchitecturesFallback_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["Qwen2ForCausalLM"],
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error (missing tokenizer), but dispatch should have reached qwen2/qwen3")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "qwen") {
		t.Errorf("expected qwen loader error, got: %v", err)
	}
}

func TestModel_LoadAndGenerateMistralDenseNative_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["MistralForCausalLM"],
		"model_type": "mistral",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`)
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(mistral) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "mistral" {
		t.Fatalf("ModelType() = %q, want mistral", model.ModelType())
	}

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

func TestModel_LoadAndGenerateHermesDenseNative_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["HermesForCausalLM"],
		"model_type": "hermes",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`)
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(hermes) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "hermes" {
		t.Fatalf("ModelType() = %q, want hermes", model.ModelType())
	}

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

func TestModel_LoadAndGenerateGraniteDenseNative_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GraniteForCausalLM"],
		"model_type": "granite",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`)
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(granite) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "granite" {
		t.Fatalf("ModelType() = %q, want granite", model.ModelType())
	}

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

func TestModel_LoadAndGeneratePhiDenseNative_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["Phi3ForCausalLM"],
		"model_type": "phi3",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`)
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(phi) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "phi" {
		t.Fatalf("ModelType() = %q, want phi", model.ModelType())
	}

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

func TestModel_LoadAndGenerateGLMDenseNative_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GlmForCausalLM"],
		"model_type": "glm",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`)
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(glm) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "glm" {
		t.Fatalf("ModelType() = %q, want glm", model.ModelType())
	}

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

func TestModel_LoadModel_Qwen3NextNestedTextConfig_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_next",
		"text_config": {
			"model_type": "qwen3_next",
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"num_attention_heads": 8,
			"num_key_value_heads": 4,
			"vocab_size": 1000
		}
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error (missing tokenizer), but dispatch should have reached qwen3_next")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "qwen") {
		t.Errorf("expected qwen loader error, got: %v", err)
	}
}

func TestModel_ProbeModelType_Qwen25And36Aliases_Good(t *testing.T) {
	cases := map[string]string{
		`{"model_type":"qwen2.5","architectures":["Qwen2.5ForCausalLM"]}`:                                   "qwen2",
		`{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}`:                      "qwen3_6",
		`{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForConditionalGeneration"]}`:               "qwen3_6_moe",
		`{"text_config":{"model_type":"qwen3_5_text"},"architectures":["Qwen3_5ForConditionalGeneration"]}`: "qwen3_6",
		`{"architectures":["MistralForCausalLM"]}`:                                                          "mistral",
		`{"architectures":["HermesForCausalLM"]}`:                                                           "hermes",
		`{"architectures":["GraniteForCausalLM"]}`:                                                          "granite",
		`{"architectures":["Phi3ForCausalLM"]}`:                                                             "phi",
		`{"architectures":["GlmForCausalLM"]}`:                                                              "glm",
	}
	for config, want := range cases {
		got, err := probeModelType([]byte(config))
		if err != nil {
			t.Fatalf("probeModelType(%s) error = %v", config, err)
		}
		if got != want {
			t.Fatalf("probeModelType(%s) = %q, want %q", config, got, want)
		}
	}
}

func TestModel_ProbeModelType_OfficialGemma4ConditionalTextPath_Good(t *testing.T) {
	got, err := probeModelType([]byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 2048,
			"num_hidden_layers": 26,
			"num_attention_heads": 8,
			"num_key_value_heads": 4,
			"head_dim": 256,
			"vocab_size": 262208,
			"max_position_embeddings": 131072
		},
		"vision_config": {"hidden_size": 1152}
	}`))
	if err != nil {
		t.Fatalf("probeModelType() error = %v", err)
	}
	if got != "gemma4_text" {
		t.Fatalf("probeModelType() = %q, want gemma4_text for official target text path", got)
	}
}

func TestModel_LoadModel_Qwen36StagedLoader_Good(t *testing.T) {
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

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(qwen3_6 staged fixture) error = %v", err)
	}
	if model.ModelType() != "qwen3_6" {
		t.Fatalf("ModelType() = %q, want qwen3_6", model.ModelType())
	}
	if model.NumLayers() != 64 {
		t.Fatalf("NumLayers() = %d, want 64", model.NumLayers())
	}
	caches := model.NewCache()
	defer FreeCaches(caches)
	if len(caches) != 32 {
		t.Fatalf("NewCache() length = %d, want one cache for each full-attention layer", len(caches))
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.Architecture != "qwen3_6" || info.VocabSize != 248320 || info.HiddenSize != 5120 || info.NumLayers != 64 || info.ContextLength != 262144 {
		t.Fatalf("Info() = %+v, want Qwen3.6 config metadata", info)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
	}
	if _, ok := model.(*qwen36StagedModel); !ok {
		t.Fatalf("model type = %T, want *qwen36StagedModel", model)
	}
}

func TestModel_LoadModel_Qwen3MoEModelTypeDispatch_Good(t *testing.T) {
	// Verifies loadModel dispatches qwen3_moe to the full model constructor.
	// This test expects model.safetensors to be missing, so LoadQwen3MoE
	// returns a weight-loading error — but the dispatch itself is correct.
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000,
		"max_position_embeddings": 32768,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 384,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for qwen3_moe without safetensors")
	}
	if !core.Contains(err.Error(), "qwen3_moe") {
		t.Fatalf("error = %v, should contain qwen3_moe", err)
	}
}

// Mixtral model-type dispatch + load coverage travels with the model in
// package metal/model/mixtral.

func TestModel_LoadModel_GptOssModelTypeDispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 201088,
		"num_local_experts": 32
	}`)
	writeMinimalTokenizer(t, dir)
	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for gpt_oss without safetensors")
	}
	if !core.Contains(err.Error(), "gpt_oss") {
		t.Fatalf("error = %v, should contain gpt_oss", err)
	}
}

// Kimi model-type dispatch + load coverage travels with the model in package
// metal/model/kimi.

func TestModel_LoadModel_MoEStagedLoadersValidateConfigAndTokenizer_Good(t *testing.T) {
	cases := []struct {
		name   string
		config string
		want   struct {
			modelType  string
			vocabSize  int
			hiddenSize int
			numLayers  int
		}
	}{
		{
			name: "deepseek",
			config: `{
				"architectures": ["DeepseekV3ForCausalLM"],
				"model_type": "deepseek_v3",
				"hidden_size": 1024,
				"num_hidden_layers": 2,
				"num_attention_heads": 8,
				"num_key_value_heads": 2,
				"vocab_size": 32000,
				"n_routed_experts": 64,
				"q_lora_rank": 1536,
				"kv_lora_rank": 512,
				"qk_nope_head_dim": 128,
				"qk_rope_head_dim": 64,
				"v_head_dim": 128
			}`,
			want: struct {
				modelType  string
				vocabSize  int
				hiddenSize int
				numLayers  int
			}{"deepseek", 32000, 1024, 2},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), tc.config)
			writeMinimalTokenizer(t, dir)

			model, err := loadModel(dir)
			if err != nil {
				t.Fatalf("loadModel(%s staged fixture) error = %v", tc.name, err)
			}
			if model.ModelType() != tc.want.modelType {
				t.Fatalf("ModelType() = %q, want %q", model.ModelType(), tc.want.modelType)
			}
			if model.NumLayers() != tc.want.numLayers {
				t.Fatalf("NumLayers() = %d, want %d", model.NumLayers(), tc.want.numLayers)
			}
			if caches := model.NewCache(); caches != nil {
				t.Fatalf("NewCache() = %#v, want nil for staged loader", caches)
			}
			if model.Tokenizer() == nil {
				t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
			}
			info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
			if info.VocabSize != tc.want.vocabSize || info.HiddenSize != tc.want.hiddenSize {
				t.Fatalf("Info() = %+v, want vocab=%d hidden=%d", info, tc.want.vocabSize, tc.want.hiddenSize)
			}
			if _, ok := model.(*moeStagedModel); !ok {
				t.Fatalf("model type = %T, want *moeStagedModel", model)
			}
			if tc.name == "deepseek" {
				staged := model.(*moeStagedModel)
				if staged.mla.KVLoRARank != 512 || staged.mla.QKHeadDim != 192 || staged.mla.VHeadDim != 128 {
					t.Fatalf("DeepSeek MLA plan = %+v, want kv rank 512 qk head 192 v head 128", staged.mla)
				}
			}
		})
	}
}

func TestModel_LoadModel_DeepSeekStagedValidatesMLA_Bad(t *testing.T) {
	base := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64,
		%s
	}`
	cases := []struct {
		name string
		mla  string
		want string
	}{
		{
			name: "missing-kv-lora",
			mla:  `"qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128`,
			want: "kv_lora_rank",
		},
		{
			name: "missing-rope-split",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "v_head_dim": 128`,
			want: "qk_nope_head_dim and qk_rope_head_dim",
		},
		{
			name: "bad-qk-sum",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "qk_head_dim": 256, "v_head_dim": 128`,
			want: "qk_head_dim",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), core.Sprintf(base, tc.mla))
			writeMinimalTokenizer(t, dir)

			_, err := loadModel(dir)
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("loadModel(deepseek invalid MLA) error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestModel_LoadModel_Qwen36MoEStagedLoaderValidatesHybridConfig_Good(t *testing.T) {
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

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(qwen3_6_moe staged fixture) error = %v", err)
	}
	if model.ModelType() != "qwen3_6_moe" {
		t.Fatalf("ModelType() = %q, want qwen3_6_moe", model.ModelType())
	}
	if model.NumLayers() != 2 {
		t.Fatalf("NumLayers() = %d, want 2", model.NumLayers())
	}
	caches := model.NewCache()
	defer FreeCaches(caches)
	if len(caches) != 1 {
		t.Fatalf("NewCache() length = %d, want one cache for the full-attention layer", len(caches))
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.VocabSize != 248320 || info.HiddenSize != 1024 {
		t.Fatalf("Info() = %+v, want vocab=248320 hidden=1024", info)
	}
	if _, ok := model.(*qwen36MoEStagedModel); !ok {
		t.Fatalf("model type = %T, want *qwen36MoEStagedModel", model)
	}
}

func TestModel_LoadModel_BERTStagedEncoderLoader_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"hidden_size": 384,
		"num_hidden_layers": 6,
		"num_attention_heads": 12,
		"intermediate_size": 1536,
		"vocab_size": 30522,
		"max_position_embeddings": 512
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(bert staged fixture) error = %v", err)
	}
	if model.ModelType() != "bert" {
		t.Fatalf("ModelType() = %q, want bert", model.ModelType())
	}
	if model.NumLayers() != 6 {
		t.Fatalf("NumLayers() = %d, want 6", model.NumLayers())
	}
	if caches := model.NewCache(); caches != nil {
		t.Fatalf("NewCache() = %#v, want nil for encoder no-KV staged loader", caches)
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged BERT loader to expose tokenizer metadata")
	}
	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.VocabSize != 30522 || info.HiddenSize != 384 || info.ContextLength != 512 {
		t.Fatalf("Info() = %+v, want BERT config metadata", info)
	}
	if _, ok := model.(*bertStagedModel); !ok {
		t.Fatalf("model type = %T, want *bertStagedModel", model)
	}
}

func TestModel_LoadModel_BERTRerankStagedLoader_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"num_attention_heads": 12,
		"intermediate_size": 3072,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`)
	writeMinimalTokenizer(t, dir)

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(bert_rerank staged fixture) error = %v", err)
	}
	if model.ModelType() != "bert_rerank" {
		t.Fatalf("ModelType() = %q, want bert_rerank", model.ModelType())
	}
	if caches := model.NewCache(); caches != nil {
		t.Fatalf("NewCache() = %#v, want nil for rerank no-KV staged loader", caches)
	}
	staged, ok := model.(*bertStagedModel)
	if !ok {
		t.Fatalf("model type = %T, want *bertStagedModel", model)
	}
	if staged.config.NumLabels != 1 {
		t.Fatalf("NumLabels = %d, want 1", staged.config.NumLabels)
	}
	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.VocabSize != 30522 || info.HiddenSize != 768 || info.ContextLength != 512 {
		t.Fatalf("Info() = %+v, want BERT rerank config metadata", info)
	}
}

func TestModel_LoadModel_BERTRerankMissingLabels_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"vocab_size": 30522,
		"max_position_embeddings": 512
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := loadModel(dir)
	if err == nil || !core.Contains(err.Error(), "bert_rerank") || !core.Contains(err.Error(), "num_labels") {
		t.Fatalf("error = %v, want bert_rerank num_labels diagnostic", err)
	}
}

func TestModel_LoadModel_MiniMaxJANGStagedLoader_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"max_position_embeddings": 1048576,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`)
	writeMinimalTokenizer(t, dir)
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(false))

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(minimax_m2 staged fixture) error = %v", err)
	}
	if model.ModelType() != "minimax_m2" {
		t.Fatalf("ModelType() = %q, want minimax_m2", model.ModelType())
	}
	if model.NumLayers() != 62 {
		t.Fatalf("NumLayers() = %d, want 62", model.NumLayers())
	}
	if caches := model.NewCache(); caches != nil {
		t.Fatalf("NewCache() = %#v, want nil until MiniMax decode kernels are linked", caches)
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.VocabSize != 200064 || info.HiddenSize != 3072 || info.ContextLength != 1048576 {
		t.Fatalf("Info() = %+v, want MiniMax config metadata", info)
	}
	if info.QuantBits != 2 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d/%d, want 2/64", info.QuantBits, info.QuantGroup)
	}
	staged, ok := model.(*miniMaxM2StagedModel)
	if !ok {
		t.Fatalf("model type = %T, want *miniMaxM2StagedModel", model)
	}
	if len(staged.plan.LayerSkeleton.Attention) != 4 || staged.plan.LayerSkeleton.RouterGate.Name == "" || staged.plan.LayerSkeleton.RouterBias == nil {
		t.Fatalf("LayerSkeleton = %+v, want attention plus router metadata", staged.plan.LayerSkeleton)
	}
	if staged.plan.LayerSkeleton.Attention[0].PackedBytes == 0 {
		t.Fatalf("LayerSkeleton attention = %+v, want packed byte metadata", staged.plan.LayerSkeleton.Attention)
	}
	payloadRefs, err := staged.plan.ResolveExpertPayloadRefs(0, []int{0})
	if err != nil {
		t.Fatalf("ResolveExpertPayloadRefs() error = %v", err)
	}
	expert0 := payloadRefs[0]
	if expert0.PackedBytes == 0 || expert0.GateProj.Path == "" || expert0.GateProj.DataStart <= 0 {
		t.Fatalf("expert payload refs = %+v, want packed byte refs without payload loading", expert0)
	}
	if expert0.GateProj.ByteLen != 1179648 || expert0.UpProj.ByteLen != 1179648 || expert0.DownProj.ByteLen != 1179648 {
		t.Fatalf("expert payload byte lengths = gate:%d up:%d down:%d, want JANGTQ packed expert refs", expert0.GateProj.ByteLen, expert0.UpProj.ByteLen, expert0.DownProj.ByteLen)
	}
}

func TestModel_LoadModel_MiniMaxJANGMissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`)
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(false))

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected MiniMax staged loader tokenizer error")
	}
	if !core.Contains(err.Error(), "minimax_m2") || !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want minimax_m2 tokenizer diagnostic", err)
	}
}

func TestModel_LoadModel_MiniMaxJANGRuntimeGuardMissingTensor_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`)
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(true))

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected MiniMax tensor validation error")
	}
	if !core.Contains(err.Error(), "minimax_m2") || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want missing expert up_proj diagnostic", err)
	}
}

func writeMiniMaxM2JANGConfig(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "jang_config.json"), `{
		"version": 1,
		"weight_format": "mxtq",
		"profile": "JANGTQ_K",
		"mxtq_bits": {
			"attention": 8,
			"routed_expert": 2,
			"embed_tokens": 8,
			"lm_head": 8
		},
		"quantization": {
			"method": "affine+mxtq",
			"group_size": 64,
			"bits_default": 2
		}
	}`); err != nil {
		t.Fatalf("write jang_config.json: %v", err)
	}
}

func miniMaxM2FirstLayerTensorNames(omitExpertUp bool) []string {
	names := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.block_sparse_moe.gate.weight",
		"model.layers.0.block_sparse_moe.e_score_correction_bias",
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight",
	}
	if !omitExpertUp {
		names = append(names, "model.layers.0.block_sparse_moe.experts.0.up_proj.weight")
	}
	return names
}

func writeMiniMaxM2SafetensorsHeader(t *testing.T, path string, names []string) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	cursor := 0
	for _, name := range names {
		dtype, shape, byteLen := miniMaxM2TestSafetensorsTensorLayout(name)
		header[name] = entry{DType: dtype, Shape: shape, DataOffsets: [2]int{cursor, cursor + byteLen}}
		cursor += byteLen
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors header: %v", result.Value)
	}
}

func miniMaxM2TestSafetensorsTensorLayout(name string) (string, []int, int) {
	const (
		hidden       = 3072
		qSize        = 6144
		kvSize       = 1024
		intermediate = 1536
		experts      = 256
	)
	switch {
	case core.Contains(name, "self_attn.q_proj.weight"):
		bytes := qSize * hidden
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "self_attn.k_proj.weight"), core.Contains(name, "self_attn.v_proj.weight"):
		bytes := kvSize * hidden
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "self_attn.o_proj.weight"):
		bytes := hidden * qSize
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "block_sparse_moe.gate.weight"):
		return "F32", []int{experts, hidden}, experts * hidden * 4
	case core.Contains(name, "e_score_correction_bias"):
		return "F32", []int{experts}, experts * 4
	case core.Contains(name, ".gate_proj.weight"), core.Contains(name, ".up_proj.weight"):
		bytes := (intermediate * hidden * 2) / 8
		return "U8", []int{bytes}, bytes
	case core.Contains(name, ".down_proj.weight"):
		bytes := (hidden * intermediate * 2) / 8
		return "U8", []int{bytes}, bytes
	default:
		return "F32", []int{1}, 4
	}
}

func TestModel_ProbeModelType_QwenFamilyArchitectures_Good(t *testing.T) {
	cases := []struct {
		name string
		data string
		want string
	}{
		{name: "moe", data: `{"architectures":["Qwen3MoeForCausalLM"]}`, want: "qwen3_moe"},
		{name: "next", data: `{"architectures":["Qwen3NextForCausalLM"]}`, want: "qwen3_next"},
		{name: "alias", data: `{"model_type":"qwen3_5"}`, want: "qwen3_6"},
		{name: "minimax", data: `{"architectures":["MiniMaxM2ForCausalLM"]}`, want: "minimax_m2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := probeModelType([]byte(tc.data))
			if err != nil {
				t.Fatalf("probeModelType() error = %v", err)
			}
			if got != tc.want {
				t.Fatalf("probeModelType() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestModel_DetectQwenModelType_ArchitecturesLlama_Good(t *testing.T) {
	got := DetectDenseModelType([]byte(`{
		"architectures": ["LlamaForCausalLM"]
	}`), nil)
	if got != "llama" {
		t.Fatalf("DetectDenseModelType() = %q, want llama", got)
	}
}

func TestModel_DetectQwenModelType_QwenFamilyVariants_Good(t *testing.T) {
	got := DetectDenseModelType([]byte(`{"architectures":["Qwen3NextForCausalLM"]}`), nil)
	if got != "qwen3_next" {
		t.Fatalf("DetectDenseModelType(next) = %q, want qwen3_next", got)
	}
	got = DetectDenseModelType([]byte(`{"architectures":["Qwen3MoeForCausalLM"]}`), nil)
	if got != "qwen3_moe" {
		t.Fatalf("DetectDenseModelType(moe) = %q, want qwen3_moe", got)
	}
}

func TestModel_DetectQwenModelType_QNormFallback_Good(t *testing.T) {
	got := DetectDenseModelType([]byte(`{}`), map[string]*Array{
		"model.layers.0.self_attn.q_norm.weight": nil,
	})
	if got != "qwen3" {
		t.Fatalf("DetectDenseModelType() = %q, want qwen3", got)
	}

	got = DetectDenseModelType([]byte(`{}`), map[string]*Array{})
	if got != "qwen2" {
		t.Fatalf("DetectDenseModelType() = %q, want qwen2", got)
	}
}

// --- LoadQwen3 error paths ---

func TestModel_LoadQwen3_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing config.json")
	}
}

func TestModel_LoadQwen3_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "{broken")

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadQwen3_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000
	}`)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadQwen3_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Errorf("error should mention safetensors, got: %v", err)
	}
}

// --- LoadAndInit error paths ---

func TestModel_LoadAndInit_MissingPath_Bad(t *testing.T) {
	_, err := LoadAndInit("/nonexistent/model/path")
	if err == nil {
		t.Fatal("expected error for nonexistent path")
	}
}

func TestModel_LoadAndInit_UnsupportedArch_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{"model_type": "falcon"}`)

	_, err := LoadAndInit(dir)
	if err == nil {
		t.Fatal("expected error for unsupported architecture")
	}
	if !core.Contains(err.Error(), "falcon") {
		t.Errorf("error should mention architecture, got: %v", err)
	}
}

func TestModel_LoadAndInit_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "gemma3")
	writeMinimalTokenizer(t, dir)

	_, err := LoadAndInit(dir, LoadConfig{ContextLen: 2048})
	if err == nil {
		t.Fatal("expected error for missing safetensors")
	}
}

// --- ParseDenseConfig ---

func TestModel_ParseQwen3Config_Defaults_Good(t *testing.T) {
	cfg, err := ParseDenseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig: %v", err)
	}
	if cfg.HeadDim != 256 { // 1024/4
		t.Errorf("HeadDim = %d, want 256 (hidden/heads)", cfg.HeadDim)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.VocabSize != 151936 {
		t.Errorf("VocabSize default = %d, want 151936", cfg.VocabSize)
	}
}

func TestModel_ParseQwen3Config_MoEFields_Good(t *testing.T) {
	cfg, err := ParseDenseConfig([]byte(`{
		"model_type": "qwen3_moe",
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 384,
		"decoder_sparse_step": 2
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig: %v", err)
	}
	if cfg.ModelType != "qwen3_moe" || !cfg.IsMoE() {
		t.Fatalf("model type/is moe = %q/%v, want qwen3_moe true", cfg.ModelType, cfg.IsMoE())
	}
	if cfg.NumExperts != 128 || cfg.NumExpertsPerTok != 8 || cfg.MoEIntermediateSize != 384 || cfg.DecoderSparseStep != 2 {
		t.Fatalf("MoE fields = experts:%d per_tok:%d intermediate:%d sparse_step:%d", cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize, cfg.DecoderSparseStep)
	}
}

func TestModel_ParseQwen3Config_InvalidJSON_Bad(t *testing.T) {
	_, err := ParseDenseConfig([]byte("{broken"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestModel_Qwen3NextGenerationNative_SkipWithoutModel_Good(t *testing.T) {
	modelPath := core.Getenv("GO_MLX_QWEN3_NEXT_MODEL")
	if modelPath == "" {
		t.Skip("set GO_MLX_QWEN3_NEXT_MODEL to run native Qwen3-Next generation smoke test")
	}
	model, err := LoadAndInit(modelPath, LoadConfig{ContextLen: 256})
	if err != nil {
		t.Fatalf("LoadAndInit() error = %v", err)
	}
	defer model.Close()

	var tokens []Token
	for token := range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokens = append(tokens, token)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Generate() produced no tokens")
	}
}

// --- ResolveWeight ---

func TestModel_ResolveWeight_Direct_Good(t *testing.T) {
	a := FromValue(float32(1))
	weights := map[string]*Array{"model.norm.weight": a}

	got := ResolveWeight(weights, "model.norm.weight")
	if got != a {
		t.Error("expected direct name resolution")
	}
}

func TestModel_ResolveWeight_LanguageModelPrefix_Good(t *testing.T) {
	a := FromValue(float32(1))
	weights := map[string]*Array{"language_model.model.norm.weight": a}

	got := ResolveWeight(weights, "model.norm.weight")
	if got != a {
		t.Error("expected language_model. prefix fallback")
	}
}

func TestModel_ResolveWeight_NotFound_Bad(t *testing.T) {
	weights := map[string]*Array{}
	got := ResolveWeight(weights, "nonexistent")
	if got != nil {
		t.Error("expected nil for missing weight")
	}
}

// --- Ugly paths ---

// TestModel_LoadModel_EmptyDir_Ugly tests loadModel on an empty temporary directory.
// Should return an error mentioning config, not panic.
func TestModel_LoadModel_EmptyDir_Ugly(t *testing.T) {
	dir := t.TempDir()
	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected error for empty directory")
	}
	if !core.Contains(err.Error(), "config") {
		t.Errorf("error should mention config, got: %v", err)
	}
}

// --- helpers ---

// writeMinimalConfig writes a minimal valid config.json for testing.
func writeMinimalConfig(t *testing.T, dir string, modelType string) {
	t.Helper()
	config := `{
		"model_type": "` + modelType + `",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 32,
		"vocab_size": 100,
		"rms_norm_eps": 1e-6
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

// writeMinimalTokenizer writes a minimal valid tokenizer.json for testing.
func writeMinimalTokenizer(t testing.TB, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {
			"type": "BPE",
			"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<pad>", "special": true},
			{"id": 1, "content": "<eos>", "special": true},
			{"id": 2, "content": "<bos>", "special": true}
		]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
}

func tinyDenseDecoderWeights() map[string]*Array {
	return map[string]*Array{
		"model.embed_tokens.weight":                      seqArray(0.01, 5, 8),
		"model.layers.0.input_layernorm.weight":          seqArray(0.02, 8),
		"model.layers.0.post_attention_layernorm.weight": seqArray(0.03, 8),
		"model.layers.0.self_attn.q_proj.weight":         seqArray(0.04, 8, 8),
		"model.layers.0.self_attn.k_proj.weight":         seqArray(0.05, 4, 8),
		"model.layers.0.self_attn.v_proj.weight":         seqArray(0.06, 4, 8),
		"model.layers.0.self_attn.o_proj.weight":         seqArray(0.07, 8, 8),
		"model.layers.0.mlp.gate_proj.weight":            seqArray(0.08, 16, 8),
		"model.layers.0.mlp.up_proj.weight":              seqArray(0.09, 16, 8),
		"model.layers.0.mlp.down_proj.weight":            seqArray(0.10, 8, 16),
		"model.norm.weight":                              seqArray(0.11, 8),
		"lm_head.weight":                                 seqArray(0.12, 5, 8),
	}
}

func freeArrayMap(arrays map[string]*Array) {
	for _, array := range arrays {
		Free(array)
	}
}
