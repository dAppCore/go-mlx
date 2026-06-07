// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/internal/metaltest"
)

// --- loadModel dispatch ---

func TestModel_LoadModel_MissingConfigJSON_Bad(t *testing.T) {
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
	if !core.Contains(err.Error(), "attached drafter") ||
		!core.Contains(err.Error(), "standalone") ||
		!core.Contains(err.Error(), "LoadSpeculativePair") {
		t.Errorf("expected attached-only boundary error (registry-driven, not name-branched), got: %v", err)
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

func TestModel_ProbeModelType_OfficialGemma412BUnifiedPath_Good(t *testing.T) {
	got, err := probeModelType([]byte(`{
		"model_type": "gemma4_unified",
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_unified_text",
			"hidden_size": 3840,
			"num_hidden_layers": 48,
			"num_attention_heads": 16,
			"num_key_value_heads": 8,
			"num_global_key_value_heads": 1,
			"head_dim": 256,
			"vocab_size": 262144,
			"max_position_embeddings": 262144
		},
		"vision_config": {"model_type": "gemma4_unified_vision"},
		"audio_config": {"model_type": "gemma4_unified_audio"}
	}`))
	if err != nil {
		t.Fatalf("probeModelType() error = %v", err)
	}
	if got != "gemma4_unified" {
		t.Fatalf("probeModelType() = %q, want gemma4_unified for official 12B Unified multimodal path", got)
	}
}

func TestModel_ProbeModelType_Gemma4UnifiedTextNormalizesToText_Good(t *testing.T) {
	got, err := probeModelType([]byte(`{
		"model_type": "gemma4_unified_text",
		"architectures": ["Gemma4TextForCausalLM"],
		"hidden_size": 3840,
		"num_hidden_layers": 48,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"head_dim": 256,
		"vocab_size": 262144,
		"max_position_embeddings": 262144
	}`))
	if err != nil {
		t.Fatalf("probeModelType() error = %v", err)
	}
	if got != "gemma4_text" {
		t.Fatalf("probeModelType() = %q, want nested gemma4_unified_text metadata to load as gemma4_text", got)
	}
}

// Qwen3 + Qwen3.6 model-type dispatch + load coverage travels with the model in
// package metal/model/qwen3.
// Mixtral model-type dispatch + load coverage travels with the model in
// package metal/model/mixtral.
// GPT-OSS model-type dispatch + load coverage travels with the model in
// package metal/model/gptoss.

// Kimi model-type dispatch + load coverage travels with the model in package
// metal/model/kimi.

// DeepSeek staged load + MLA validation coverage travels with the model in
// package metal/model/deepseek.

// BERT staged load + rerank validation coverage travels with the model in
// package metal/model/bert.

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

// Qwen3 load error-path coverage travels with the model in package
// metal/model/qwen3.

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
	if cfg.RopeTheta != 10000 {
		t.Errorf("RopeTheta default = %f, want 10000 (transformers default when omitted — Qwen/long-context declare a larger base in config)", cfg.RopeTheta)
	}
	if cfg.VocabSize != 0 {
		t.Errorf("VocabSize at parse = %d, want 0 (dimension not fabricated — the dense loaders derive it from the embed tensor; 151936 is Qwen-only)", cfg.VocabSize)
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
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to run native Qwen3-Next generation smoke test")
	}
	modelPath := metaltest.HFModelPath(t, "mlx-community/Qwen3-Next*")
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
