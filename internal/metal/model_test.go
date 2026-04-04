//go:build darwin && arm64

package metal

import (
	"testing"

	"dappco.re/go/core"

	coreio "forge.lthn.ai/core/go-io"
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

// --- LoadGemma3 error paths ---

func TestModel_LoadGemma3_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma3",
		"hidden_size": 1152,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"vocab_size": 1000
	}`)

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadGemma3_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadGemma3_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "gemma3")
	writeMinimalTokenizer(t, dir)

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Errorf("error should mention safetensors, got: %v", err)
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

// --- parseConfig ---

func TestModel_ParseConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 128
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.RopeLocalBaseFreq != 10000 {
		t.Errorf("RopeLocalBaseFreq default = %f, want 10000", cfg.RopeLocalBaseFreq)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Errorf("RMSNormEps default = %f, want 1e-6", cfg.RMSNormEps)
	}
	if cfg.SlidingWindowPattern != 6 {
		t.Errorf("SlidingWindowPattern default = %d, want 6", cfg.SlidingWindowPattern)
	}
	if cfg.VocabSize != 262208 {
		t.Errorf("VocabSize default = %d, want 262208", cfg.VocabSize)
	}
}

func TestModel_ParseConfig_QuantizationTopLevel_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"head_dim": 128,
		"quantization": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.Quantization == nil {
		t.Fatal("expected quantization config")
	}
	if cfg.Quantization.GroupSize != 64 {
		t.Errorf("GroupSize = %d, want 64", cfg.Quantization.GroupSize)
	}
	if cfg.Quantization.Bits != 4 {
		t.Errorf("Bits = %d, want 4", cfg.Quantization.Bits)
	}
}

func TestModel_ParseConfig_NestedTextConfig_Good(t *testing.T) {
	// Multimodal Gemma3 has text_config nested inside a wrapper.
	cfg, err := parseConfig([]byte(`{
		"model_type": "gemma3",
		"text_config": {
			"hidden_size": 2048,
			"num_hidden_layers": 16,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"vocab_size": 262144
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want 2048", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 16 {
		t.Errorf("NumHiddenLayers = %d, want 16", cfg.NumHiddenLayers)
	}
}

func TestModel_ParseConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- parseQwen3Config ---

func TestModel_ParseQwen3Config_Defaults_Good(t *testing.T) {
	cfg, err := parseQwen3Config([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("parseQwen3Config: %v", err)
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

func TestModel_ParseQwen3Config_InvalidJSON_Bad(t *testing.T) {
	_, err := parseQwen3Config([]byte("{broken"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- isLayerSliding ---

func TestModel_IsLayerSliding_Good(t *testing.T) {
	// Pattern=6: every 6th layer is NOT sliding (global attention).
	// Layer 5 (index=5, i+1=6) → 6%6=0 → not sliding (global)
	// Layer 0 (index=0, i+1=1) → 1%6=1 → sliding
	tests := []struct {
		idx     int32
		pattern int32
		want    bool
	}{
		{0, 6, true},   // layer 1: 1%6=1 → sliding
		{4, 6, true},   // layer 5: 5%6=5 → sliding
		{5, 6, false},  // layer 6: 6%6=0 → global
		{11, 6, false}, // layer 12: 12%6=0 → global
		{0, 0, false},  // pattern=0 → no sliding
		{0, -1, false}, // pattern<0 → no sliding
	}
	for _, tt := range tests {
		got := isLayerSliding(tt.idx, tt.pattern)
		if got != tt.want {
			t.Errorf("isLayerSliding(%d, %d) = %v, want %v", tt.idx, tt.pattern, got, tt.want)
		}
	}
}

// --- resolveWeight ---

func TestModel_ResolveWeight_Direct_Good(t *testing.T) {
	a := FromValue(float32(1))
	weights := map[string]*Array{"model.norm.weight": a}

	got := resolveWeight(weights, "model.norm.weight")
	if got != a {
		t.Error("expected direct name resolution")
	}
}

func TestModel_ResolveWeight_LanguageModelPrefix_Good(t *testing.T) {
	a := FromValue(float32(1))
	weights := map[string]*Array{"language_model.model.norm.weight": a}

	got := resolveWeight(weights, "model.norm.weight")
	if got != a {
		t.Error("expected language_model. prefix fallback")
	}
}

func TestModel_ResolveWeight_NotFound_Bad(t *testing.T) {
	weights := map[string]*Array{}
	got := resolveWeight(weights, "nonexistent")
	if got != nil {
		t.Error("expected nil for missing weight")
	}
}

// --- Ugly paths ---

// TestModel_ParseConfig_NullBytes_Ugly tests parseConfig with null bytes in input.
// Should return a parse error, not panic.
func TestModel_ParseConfig_NullBytes_Ugly(t *testing.T) {
	_, err := parseConfig([]byte("\x00\x00\x00"))
	if err == nil {
		t.Fatal("expected error for null-byte input")
	}
}

// TestModel_ParseConfig_TruncatedJSON_Ugly tests parseConfig with truncated JSON.
// Should return a parse error, not panic.
func TestModel_ParseConfig_TruncatedJSON_Ugly(t *testing.T) {
	_, err := parseConfig([]byte(`{"hidden_size": 102`))
	if err == nil {
		t.Fatal("expected error for truncated JSON")
	}
}

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
func writeMinimalTokenizer(t *testing.T, dir string) {
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
