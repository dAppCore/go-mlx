// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

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
	writeMinimalGemma3Config(t, dir, "gemma3")
	writeMinimalGemma3Tokenizer(t, dir)

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Errorf("error should mention safetensors, got: %v", err)
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
	if cfg.VocabSize != 0 {
		t.Errorf("VocabSize at parse = %d, want 0 (dimension not fabricated — derived from the embed tensor at load)", cfg.VocabSize)
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

func TestModel_ParseConfig_PreservesModelType_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"model_type": "gemma2",
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 128
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.ModelType != "gemma2" {
		t.Fatalf("ModelType = %q, want gemma2", cfg.ModelType)
	}

	cfg, err = parseConfig([]byte(`{
		"model_type": "gemma2",
		"text_config": {
			"hidden_size": 2048,
			"num_hidden_layers": 16,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig nested: %v", err)
	}
	if cfg.ModelType != "gemma2" {
		t.Fatalf("nested ModelType = %q, want gemma2", cfg.ModelType)
	}
}

func TestModel_ParseConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseConfig([]byte("not json"))
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

// writeMinimalGemma3Config writes a minimal valid config.json for load tests.
func writeMinimalGemma3Config(t *testing.T, dir string, modelType string) {
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

// writeMinimalGemma3Tokenizer writes a minimal valid tokenizer.json for load tests.
func writeMinimalGemma3Tokenizer(t *testing.T, dir string) {
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
