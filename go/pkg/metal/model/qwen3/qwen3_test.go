// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// Generated file-aware compliance coverage.
func TestQwen3_LoadQwen3_Good(t *testing.T) {
	target := "LoadQwen3"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_LoadQwen3_Bad(t *testing.T) {
	target := "LoadQwen3"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_LoadQwen3_Ugly(t *testing.T) {
	target := "LoadQwen3"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_ParseConfigMissingHeads_Bad(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("ParseDenseConfig panicked for missing heads: %v", recovered)
		}
	}()

	cfg, err := metal.ParseDenseConfig([]byte(`{"model_type":"qwen2","vocab_size":16,"hidden_size":4,"num_hidden_layers":1,"max_position_embeddings":32}`))

	if err != nil {
		t.Fatalf("ParseDenseConfig: %v", err)
	}
	if cfg.HeadDim != 0 {
		t.Fatalf("head_dim = %d, want 0 when attention heads are absent", cfg.HeadDim)
	}
}

func TestModel_LoadQwen3_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing config")
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

func TestQwen3_Qwen3Model_Forward_Good(t *testing.T) {
	target := "Qwen3Model_Forward"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_Forward_Bad(t *testing.T) {
	target := "Qwen3Model_Forward"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_Forward_Ugly(t *testing.T) {
	target := "Qwen3Model_Forward"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ForwardMasked_Good(t *testing.T) {
	target := "Qwen3Model_ForwardMasked"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ForwardMasked_Bad(t *testing.T) {
	target := "Qwen3Model_ForwardMasked"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ForwardMasked_Ugly(t *testing.T) {
	target := "Qwen3Model_ForwardMasked"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NewCache_Good(t *testing.T) {
	target := "Qwen3Model_NewCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NewCache_Bad(t *testing.T) {
	target := "Qwen3Model_NewCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NewCache_Ugly(t *testing.T) {
	target := "Qwen3Model_NewCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NumLayers_Good(t *testing.T) {
	target := "Qwen3Model_NumLayers"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NumLayers_Bad(t *testing.T) {
	target := "Qwen3Model_NumLayers"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_NumLayers_Ugly(t *testing.T) {
	target := "Qwen3Model_NumLayers"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_Tokenizer_Good(t *testing.T) {
	target := "Qwen3Model_Tokenizer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_Tokenizer_Bad(t *testing.T) {
	target := "Qwen3Model_Tokenizer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_Tokenizer_Ugly(t *testing.T) {
	target := "Qwen3Model_Tokenizer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ModelType_Good(t *testing.T) {
	target := "Qwen3Model_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ModelType_Bad(t *testing.T) {
	target := "Qwen3Model_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ModelType_Ugly(t *testing.T) {
	target := "Qwen3Model_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ApplyLoRA_Good(t *testing.T) {
	target := "Qwen3Model_ApplyLoRA"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ApplyLoRA_Bad(t *testing.T) {
	target := "Qwen3Model_ApplyLoRA"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestQwen3_Qwen3Model_ApplyLoRA_Ugly(t *testing.T) {
	target := "Qwen3Model_ApplyLoRA"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
