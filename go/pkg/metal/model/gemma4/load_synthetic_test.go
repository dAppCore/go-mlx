// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4MultiModalTextConfigJSON is gemma4DenseTextConfigJSON's twin with the
// multimodal model_type id, so the loadGemma4MultiModalModel path can be driven
// by the canonical gemma4TinyWeights fixture without any vision/audio tensors.
// A text-shaped checkpoint is enough to cover the multimodal loader wrapper and
// the model-type finaliser; the aux vision/audio feature-config branches need
// their own modality weights and live in the modality test files.
func writeGemma4MultiModalTextConfig(t testing.TB, dir, modelType string) {
	t.Helper()
	config := `{
		"model_type": "` + modelType + `",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

func writeSyntheticGemma4MultiModalDir(t *testing.T, modelType string) string {
	t.Helper()
	dir := t.TempDir()
	writeGemma4MultiModalTextConfig(t, dir, modelType)
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	return dir
}

// TestLoad_loadGemma4MultiModalModel_Good drives the registrar-side multimodal
// loader (the closure metal.RegisterModelLoader("gemma4", ...) installs) on a
// tiny synthetic on-disk checkpoint. The loader wraps LoadGemma4 and then
// finalises the model-type id to the multimodal "gemma4" — the text-only
// LoadGemma4 leaves the declared id untouched, so this test pins the finaliser's
// effect that the lower-level load.go test does not see.
func TestLoad_loadGemma4MultiModalModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := writeSyntheticGemma4MultiModalDir(t, "gemma4")
	m, err := loadGemma4MultiModalModel(dir)
	if err != nil {
		t.Fatalf("loadGemma4MultiModalModel: %v", err)
	}
	defer closeGemma4(m)

	if m.ModelType() != "gemma4" {
		t.Fatalf("ModelType() = %q, want gemma4", m.ModelType())
	}
	if m.Cfg == nil || m.Cfg.ModelType != "gemma4" {
		t.Fatalf("Cfg.ModelType = %v, want gemma4", m.Cfg)
	}
	if m.NumLayers() != 2 {
		t.Fatalf("NumLayers() = %d, want 2", m.NumLayers())
	}
}

// TestLoad_loadGemma4MultiModalModel_Unified covers the gemma4_unified branch of
// the model-type finaliser: a checkpoint declaring the unified id keeps it
// through the multimodal loader rather than collapsing to plain "gemma4".
func TestLoad_loadGemma4MultiModalModel_Unified(t *testing.T) {
	requireMetalRuntime(t)

	dir := writeSyntheticGemma4MultiModalDir(t, "gemma4_unified")
	m, err := loadGemma4MultiModalModel(dir)
	if err != nil {
		t.Fatalf("loadGemma4MultiModalModel(unified): %v", err)
	}
	defer closeGemma4(m)

	if m.ModelType() != "gemma4_unified" {
		t.Fatalf("ModelType() = %q, want gemma4_unified", m.ModelType())
	}
	if m.Cfg == nil || m.Cfg.ModelType != "gemma4_unified" {
		t.Fatalf("Cfg.ModelType = %v, want gemma4_unified", m.Cfg)
	}
}

// TestLoad_loadGemma4MultiModalModel_Bad covers the loader's error surface: a
// directory with no config.json fails in LoadGemma4 before the finaliser runs,
// so the multimodal wrapper returns the wrapped load error.
func TestLoad_loadGemma4MultiModalModel_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	if _, err := loadGemma4MultiModalModel(dir); err == nil {
		t.Fatal("loadGemma4MultiModalModel() error = nil, want missing config")
	}
}

// TestLoad_finalizeGemma4MultiModalModelType_NilSafe covers the finaliser's
// nil-model and nil-config guards directly: a nil model is a no-op, and a model
// with no config still has its surface model-type id stamped.
func TestLoad_finalizeGemma4MultiModalModelType_NilSafe(t *testing.T) {
	finalizeGemma4MultiModalModelType(nil) // must not panic

	m := &Gemma4Model{}
	finalizeGemma4MultiModalModelType(m)
	if m.ModelType() != "gemma4" {
		t.Fatalf("nil-config finalise ModelType() = %q, want gemma4", m.ModelType())
	}
}
