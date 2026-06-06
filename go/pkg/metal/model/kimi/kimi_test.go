// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// --- LoadKimi error paths ---

func TestModel_LoadKimi_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000
	}`)

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadKimi_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadKimi_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalKimiConfig(t, dir)
	writeMinimalKimiTokenizer(t, dir)

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "kimi") {
		t.Errorf("error should mention kimi, got: %v", err)
	}
}

// --- parseKimiConfig ---

func TestModel_ParseKimiConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseKimiConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("parseKimiConfig: %v", err)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-5 {
		t.Errorf("RMSNormEps default = %g, want 1e-5", cfg.RMSNormEps)
	}
	if cfg.VocabSize != 0 {
		t.Errorf("VocabSize at parse = %d, want 0 (dimension not fabricated — derived from the embed tensor at load)", cfg.VocabSize)
	}
	// head_dim inferred from hidden/heads when absent.
	if cfg.HeadDim != 128 {
		t.Errorf("HeadDim inferred = %d, want 128", cfg.HeadDim)
	}
}

func TestModel_ParseKimiConfig_QuantizationNested_Good(t *testing.T) {
	cfg, err := parseKimiConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"head_dim": 128,
		"quantization_config": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseKimiConfig: %v", err)
	}
	if cfg.Quantization == nil {
		t.Fatal("expected quantization config from quantization_config key")
	}
	if cfg.Quantization.GroupSize != 64 {
		t.Errorf("GroupSize = %d, want 64", cfg.Quantization.GroupSize)
	}
	if cfg.Quantization.Bits != 4 {
		t.Errorf("Bits = %d, want 4", cfg.Quantization.Bits)
	}
}

func TestModel_ParseKimiConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseKimiConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- KimiConfig expert sizing ---

func TestModel_KimiConfig_ExpertCount_Good(t *testing.T) {
	// num_experts wins when present.
	if got := (&KimiConfig{NumExperts: 16}).expertCount(); got != 16 {
		t.Errorf("expertCount(num_experts=16) = %d, want 16", got)
	}
	// falls back to num_local_experts, then n_routed_experts.
	if got := (&KimiConfig{NumLocalExperts: 8}).expertCount(); got != 8 {
		t.Errorf("expertCount(num_local_experts=8) = %d, want 8", got)
	}
	if got := (&KimiConfig{NRoutedExperts: 4}).expertCount(); got != 4 {
		t.Errorf("expertCount(n_routed_experts=4) = %d, want 4", got)
	}
	// default when none set.
	if got := (&KimiConfig{}).expertCount(); got != 8 {
		t.Errorf("expertCount(default) = %d, want 8", got)
	}
}

func TestModel_KimiConfig_TopK_Good(t *testing.T) {
	if got := (&KimiConfig{NumExpertsPerTok: 6}).topK(); got != 6 {
		t.Errorf("topK(num_experts_per_tok=6) = %d, want 6", got)
	}
	if got := (&KimiConfig{MoETopK: 3}).topK(); got != 3 {
		t.Errorf("topK(moe_topk=3) = %d, want 3", got)
	}
	if got := (&KimiConfig{}).topK(); got != 2 {
		t.Errorf("topK(default) = %d, want 2", got)
	}
}

// --- MoETextRuntimeAvailable (relocated from package metal) ---

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	m := &KimiModel{
		Layers: []*KimiDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &KimiMoEBlock{
				Router:        router,
				Experts:       []*KimiExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !m.MoETextRuntimeAvailable() {
		t.Fatal("KimiModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := m.MoETextDecodeFamily(); got != "kimi" {
		t.Fatalf("MoETextDecodeFamily() = %q, want kimi", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&KimiModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty KimiModel.MoETextRuntimeAvailable() = true, want false")
	}
	incomplete := &KimiModel{Layers: []*KimiDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete KimiModel.MoETextRuntimeAvailable() = true, want false")
	}
}

// --- helpers ---

func moeReadyRuntimeParts(t *testing.T) (*metal.MoERouter, *metal.MoESwiGLUExperts, func()) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	routerWeight := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	gate := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil)}
	up := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{0.5, 0, 0, 0.5}, 2, 2), nil)}
	down := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil)}
	experts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		t.Fatal("NewMoESwiGLUExpertsFromLinears() ok = false, want true")
	}
	metal.Materialize(routerWeight)
	cleanup := func() {
		metal.Free(routerWeight)
		metal.FreeMoESwiGLUExperts(experts)
	}
	return &metal.MoERouter{Weight: routerWeight}, experts, cleanup
}

func writeMinimalKimiConfig(t *testing.T, dir string) {
	t.Helper()
	config := `{
		"model_type": "kimi",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"vocab_size": 100,
		"rms_norm_eps": 1e-5,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

func writeMinimalKimiTokenizer(t *testing.T, dir string) {
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
