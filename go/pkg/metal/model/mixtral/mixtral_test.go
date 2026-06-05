// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// --- LoadMixtral error paths ---

func TestModel_LoadMixtral_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "mixtral",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000
	}`)

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadMixtral_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadMixtral_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalMixtralConfig(t, dir)
	writeMinimalMixtralTokenizer(t, dir)

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "mixtral") {
		t.Errorf("error should mention mixtral, got: %v", err)
	}
}

// --- parseMixtralConfig ---

func TestModel_ParseMixtralConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseMixtralConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("parseMixtralConfig: %v", err)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-5 {
		t.Errorf("RMSNormEps default = %g, want 1e-5", cfg.RMSNormEps)
	}
	if cfg.VocabSize != 32000 {
		t.Errorf("VocabSize default = %d, want 32000", cfg.VocabSize)
	}
	if cfg.NumLocalExperts != 8 {
		t.Errorf("NumLocalExperts default = %d, want 8", cfg.NumLocalExperts)
	}
	if cfg.NumExpertsPerTok != 2 {
		t.Errorf("NumExpertsPerTok default = %d, want 2", cfg.NumExpertsPerTok)
	}
	// head_dim inferred from hidden/heads when absent.
	if cfg.HeadDim != 128 {
		t.Errorf("HeadDim inferred = %d, want 128", cfg.HeadDim)
	}
}

func TestModel_ParseMixtralConfig_QuantizationNested_Good(t *testing.T) {
	cfg, err := parseMixtralConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"head_dim": 128,
		"quantization_config": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseMixtralConfig: %v", err)
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

func TestModel_ParseMixtralConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseMixtralConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// --- mixtralMoELayerMask ---

func TestModel_MixtralMoELayerMask_Good(t *testing.T) {
	// SparseStep<=0 → every layer is MoE.
	dense := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 3, SparseStep: 0})
	for i, m := range dense {
		if !m {
			t.Errorf("SparseStep=0 layer %d = %v, want true", i, m)
		}
	}
	// SparseStep=2 → only every 2nd layer (i%2==1) is MoE.
	stepped := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 4, SparseStep: 2})
	want := []bool{false, true, false, true}
	for i := range want {
		if stepped[i] != want[i] {
			t.Errorf("SparseStep=2 layer %d = %v, want %v", i, stepped[i], want[i])
		}
	}
}

// --- MoETextRuntimeAvailable (relocated from package metal) ---

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	m := &MixtralModel{
		Layers: []*MixtralDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &MixtralMoEBlock{
				Router:        router,
				Experts:       []*MixtralExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !m.MoETextRuntimeAvailable() {
		t.Fatal("MixtralModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := m.MoETextDecodeFamily(); got != "mixtral" {
		t.Fatalf("MoETextDecodeFamily() = %q, want mixtral", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&MixtralModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
	incomplete := &MixtralModel{Layers: []*MixtralDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete MixtralModel.MoETextRuntimeAvailable() = true, want false")
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

func writeMinimalMixtralConfig(t *testing.T, dir string) {
	t.Helper()
	config := `{
		"model_type": "mixtral",
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

func writeMinimalMixtralTokenizer(t *testing.T, dir string) {
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
