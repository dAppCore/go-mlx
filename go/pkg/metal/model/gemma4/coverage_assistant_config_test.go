// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the pure assistant (MTP drafter) config validators, the JSON
// config parser, and the gguf metadata coercers. Every case here is driven by a
// crafted struct / JSON byte slice / metadata map — no checkpoint load. Helpers
// prefixed cov4ac.

// cov4acTextConfig returns a minimal-but-valid text config that
// validateGemma4AssistantConfig accepts, so each test can corrupt one field and
// assert the matching guard fires.
func cov4acTextConfig() *Gemma4TextConfig {
	return &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        8,
			NumHiddenLayers:   2,
			NumAttentionHeads: 1,
			HeadDim:           4,
			VocabSize:         10,
		},
	}
}

// cov4acAssistantConfig wraps cov4acTextConfig with valid assistant fields.
func cov4acAssistantConfig() *Gemma4AssistantConfig {
	return &Gemma4AssistantConfig{
		ModelType:          "gemma4_assistant",
		BackboneHiddenSize: 8,
		TextConfig:         cov4acTextConfig(),
	}
}

// TestAssistant_ValidateConfig_Branches walks every guard in
// validateGemma4AssistantConfig: each corruption must produce an error naming
// the offending field, and the pristine config must pass.
func TestAssistant_ValidateConfig_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// Happy path passes.
	if err := validateGemma4AssistantConfig(cov4acAssistantConfig()); err != nil {
		t.Fatalf("valid config should pass, got %v", err)
	}

	cases := []struct {
		name   string
		mutate func(*Gemma4AssistantConfig)
		want   string
	}{
		{"nil cfg", func(c *Gemma4AssistantConfig) { *c = Gemma4AssistantConfig{} }, "config is nil"},
		{"nil text", func(c *Gemma4AssistantConfig) { c.TextConfig = nil }, "config is nil"},
		{"bad model_type", func(c *Gemma4AssistantConfig) { c.ModelType = "llama" }, "unsupported model_type"},
		{"bad backbone", func(c *Gemma4AssistantConfig) { c.BackboneHiddenSize = 0 }, "backbone_hidden_size"},
		{"bad hidden", func(c *Gemma4AssistantConfig) { c.TextConfig.HiddenSize = 0 }, "hidden_size"},
		{"bad layers", func(c *Gemma4AssistantConfig) { c.TextConfig.NumHiddenLayers = 0 }, "num_hidden_layers"},
		{"bad heads", func(c *Gemma4AssistantConfig) { c.TextConfig.NumAttentionHeads = 0 }, "num_attention_heads"},
		{"bad head_dim", func(c *Gemma4AssistantConfig) { c.TextConfig.HeadDim = 0 }, "head_dim"},
		{"ordered without centroids", func(c *Gemma4AssistantConfig) {
			c.UseOrderedEmbeddings = true
			c.NumCentroids = 0
		}, "num_centroids"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := cov4acAssistantConfig()
			tc.mutate(cfg)
			err := validateGemma4AssistantConfig(cfg)
			if err == nil {
				t.Fatalf("%s: expected error, got nil", tc.name)
			}
			if !core.Contains(err.Error(), tc.want) {
				t.Fatalf("%s: error = %v, want substring %q", tc.name, err, tc.want)
			}
		})
	}

	// gemma4_unified_assistant is the other accepted model_type.
	uni := cov4acAssistantConfig()
	uni.ModelType = "gemma4_unified_assistant"
	if err := validateGemma4AssistantConfig(uni); err != nil {
		t.Fatalf("unified assistant model_type should pass, got %v", err)
	}
}

// TestAssistant_ParseConfig_Branches drives parseGemma4AssistantConfig's dark
// error/branch lines from crafted JSON: malformed JSON, an unparsable text
// config, the empty-model_type default, and the unified-assistant rename.
func TestAssistant_ParseConfig_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// Malformed JSON fails at the wrapper unmarshal.
	if _, err := parseGemma4AssistantConfig([]byte("{not json")); err == nil {
		t.Fatal("malformed JSON should fail")
	}

	// Valid wrapper JSON but a text config that parseGemma4Config rejects
	// (negative hidden size) fails at the text-config parse.
	badText := `{
		"model_type": "gemma4_assistant",
		"backbone_hidden_size": 8,
		"hidden_size": -1,
		"num_hidden_layers": 1,
		"intermediate_size": 8,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"max_position_embeddings": 16,
		"layer_types": ["full_attention"]
	}`
	if _, err := parseGemma4AssistantConfig([]byte(badText)); err == nil {
		t.Fatal("invalid text config should fail")
	}

	// Empty model_type defaults to gemma4_assistant and the config validates.
	noType := `{
		"backbone_hidden_size": 8,
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"max_position_embeddings": 16,
		"layer_types": ["full_attention"]
	}`
	cfg, err := parseGemma4AssistantConfig([]byte(noType))
	if err != nil {
		t.Fatalf("empty model_type should default + pass, got %v", err)
	}
	if cfg.ModelType != "gemma4_assistant" {
		t.Fatalf("default ModelType = %q, want gemma4_assistant", cfg.ModelType)
	}
	if cfg.TextConfig.ModelType != "gemma4_assistant" {
		t.Fatalf("text ModelType = %q, want gemma4_assistant", cfg.TextConfig.ModelType)
	}

	// gemma4_unified_assistant renames the text config to gemma4_unified_text.
	uniType := `{
		"model_type": "gemma4_unified_assistant",
		"backbone_hidden_size": 8,
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"max_position_embeddings": 16,
		"layer_types": ["full_attention"]
	}`
	uni, err := parseGemma4AssistantConfig([]byte(uniType))
	if err != nil {
		t.Fatalf("unified assistant should parse, got %v", err)
	}
	if uni.TextConfig.ModelType != "gemma4_unified_text" {
		t.Fatalf("unified text ModelType = %q, want gemma4_unified_text", uni.TextConfig.ModelType)
	}
}

// TestAssistant_GGUFMetaCoercers covers every type case in ggufMetaInt32 and
// ggufMetaFloat32, plus the absent/non-numeric zero fallbacks. The gguf parser
// yields assorted integer/float widths, so each width must coerce correctly.
func TestAssistant_GGUFMetaCoercers(t *testing.T) {
	requireMetalRuntime(t)

	intMeta := map[string]any{
		"u32": uint32(7),
		"i32": int32(8),
		"u64": uint64(9),
		"i64": int64(10),
		"int": int(11),
		"f64": float64(12),
		"str": "nope",
	}
	intWant := map[string]int32{"u32": 7, "i32": 8, "u64": 9, "i64": 10, "int": 11, "f64": 12, "str": 0, "absent": 0}
	for key, want := range intWant {
		if got := ggufMetaInt32(intMeta, key); got != want {
			t.Fatalf("ggufMetaInt32(%q) = %d, want %d", key, got, want)
		}
	}

	floatMeta := map[string]any{
		"f32": float32(1.5),
		"f64": float64(2.5),
		"u32": uint32(3),
		"i32": int32(4),
		"str": "nope",
	}
	floatWant := map[string]float32{"f32": 1.5, "f64": 2.5, "u32": 3, "i32": 4, "str": 0, "absent": 0}
	for key, want := range floatWant {
		if got := ggufMetaFloat32(floatMeta, key); got != want {
			t.Fatalf("ggufMetaFloat32(%q) = %g, want %g", key, got, want)
		}
	}
}

// TestAssistant_GGUFConfig_Defaults drives gemma4AssistantConfigFromGGUF's
// fallback branches: an omitted embedding_length_out defaults backbone to the
// hidden size, an absent sliding_window_pattern collapses to 1 (every layer
// full), and absent eps / rope freqs take their canonical defaults.
func TestAssistant_GGUFConfig_Defaults(t *testing.T) {
	requireMetalRuntime(t)

	const p = "gemma4-assistant."
	meta := map[string]any{
		"general.architecture":     "gemma4-assistant",
		p + "block_count":          uint32(2),
		p + "embedding_length":     uint32(16),
		p + "attention.head_count": uint32(2),
		p + "attention.key_length": uint32(8),
		// embedding_length_out, sliding_window_pattern, eps and rope freqs all
		// omitted so the default branches run.
	}
	cfg, err := gemma4AssistantConfigFromGGUF(meta)
	if err != nil {
		t.Fatalf("defaulted gguf config should build, got %v", err)
	}
	if cfg.BackboneHiddenSize != 16 {
		t.Fatalf("backbone default = %d, want hidden 16", cfg.BackboneHiddenSize)
	}
	for i, lt := range cfg.TextConfig.LayerTypes {
		if lt != "full_attention" {
			t.Fatalf("pattern<=0 layer %d = %q, want full_attention", i, lt)
		}
	}
	if cfg.TextConfig.RMSNormEps != 1e-6 {
		t.Fatalf("eps default = %v, want 1e-6", cfg.TextConfig.RMSNormEps)
	}
	full := cfg.TextConfig.RopeParameters["full_attention"]
	if full.RopeTheta != 1000000.0 {
		t.Fatalf("rope freq_base default = %v, want 1e6", full.RopeTheta)
	}
}

// TestAudio_NormalizeFeatureConfig_Branches drives normalizeGemma4AudioFeatureConfig's
// defaulting ladder: the nil fast-return, the bare-defaults path (empty config →
// 128 mel / 16kHz / 20ms frame / 10ms hop), and the ms-alias path where
// frame_length_ms / hop_length_ms are present but the sample counts are not.
func TestAudio_NormalizeFeatureConfig_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// nil → nil.
	if normalizeGemma4AudioFeatureConfig(nil) != nil {
		t.Fatal("nil config should normalise to nil")
	}

	// Empty config takes every default (FeatureSize<=0→128, SamplingRate<=0→16k,
	// FrameLength<=0→msToSamples(20), HopLength<=0→msToSamples(10)).
	bare := normalizeGemma4AudioFeatureConfig(&Gemma4AudioFeatureConfig{})
	if bare.FeatureSize != 128 {
		t.Fatalf("default FeatureSize = %d, want 128", bare.FeatureSize)
	}
	if bare.SamplingRate != 16_000 {
		t.Fatalf("default SamplingRate = %d, want 16000", bare.SamplingRate)
	}
	if bare.FrameLength != 320 { // 16000 * 20 / 1000
		t.Fatalf("default FrameLength = %d, want 320", bare.FrameLength)
	}
	if bare.HopLength != 160 { // 16000 * 10 / 1000
		t.Fatalf("default HopLength = %d, want 160", bare.HopLength)
	}

	// ms aliases convert to sample counts when the explicit counts are absent.
	ms := normalizeGemma4AudioFeatureConfig(&Gemma4AudioFeatureConfig{
		SamplingRate:  16_000,
		FrameLengthMs: 25.0,
		HopLengthMs:   12.5,
	})
	if ms.FrameLength != 400 { // 16000 * 25 / 1000
		t.Fatalf("ms FrameLength = %d, want 400", ms.FrameLength)
	}
	if ms.HopLength != 200 { // 16000 * 12.5 / 1000
		t.Fatalf("ms HopLength = %d, want 200", ms.HopLength)
	}
}

// cov4acValidAssistantModel builds a complete 1-layer Gemma4AssistantModel whose
// every required tensor is present, so a validate call passes. Tests clone it
// and null exactly one field to drive each missing-tensor / nil-guard branch of
// validateGemma4AssistantModel. Caller frees nothing — seqArray tensors are tiny
// and the package's other validator tests follow the same no-free convention for
// these synthetic structs.
func cov4acValidAssistantModel() *Gemma4AssistantModel {
	const hidden, backbone = 8, 8
	norm := func() *metal.RMSNormModule { return &metal.RMSNormModule{Weight: seqArray(0.1, hidden)} }
	lin := func(out, in int) *metal.Linear { return &metal.Linear{Weight: seqArray(0.1, out, in)} }
	return &Gemma4AssistantModel{
		Cfg:                &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: hidden, VocabSize: 10}},
		BackboneHiddenSize: backbone,
		EmbedTokens:        &metal.Embedding{Weight: seqArray(0.1, 10, hidden)},
		Norm:               norm(),
		PreProjection:      lin(hidden, backbone*2),
		PostProjection:     lin(backbone, hidden),
		Layers: []*Gemma4AssistantLayer{{
			InputNorm:    norm(),
			PostAttnNorm: norm(),
			PreFFNorm:    norm(),
			PostFFNorm:   norm(),
			LayerScalar:  seqArray(0.1, hidden),
			Attention: &Gemma4AssistantAttention{
				QProj:   lin(hidden, hidden),
				OProj:   lin(hidden, hidden),
				QNorm:   norm(),
				HeadDim: 4,
				NHeads:  2,
			},
			MLP: &metal.MLP{
				GateProj: lin(hidden, hidden),
				UpProj:   lin(hidden, hidden),
				DownProj: lin(hidden, hidden),
			},
		}},
	}
}

// TestAssistant_ValidateModel_Branches confirms the pristine synthetic model
// passes, then nulls one sub-field at a time to drive validateGemma4AssistantModel's
// nil-guards and missing-tensor reports. Each case asserts a non-nil error so the
// test verifies the validator actually detects that class of corruption.
func TestAssistant_ValidateModel_Branches(t *testing.T) {
	requireMetalRuntime(t)

	if err := validateGemma4AssistantModel(cov4acValidAssistantModel()); err != nil {
		t.Fatalf("complete model should validate, got %v", err)
	}

	cases := []struct {
		name   string
		mutate func(*Gemma4AssistantModel)
	}{
		{"nil model cfg", func(m *Gemma4AssistantModel) { m.Cfg = nil }},
		{"bad backbone", func(m *Gemma4AssistantModel) { m.BackboneHiddenSize = 0 }},
		{"nil embed", func(m *Gemma4AssistantModel) { m.EmbedTokens = nil }},
		{"nil norm module", func(m *Gemma4AssistantModel) { m.Norm = nil }},
		{"nil pre-projection", func(m *Gemma4AssistantModel) { m.PreProjection = nil }},
		{"nil layer", func(m *Gemma4AssistantModel) { m.Layers[0] = nil }},
		{"nil layer norm", func(m *Gemma4AssistantModel) { m.Layers[0].InputNorm = nil }},
		{"nil attention", func(m *Gemma4AssistantModel) { m.Layers[0].Attention = nil }},
		{"zero head dim", func(m *Gemma4AssistantModel) { m.Layers[0].Attention.HeadDim = 0 }},
		{"zero heads", func(m *Gemma4AssistantModel) { m.Layers[0].Attention.NHeads = 0 }},
		{"nil mlp", func(m *Gemma4AssistantModel) { m.Layers[0].MLP = nil }},
		{"bad projection shape", func(m *Gemma4AssistantModel) {
			m.PreProjection = &metal.Linear{Weight: seqArray(0.1, 99, 16)}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := cov4acValidAssistantModel()
			tc.mutate(m)
			if err := validateGemma4AssistantModel(m); err == nil {
				t.Fatalf("%s: expected validation error, got nil", tc.name)
			}
		})
	}

	// A nil model pointer hits the first guard directly.
	if err := validateGemma4AssistantModel(nil); err == nil {
		t.Fatal("nil model pointer should fail validation")
	}
}
