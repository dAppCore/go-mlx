// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Coverage companions for qwen36_staged.go + qwen36_moe_staged.go — they drive
// the staged loaders' own error arms (parse / validate / tokenizer failures),
// the text-config re-resolution + quantization-copy branches of the staged
// config parser, and the MoE staged cache-plan rebuild failure. All synthetic
// (temp-dir configs + hand-built structs), no live model.

package qwen3

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
)

// TestQwen36_LoadStaged_ParseError_Bad covers loadQwen36StagedModel's parse arm:
// malformed config bytes never reach validation.
func TestQwen36_LoadStaged_ParseError_Bad(t *testing.T) {
	if _, err := loadQwen36StagedModel(t.TempDir(), []byte("{broken")); err == nil {
		t.Fatal("loadQwen36StagedModel(malformed config) = nil, want a parse error")
	}
}

// TestQwen36_LoadStaged_ValidateError_Bad covers loadQwen36StagedModel's validate
// arm: a config that parses as a qwen3_6 family member but is missing the head
// counts validate() requires.
func TestQwen36_LoadStaged_ValidateError_Bad(t *testing.T) {
	cfg := []byte(`{
		"model_type": "qwen3_5",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"vocab_size": 100,
		"max_position_embeddings": 4096
	}`)
	if _, err := loadQwen36StagedModel(t.TempDir(), cfg); err == nil {
		t.Fatal("loadQwen36StagedModel(no head counts) = nil, want a validate error")
	}
}

// TestQwen36_LoadStaged_TokenizerError_Bad covers loadQwen36StagedModel's
// tokenizer arm: a fully valid hybrid config but no tokenizer.json in the
// directory, so the post-validation tokenizer load fails.
func TestQwen36_LoadStaged_TokenizerError_Bad(t *testing.T) {
	cfg := []byte(`{
		"model_type": "qwen3_5",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 100,
		"max_position_embeddings": 4096,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	_, err := loadQwen36StagedModel(t.TempDir(), cfg) // no tokenizer.json
	if err == nil {
		t.Fatal("loadQwen36StagedModel(no tokenizer) = nil, want a tokenizer-load error")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want a tokenizer-load failure", err)
	}
}

// TestQwen36_ParseStagedConfig_TextConfigResolve_Good covers parseQwen36Staged
// Config's text-config branch: with no top-level model_type or architectures,
// the family is resolved from text_config.model_type after applyTextConfig (the
// detected == "" re-resolution), and the text-config quantization is copied
// across (the applyTextConfig quant branch).
func TestQwen36_ParseStagedConfig_TextConfigResolve_Good(t *testing.T) {
	cfg, err := parseQwen36StagedConfig([]byte(`{
		"text_config": {
			"model_type": "qwen3_5_text",
			"hidden_size": 16,
			"intermediate_size": 32,
			"num_hidden_layers": 2,
			"num_attention_heads": 4,
			"num_key_value_heads": 2,
			"head_dim": 4,
			"vocab_size": 100,
			"max_position_embeddings": 4096,
			"layer_types": ["linear_attention", "full_attention"],
			"quantization": {"bits": 4, "group_size": 64}
		}
	}`))
	if err != nil {
		t.Fatalf("parseQwen36StagedConfig(text-config family) error = %v", err)
	}
	if cfg.ModelType != "qwen3_6" {
		t.Fatalf("ModelType = %q, want qwen3_6 resolved from text_config", cfg.ModelType)
	}
	if cfg.HiddenSize != 16 || cfg.VocabSize != 100 {
		t.Fatalf("text-config dims not applied: hidden=%d vocab=%d", cfg.HiddenSize, cfg.VocabSize)
	}
	if cfg.Quantization.Bits != 4 || cfg.Quantization.GroupSize != 64 {
		t.Fatalf("text-config quantization not copied: %+v", cfg.Quantization)
	}
}

// TestQwen36_LoadMoEStaged_TokenizerError_Bad covers loadQwen36MoEStagedModel's
// tokenizer arm: a valid qwen3_6_moe config but no tokenizer.json.
func TestQwen36_LoadMoEStaged_TokenizerError_Bad(t *testing.T) {
	cfg := []byte(`{
		"model_type": "qwen3_6_moe",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"num_experts": 8,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 32,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	_, err := loadQwen36MoEStagedModel(t.TempDir(), cfg) // no tokenizer.json
	if err == nil {
		t.Fatal("loadQwen36MoEStagedModel(no tokenizer) = nil, want a tokenizer-load error")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want a tokenizer-load failure", err)
	}
}

// TestQwen36_MoEStagedModel_CachePlanRebuildFails_Ugly covers the rebuild-failure
// arm of qwen36MoEStagedModel.qwen36HybridCachePlan: the config is non-nil (so it
// passes the nil guard) but the empty plan forces a rebuild, and the layer types
// cannot form a valid hybrid plan, so the rebuild errors and the model reports no
// caches with ok=false (rather than the nil-config short circuit the sibling test
// covers).
func TestQwen36_MoEStagedModel_CachePlanRebuildFails_Ugly(t *testing.T) {
	cfg := &metal.DenseConfig{}
	cfg.NumHiddenLayers = 2
	cfg.LayerTypes = []string{"linear_attention", "mystery_attention"}
	m := &qwen36MoEStagedModel{config: cfg} // non-nil config, empty plan → rebuild attempt fails

	if caches := m.NewCache(); caches != nil {
		t.Fatalf("NewCache(bad layer types, non-nil config) = %v, want nil", caches)
	}
	if _, ok := m.HybridAttentionCachePlan(); ok {
		t.Error("HybridAttentionCachePlan(bad layer types) ok = true, want false")
	}
}

// TestQwen3_CloseQwen3_NilLayerEntry_Ugly closes a dense Qwen3Model whose Layers
// slice has a nil entry, driving the nil-layer continue guard of closeQwen3 that
// the populated-layer close tests do not reach.
func TestQwen3_CloseQwen3_NilLayerEntry_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	embed := &metal.Embedding{Weight: metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)}
	norm := &metal.RMSNormModule{Weight: metal.FromValues([]float32{1, 1}, 2)}
	metal.Materialize(embed.Weight, norm.Weight)
	m := &Qwen3Model{
		EmbedTokens: embed,
		Norm:        norm,
		Output:      embed.AsLinear(),
		Layers:      []*metal.DenseDecoderLayer{nil}, // nil entry → continue guard
	}
	closeQwen3(m) // must not panic on the nil layer
	if embed.Weight.Valid() {
		t.Error("embedding weight should be freed")
	}
}

// TestQwen3MoE_MoETextRuntimeAvailable_NilModel_Bad covers the nil-receiver guard
// of Qwen3MoEModel.MoETextRuntimeAvailable: a nil model reports unavailable
// rather than dereferencing.
func TestQwen3MoE_MoETextRuntimeAvailable_NilModel_Bad(t *testing.T) {
	var m *Qwen3MoEModel
	if m.MoETextRuntimeAvailable() {
		t.Fatal("nil Qwen3MoEModel.MoETextRuntimeAvailable() = true, want false")
	}
}
