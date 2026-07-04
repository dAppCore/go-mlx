// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deepseek

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// TestDeepSeek_LoadStagedModelTokenizerMissing_Bad covers loadStagedModel's
// tokenizer-load failure branch: the config parses, validates, and yields a
// complete MLA plan, but the checkpoint directory carries no tokenizer.json, so
// metal.LoadTokenizer fails and the loader returns a wrapped "load tokenizer"
// diagnostic rather than a half-built model. All other loadStagedModel tests
// write a tokenizer, so this is the only path that exercises the error wrap on
// lines 108-110.
func TestDeepSeek_LoadStagedModelTokenizerMissing_Bad(t *testing.T) {
	dir := t.TempDir()
	config := `{
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
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	// Deliberately no tokenizer.json — the directory is otherwise a valid
	// DeepSeek pack, so the loader reaches the tokenizer step and fails there.

	model, err := loadStagedModel(dir, []byte(config))
	if model != nil {
		t.Fatalf("loadStagedModel(no tokenizer) model = %+v, want nil", model)
	}
	if err == nil || !core.Contains(err.Error(), "load tokenizer") {
		t.Fatalf("loadStagedModel(no tokenizer) error = %v, want load-tokenizer diagnostic", err)
	}
}

// TestDeepSeek_LoadStagedModelMalformedConfig_Ugly covers loadStagedModel's
// first error gate: when the config bytes do not parse, parseStagedConfig fails
// and loadStagedModel returns that error before touching validation, the MLA
// plan, or the filesystem. The existing _Bad cases all parse and fail later, so
// this is the only path through the parse-error return at the top of the loader.
func TestDeepSeek_LoadStagedModelMalformedConfig_Ugly(t *testing.T) {
	model, err := loadStagedModel(t.TempDir(), []byte(`{"hidden_size":`))
	if model != nil {
		t.Fatalf("loadStagedModel(malformed config) model = %+v, want nil", model)
	}
	if err == nil {
		t.Fatal("loadStagedModel(malformed config) error = nil, want JSON parse error")
	}
}

// TestDeepSeek_DecodeStubsReturnNil documents the staged loader's inert-decode
// contract directly: Forward, ForwardMasked, and ApplyLoRA all return nil
// because the native sparse-expert decode kernels have not landed (the same
// contract DecodeUnavailableError reports). Callers must route through the
// diagnostic, not these stubs, until the MoE kernels exist. Pure metadata —
// no Metal runtime, no GPU allocation.
func TestDeepSeek_DecodeStubsReturnNil(t *testing.T) {
	model := &StagedModel{}

	if out := model.Forward(nil, nil); out != nil {
		t.Fatalf("Forward() = %v, want nil (no native decode kernels)", out)
	}
	if out := model.ForwardMasked(nil, nil, nil); out != nil {
		t.Fatalf("ForwardMasked() = %v, want nil (no native decode kernels)", out)
	}
	if adapter := model.ApplyLoRA(metal.LoRAConfig{}); adapter != nil {
		t.Fatalf("ApplyLoRA() = %v, want nil (no native decode kernels)", adapter)
	}
}
