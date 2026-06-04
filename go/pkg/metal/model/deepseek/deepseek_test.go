// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deepseek

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func TestDeepSeek_LoadStagedModelValidatesMLA_Good(t *testing.T) {
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
	writeMinimalTokenizer(t, dir)

	model, err := loadStagedModel(dir, []byte(config))
	if err != nil {
		t.Fatalf("loadStagedModel(deepseek) error = %v", err)
	}
	if model.ModelType() != "deepseek" || model.NumLayers() != 2 {
		t.Fatalf("model metadata = %s/%d, want deepseek/2", model.ModelType(), model.NumLayers())
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	if model.mla.KVLoRARank != 512 || model.mla.QKHeadDim != 192 || model.mla.VHeadDim != 128 {
		t.Fatalf("DeepSeek MLA plan = %+v, want kv rank 512 qk head 192 v head 128", model.mla)
	}
	info := metal.ModelInfo{Architecture: model.ModelType(), NumLayers: model.NumLayers()}
	model.FillModelInfo(&info)
	if info.VocabSize != 32000 || info.HiddenSize != 1024 {
		t.Fatalf("FillModelInfo = %+v, want vocab=32000 hidden=1024", info)
	}
}

func TestDeepSeek_LoadStagedModelValidatesMLA_Bad(t *testing.T) {
	base := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64,
		%s
	}`
	cases := []struct {
		name string
		mla  string
		want string
	}{
		{
			name: "missing-kv-lora",
			mla:  `"qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128`,
			want: "kv_lora_rank",
		},
		{
			name: "missing-rope-split",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "v_head_dim": 128`,
			want: "qk_nope_head_dim and qk_rope_head_dim",
		},
		{
			name: "bad-qk-sum",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "qk_head_dim": 256, "v_head_dim": 128`,
			want: "qk_head_dim",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := core.Sprintf(base, tc.mla)
			_, err := loadStagedModel(t.TempDir(), []byte(config))
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("loadStagedModel(deepseek invalid MLA) error = %v, want %q", err, tc.want)
			}
		})
	}
}

func writeMinimalTokenizer(t *testing.T, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {"type": "BPE", "vocab": {"hello": 0, "<unk>": 1}, "merges": []},
		"pre_tokenizer": {"type": "ByteLevel"},
		"decoder": {"type": "ByteLevel"}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
}
