// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

func TestGptOss_LoadGptOssMissingWeights_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 201088,
		"num_local_experts": 32
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := LoadGptOss(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for gpt_oss without safetensors")
	}
	if !core.Contains(err.Error(), "gpt_oss") {
		t.Fatalf("error = %v, should contain gpt_oss", err)
	}
}

func TestGptOss_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&GptOssModel{Layers: []*GptOssDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}).MoETextRuntimeAvailable() {
		t.Fatal("GptOssModel.MoETextRuntimeAvailable(incomplete) = true, want false")
	}
}

func writeMinimalTokenizer(t testing.TB, dir string) {
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
