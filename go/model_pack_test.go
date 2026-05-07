// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

const modelPackTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "▁": 4,
      "he": 5,
      "ll": 6
    },
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

func writeGoodSafetensorsPack(t *testing.T, dir string, modelType string) {
	t.Helper()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), core.Sprintf(`{
		"model_type": %q,
		"vocab_size": 262208,
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`, modelType))
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")
}

func TestInspectModelPack_SafetensorsGemma4_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	pack, err := InspectModelPack(dir, WithPackQuantization(4), WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Format != ModelPackFormatSafetensors {
		t.Fatalf("Format = %q, want safetensors", pack.Format)
	}
	if pack.Architecture != "gemma4_text" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported gemma4_text", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable || pack.RequiresPythonConversion {
		t.Fatalf("NativeLoadable=%v RequiresPythonConversion=%v, want native/no conversion", pack.NativeLoadable, pack.RequiresPythonConversion)
	}
	if !pack.HasTokenizer || !pack.HasChatTemplate || pack.ChatTemplateSource != ModelPackChatTemplateNative {
		t.Fatalf("tokenizer/chat = tokenizer:%v template:%v source:%q", pack.HasTokenizer, pack.HasChatTemplate, pack.ChatTemplateSource)
	}
	if pack.QuantBits != 4 || pack.QuantGroup != 64 || pack.ContextLength != 131072 {
		t.Fatalf("metadata = quant %d group %d ctx %d", pack.QuantBits, pack.QuantGroup, pack.ContextLength)
	}
}

func TestInspectModelPack_GGUFQwen3_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	ggufPath := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"},
			{Key: "qwen3.context_length", ValueType: ggufValueTypeUint32, Value: uint32(40960)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{128, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{128, 128}},
		},
	)

	pack, err := InspectModelPack(ggufPath, WithPackQuantization(4), WithPackMaxContextLength(65536))
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Format != ModelPackFormatGGUF {
		t.Fatalf("Format = %q, want gguf", pack.Format)
	}
	if pack.Architecture != "qwen3" || pack.QuantBits != 4 || pack.ContextLength != 40960 {
		t.Fatalf("metadata = arch %q quant %d ctx %d", pack.Architecture, pack.QuantBits, pack.ContextLength)
	}
	if pack.GGUF == nil || pack.GGUF.TensorCount != 2 {
		t.Fatalf("GGUF metadata = %+v, want 2 tensors", pack.GGUF)
	}
}

func TestValidateModelPack_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := ValidateModelPack(dir)
	if err == nil {
		t.Fatal("expected validation error for missing tokenizer")
	}
	if !pack.HasIssue(ModelPackIssueMissingTokenizer) {
		t.Fatalf("issues = %+v, want missing tokenizer", pack.Issues)
	}
}

func TestValidateModelPack_QuantizationAndContext_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	pack, err := ValidateModelPack(dir, WithPackQuantization(8), WithPackMaxContextLength(8192))
	if err == nil {
		t.Fatal("expected validation error for quantization/context mismatch")
	}
	if !pack.HasIssue(ModelPackIssueQuantizationMismatch) || !pack.HasIssue(ModelPackIssueContextTooLarge) {
		t.Fatalf("issues = %+v, want quantization mismatch and context too large", pack.Issues)
	}
}
