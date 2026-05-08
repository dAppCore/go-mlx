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
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
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
	if pack.QuantType != "q4_k" || pack.QuantFamily != "qk" || pack.Quantization == nil || len(pack.Quantization.TensorTypes) != 1 {
		t.Fatalf("quant details = type:%q family:%q details:%+v", pack.QuantType, pack.QuantFamily, pack.Quantization)
	}
	if pack.GGUF == nil || pack.GGUF.TensorCount != 2 {
		t.Fatalf("GGUF metadata = %+v, want 2 tensors", pack.GGUF)
	}
}

func TestInspectModelPack_SafetensorsQwen3Next_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "qwen3_next")

	pack, err := InspectModelPack(dir, WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen3_next" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported qwen3_next", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable || pack.RequiresPythonConversion {
		t.Fatalf("NativeLoadable=%v RequiresPythonConversion=%v, want native/no conversion", pack.NativeLoadable, pack.RequiresPythonConversion)
	}
	if pack.ChatTemplateSource != ModelPackChatTemplateNative || pack.ChatTemplate != "qwen" {
		t.Fatalf("chat template = source:%q name:%q, want native qwen", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsQwen3MoEArchitectureFallback_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Qwen3MoeForCausalLM"],
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 32768,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 768
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen3_moe" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported qwen3_moe", pack.Architecture, pack.SupportedArchitecture)
	}
	if pack.NativeLoadable || !pack.HasIssue(ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("native/runtime = loadable:%v issues:%+v, want recognized but runtime-gated MoE", pack.NativeLoadable, pack.Issues)
	}
	if pack.ChatTemplate != "qwen" {
		t.Fatalf("ChatTemplate = %q, want qwen", pack.ChatTemplate)
	}
}

func TestInspectModelPack_GGUFQuantizationFlowsToMemoryPlan_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	ggufPath := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"},
			{Key: "general.file_type", ValueType: ggufValueTypeUint32, Value: uint32(15)},
		},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}}},
	)

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	plan := PlanMemory(MemoryPlanInput{
		Device: DeviceInfo{MemorySize: 96 * MemoryGiB, MaxRecommendedWorkingSetSize: 86 * MemoryGiB},
		Pack:   &pack,
	})
	if plan.ModelQuantization != 4 || plan.ModelQuantizationType != "q4_k_m" || plan.ModelQuantizationFamily != "qk" {
		t.Fatalf("memory quantization = %+v", plan)
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

func TestValidateModelPack_GGUFInvalidTensorMetadata_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 2048,
		"num_hidden_layers": 28
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{127, 128}}},
	)

	pack, err := ValidateModelPack(dir)
	if err == nil {
		t.Fatal("expected validation error for invalid GGUF tensor metadata")
	}
	if !pack.HasIssue(ModelPackIssueInvalidGGUF) {
		t.Fatalf("issues = %+v, want invalid GGUF", pack.Issues)
	}
}
