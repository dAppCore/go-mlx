// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/quant/codebook"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/model/minimax/m2"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
	"dappco.re/go/inference/quant/autoround"
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

	pack, err := Inspect(dir, mp.WithPackQuantization(4), mp.WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Format != mp.ModelPackFormatSafetensors {
		t.Fatalf("Format = %q, want safetensors", pack.Format)
	}
	if pack.Architecture != "gemma4_text" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported gemma4_text", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable {
		t.Fatalf("NativeLoadable=%v, want native/no conversion", pack.NativeLoadable)
	}
	if !pack.HasTokenizer || !pack.HasChatTemplate || pack.ChatTemplateSource != mp.ModelPackChatTemplateNative {
		t.Fatalf("tokenizer/chat = tokenizer:%v template:%v source:%q", pack.HasTokenizer, pack.HasChatTemplate, pack.ChatTemplateSource)
	}
	if pack.QuantBits != 4 || pack.QuantGroup != 64 || pack.ContextLength != 131072 {
		t.Fatalf("metadata = quant %d group %d ctx %d", pack.QuantBits, pack.QuantGroup, pack.ContextLength)
	}
}

func TestInspectModelPack_OfficialGemma4ConditionalTextPath_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262208,
			"hidden_size": 2048,
			"num_hidden_layers": 26,
			"max_position_embeddings": 131072
		},
		"vision_config": {
			"hidden_size": 1152
		},
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6), mp.WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if pack.Architecture != "gemma4_text" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported gemma4_text text path", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable {
		t.Fatalf("NativeLoadable=%v, want native text path/no conversion", pack.NativeLoadable)
	}
	if pack.ChatTemplate != "gemma4" || pack.ChatTemplateSource != mp.ModelPackChatTemplateNative {
		t.Fatalf("chat template = %q source=%q, want native gemma4", pack.ChatTemplate, pack.ChatTemplateSource)
	}
	if pack.QuantBits != 6 || pack.QuantGroup != 64 || pack.ContextLength != 131072 {
		t.Fatalf("metadata = quant %d group %d ctx %d", pack.QuantBits, pack.QuantGroup, pack.ContextLength)
	}
}

func TestInspectModelPack_Gemma4AssistantAlias_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_assistant",
		"architectures": ["Gemma4AssistantForCausalLM"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 256,
			"num_hidden_layers": 4,
			"max_position_embeddings": 131072
		}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if pack.Architecture != "gemma4_assistant" || !pack.SupportedArchitecture || !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("architecture = %q supported=%v native=%v issues=%+v, want native attached gemma4_assistant", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable, pack.Issues)
	}
	if pack.HasChatTemplate || pack.ChatTemplate != "" {
		t.Fatalf("chat template = has:%v name:%q, want no standalone assistant chat template", pack.HasChatTemplate, pack.ChatTemplate)
	}
	if pack.NumLayers != 4 || pack.HiddenSize != 256 || pack.ContextLength != 131072 {
		t.Fatalf("metadata = layers:%d hidden:%d ctx:%d, want assistant text_config metadata", pack.NumLayers, pack.HiddenSize, pack.ContextLength)
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
			{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "qwen3"},
			{Key: "qwen3.context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(40960)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
		},
	)

	pack, err := Inspect(ggufPath, mp.WithPackQuantization(4), mp.WithPackMaxContextLength(98304))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Format != mp.ModelPackFormatGGUF {
		t.Fatalf("Format = %q, want gguf", pack.Format)
	}
	if pack.Architecture != "qwen3" || pack.QuantBits != 4 || pack.ContextLength != 40960 {
		t.Fatalf("metadata = arch %q quant %d ctx %d", pack.Architecture, pack.QuantBits, pack.ContextLength)
	}
	quant, _ := pack.Quantization.(*gguf.QuantizationInfo)
	if pack.QuantType != "q4_k" || pack.QuantFamily != "qk" || quant == nil || len(quant.TensorTypes) != 1 {
		t.Fatalf("quant details = type:%q family:%q details:%+v", pack.QuantType, pack.QuantFamily, quant)
	}
	ggufInfo, _ := pack.GGUF.(*gguf.Info)
	if ggufInfo == nil || ggufInfo.TensorCount != 2 {
		t.Fatalf("GGUF metadata = %+v, want 2 tensors", ggufInfo)
	}
}

func TestInspectModelPack_WeightAndConfigEdgeCases_Bad(t *testing.T) {
	t.Run("mixed_weights", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
		writeModelPackFile(t, core.PathJoin(dir, "model.gguf"), "stub")

		pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect() error = %v", err)
		}
		if pack.Format != mp.ModelPackFormatMixed || !pack.HasIssue(mp.ModelPackIssueMixedWeightFormats) {
			t.Fatalf("pack = %+v, want mixed weight issue", pack)
		}
	})

	t.Run("multiple_gguf", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "a.gguf"), "stub")
		writeModelPackFile(t, core.PathJoin(dir, "b.gguf"), "stub")

		pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect() error = %v", err)
		}
		if pack.Format != mp.ModelPackFormatGGUF || !pack.HasIssue(mp.ModelPackIssueMultipleGGUF) {
			t.Fatalf("pack = %+v, want multiple GGUF issue", pack)
		}
	})

	t.Run("missing_and_invalid_config", func(t *testing.T) {
		missing := t.TempDir()
		writeModelPackFile(t, core.PathJoin(missing, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(missing, "model.safetensors"), "stub")
		pack, err := Inspect(missing, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect(missing config) error = %v", err)
		}
		if !pack.HasIssue(mp.ModelPackIssueMissingConfig) || !pack.HasIssue(mp.ModelPackIssueMissingArchitecture) {
			t.Fatalf("issues = %+v, want missing config and architecture", pack.Issues)
		}

		invalid := t.TempDir()
		writeModelPackFile(t, core.PathJoin(invalid, "config.json"), "{")
		writeModelPackFile(t, core.PathJoin(invalid, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(invalid, "model.safetensors"), "stub")
		pack, err = Inspect(invalid, mp.WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("Inspect(invalid config) error = %v", err)
		}
		if !pack.HasIssue(mp.ModelPackIssueInvalidConfig) {
			t.Fatalf("issues = %+v, want invalid config", pack.Issues)
		}
	})
}

func TestModelPackChatTemplateParsing_GoodBad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer_config.json")

	writeModelPackFile(t, path, `{"chat_template":"  {{ messages }}  "}`)
	template, ok, err := readTokenizerChatTemplate(path)
	if err != nil || !ok || template != "{{ messages }}" {
		t.Fatalf("readTokenizerChatTemplate(string) = %q/%v/%v", template, ok, err)
	}

	writeModelPackFile(t, path, `{"chat_template":[{"name":"default"}]}`)
	template, ok, err = readTokenizerChatTemplate(path)
	if err != nil || !ok || template != "named_chat_templates" {
		t.Fatalf("readTokenizerChatTemplate(named) = %q/%v/%v", template, ok, err)
	}

	writeModelPackFile(t, path, `{"chat_template":""}`)
	template, ok, err = readTokenizerChatTemplate(path)
	if err != nil || ok || template != "" {
		t.Fatalf("readTokenizerChatTemplate(empty) = %q/%v/%v", template, ok, err)
	}

	writeModelPackFile(t, path, "{")
	if _, _, err := readTokenizerChatTemplate(path); err == nil {
		t.Fatal("readTokenizerChatTemplate(invalid JSON) error = nil")
	}
}

func TestInspectModelPack_SafetensorsQwen3Next_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "qwen3_next")

	pack, err := Inspect(dir, mp.WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen3_next" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported qwen3_next", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable {
		t.Fatalf("NativeLoadable=%v, want native/no conversion", pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "qwen" {
		t.Fatalf("chat template = source:%q name:%q, want native qwen", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_Gemma412BUnifiedMetadataOnly_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_unified",
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"audio_token_id": 258881,
		"image_token_id": 258880,
		"video_token_id": 258884,
		"text_config": {
			"model_type": "gemma4_unified_text",
			"vocab_size": 262144,
			"vocab_size_per_layer_input": 262144,
			"hidden_size": 3840,
			"hidden_size_per_layer_input": 0,
			"intermediate_size": 15360,
			"num_hidden_layers": 48,
			"num_attention_heads": 16,
			"num_key_value_heads": 8,
			"num_global_key_value_heads": 1,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 262144,
			"sliding_window": 1024,
			"attention_k_eq_v": true,
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
		},
		"vision_config": {
			"model_type": "gemma4_unified_vision",
			"mm_embed_dim": 3840,
			"num_soft_tokens": 280,
			"output_proj_dims": 3840
		},
		"audio_config": {
			"model_type": "gemma4_unified_audio",
			"hidden_size": 640,
			"audio_embed_dim": 640,
			"audio_samples_per_token": 640,
			"output_proj_dims": 640
		},
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "gemma4_unified" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native Gemma 4 Unified", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ContextLength != 262144 || pack.NumLayers != 48 || pack.HiddenSize != 3840 || pack.VocabSize != 262144 {
		t.Fatalf("metadata = ctx:%d layers:%d hidden:%d vocab:%d, want official 12B Unified shape", pack.ContextLength, pack.NumLayers, pack.HiddenSize, pack.VocabSize)
	}
	if pack.QuantBits != 6 || pack.QuantGroup != 64 {
		t.Fatalf("quant = bits:%d group:%d, want q6 group 64", pack.QuantBits, pack.QuantGroup)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "gemma4" {
		t.Fatalf("chat template = source:%q name:%q, want native gemma4", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsQwen25Native_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Qwen2.5ForCausalLM"],
		"model_type": "qwen2.5",
		"vocab_size": 152064,
		"hidden_size": 3584,
		"num_hidden_layers": 28,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen2" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native qwen2", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplate != "qwen" {
		t.Fatalf("ChatTemplate = %q, want qwen", pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsMistralNative_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["MistralForCausalLM"],
		"model_type": "mistral",
		"vocab_size": 32000,
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"max_position_embeddings": 32768,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "mistral" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native mistral with no Python fallback", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "mistral" {
		t.Fatalf("chat template = source:%q name:%q, want native mistral", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsHermesNative_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["HermesForCausalLM"],
		"model_type": "hermes",
		"vocab_size": 32000,
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"max_position_embeddings": 32768,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "hermes" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native hermes with no Python fallback", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "hermes" {
		t.Fatalf("chat template = source:%q name:%q, want native hermes", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsGraniteNative_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["GraniteForCausalLM"],
		"model_type": "granite",
		"vocab_size": 32000,
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"max_position_embeddings": 32768,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "granite" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native granite with no Python fallback", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "granite" {
		t.Fatalf("chat template = source:%q name:%q, want native granite", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsPhiNative_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Phi3ForCausalLM"],
		"model_type": "phi3",
		"vocab_size": 32064,
		"hidden_size": 3072,
		"num_hidden_layers": 32,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "phi" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native phi with no Python fallback", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "phi" {
		t.Fatalf("chat template = source:%q name:%q, want native phi", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_SafetensorsGLMNative_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["GlmForCausalLM"],
		"model_type": "glm",
		"vocab_size": 151552,
		"hidden_size": 4096,
		"num_hidden_layers": 40,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackQuantization(6))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "glm" || !pack.SupportedArchitecture || !pack.NativeLoadable {
		t.Fatalf("architecture/native = %q/%v/%v, want native glm with no Python fallback", pack.Architecture, pack.SupportedArchitecture, pack.NativeLoadable)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateNative || pack.ChatTemplate != "glm" {
		t.Fatalf("chat template = source:%q name:%q, want native glm", pack.ChatTemplateSource, pack.ChatTemplate)
	}
}

func TestInspectModelPack_Qwen36HybridMetadataOnly_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"language_model_only": false,
		"text_config": {
			"model_type": "qwen3_5_text",
			"vocab_size": 248320,
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"num_hidden_layers": 64,
			"num_attention_heads": 24,
			"num_key_value_heads": 4,
			"head_dim": 256,
			"max_position_embeddings": 262144,
			"layer_types": ["linear_attention", "full_attention"],
			"partial_rotary_factor": 0.25
		},
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen3_6" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported qwen3_6", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("runtime = native:%v issues:%+v, want staged native Qwen3.6", pack.NativeLoadable, pack.Issues)
	}
	if pack.ContextLength != 262144 || pack.NumLayers != 64 || pack.HiddenSize != 5120 || pack.QuantBits != 4 || pack.QuantGroup != 64 {
		t.Fatalf("metadata = ctx:%d layers:%d hidden:%d quant:%d group:%d", pack.ContextLength, pack.NumLayers, pack.HiddenSize, pack.QuantBits, pack.QuantGroup)
	}
	if !pack.HasTokenizer {
		t.Fatalf("HasTokenizer = false, want tokenizer metadata for staged Qwen3.6 loader")
	}
	if pack.ArchitectureProfile == nil || pack.ArchitectureProfile.Generation || pack.ArchitectureProfile.Chat {
		t.Fatalf("profile = %+v, want staged Qwen3.6 loader without standalone generation/chat", pack.ArchitectureProfile)
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

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "qwen3_moe" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported qwen3_moe", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("native/runtime = loadable:%v issues:%+v, want staged native MoE", pack.NativeLoadable, pack.Issues)
	}
	if pack.ArchitectureProfile == nil || pack.ArchitectureProfile.Generation || pack.ArchitectureProfile.Chat {
		t.Fatalf("profile = %+v, want staged Qwen3 MoE loader without standalone generation/chat", pack.ArchitectureProfile)
	}
}

func TestInspectModelPack_MiniMaxJANGTQPack_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["MiniMaxM2ForCausalLM"],
		"model_type": "minimax_m2",
		"vocab_size": 200064,
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"max_position_embeddings": 196608,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "jang_config.json"), `{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"source_model": {"name": "MiniMax-M2.7", "org": "MiniMaxAI", "architecture": "minimax_m2"},
		"mxtq_bits": {"attention": 8, "shared_expert": 8, "routed_expert": 2, "embed_tokens": 8, "lm_head": 8},
		"quantization": {"method": "affine+mxtq", "group_size": 64, "bits_default": 2},
		"capabilities": {"reasoning_parser": "qwen3", "tool_parser": "minimax", "supports_tools": true, "supports_thinking": true}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "chat_template.jinja"), "{{ messages }}")
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00061.safetensors"), "stub")
	writeModelPackFile(t, core.PathJoin(dir, "jangtq_runtime.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "minimax_m2" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported minimax_m2", pack.Architecture, pack.SupportedArchitecture)
	}
	if !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("runtime gate = native:%v issues:%+v, want staged native MiniMax M2 loader", pack.NativeLoadable, pack.Issues)
	}
	if pack.ChatTemplateSource != mp.ModelPackChatTemplateJinja || !pack.HasChatTemplate {
		t.Fatalf("chat template = source:%q has:%v, want chat_template.jinja", pack.ChatTemplateSource, pack.HasChatTemplate)
	}
	if pack.QuantBits != 2 || pack.QuantGroup != 64 || pack.QuantType != "jangtq" || pack.QuantFamily != "jang" {
		t.Fatalf("quant metadata = bits:%d group:%d type:%q family:%q", pack.QuantBits, pack.QuantGroup, pack.QuantType, pack.QuantFamily)
	}
	if pack.JANG == nil || pack.JANG.Profile != "JANGTQ" || pack.JANG.RoutedExpertBits != 2 || !pack.JANG.Capabilities.SupportsThinking {
		t.Fatalf("JANG metadata = %+v, want JANGTQ routed expert metadata", pack.JANG)
	}
	if pack.PackedQuantization == nil || pack.PackedQuantization.Format != "mxtq" || pack.PackedQuantization.RoleBits[string(jang.TensorRoleRoutedExpert)] != 2 {
		t.Fatalf("packed quantization = %+v, want MXTQ routed expert profile", pack.PackedQuantization)
	}
	mmPlan, _ := pack.MiniMaxM2.(*m2.TensorPlan)
	if mmPlan == nil || mmPlan.Config.NumLocalExperts != 256 || mmPlan.Config.NumExpertsPerToken != 8 {
		t.Fatalf("MiniMaxM2 plan = %+v, want expert routing config", mmPlan)
	}
	specs, err := mmPlan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("MiniMaxM2.LayerTensorSpecs() error = %v", err)
	}
	if expert := findMiniMaxM2Spec(specs, m2.TensorRoleExpertDown); expert.Packed == nil || expert.Packed.Bits != 2 {
		t.Fatalf("MiniMaxM2 expert descriptor = %+v, want 2-bit packed expert", expert)
	}
}

func TestInspectModelPack_CodebookVQPackFailsClearly_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 32000,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "codebook_config.json"), `{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 4,
		"code_dim": 2,
		"index_bits": 8,
		"tensors": [
			{"name": "model.layers.0.mlp.down_proj.weight", "shape": [2, 4]}
		]
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if pack.Codebook == nil || pack.Codebook.Format != codebook.FormatVQ || len(pack.Codebook.Tensors) != 1 {
		t.Fatalf("codebook profile = %+v, want VQ model-pack feature flag", pack.Codebook)
	}
	if pack.NativeLoadable || pack.Valid() || !pack.HasIssue(mp.ModelPackIssueUnsupportedCodebook) {
		t.Fatalf("pack loadability = native:%v valid:%v issues:%+v, want clear unsupported codebook issue", pack.NativeLoadable, pack.Valid(), pack.Issues)
	}
}

func TestInspectModelPack_AutoRoundNativePackFailsClearly_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 32000,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "quantization_config.json"), `{
		"bits": 4,
		"group_size": 128,
		"sym": true,
		"data_type": "int",
		"iters": 200,
		"nsamples": 128,
		"seqlen": 2048,
		"quant_method": "auto-round",
		"packing_format": "auto_round"
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if pack.AutoRound == nil || pack.AutoRound.Scheme != autoround.SchemeW4A16 || pack.AutoRound.Iters != 200 {
		t.Fatalf("AutoRound metadata = %+v, want W4A16 sidecar", pack.AutoRound)
	}
	if pack.QuantBits != 4 || pack.QuantGroup != 128 || pack.QuantType != "W4A16" || pack.QuantFamily != autoround.QuantFamilyAutoRound {
		t.Fatalf("quant metadata = bits:%d group:%d type:%q family:%q", pack.QuantBits, pack.QuantGroup, pack.QuantType, pack.QuantFamily)
	}
	if pack.Valid() || pack.NativeLoadable || !pack.HasIssue(mp.ModelPackIssueUnsupportedAutoRound) {
		t.Fatalf("pack validity native=%v valid=%v issues=%+v, want unsupported AutoRound native loader issue", pack.NativeLoadable, pack.Valid(), pack.Issues)
	}
	if !modelPackHasCapability(pack, inference.CapabilityQuantization) {
		t.Fatalf("capabilities = %+v, want quantization capability", pack.Capabilities)
	}
}

func TestInspectModelPack_AutoRoundNativeTensorMapMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 32000,
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "auto_round_config.json"), `{
		"bits": 4,
		"group_size": 32,
		"sym": true,
		"data_type": "int",
		"iters": 200,
		"nsamples": 128,
		"seqlen": 2048,
		"quant_method": "auto-round",
		"packing_format": "auto_round",
		"tensors": [
			{
				"name": "model.layers.0.self_attn.q_proj.weight",
				"packed": "model.layers.0.self_attn.q_proj.weight.packed",
				"scales": "model.layers.0.self_attn.q_proj.weight.scales",
				"zero_points": "model.layers.0.self_attn.q_proj.weight.zeros",
				"bias": "model.layers.0.self_attn.q_proj.bias",
				"shape": [4, 8],
				"bits": 4,
				"group_size": 32,
				"sym": true
			}
		]
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeMiniMaxM2RawSafetensors(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), []miniMaxM2RawSafetensor{
		{Name: "model.layers.0.self_attn.q_proj.weight.packed", DType: "U8", Shape: []int{16}, Raw: make([]byte, 16)},
		miniMaxM2F32RawTensor("model.layers.0.self_attn.q_proj.weight.scales", []float32{1}, 1),
		miniMaxM2F32RawTensor("model.layers.0.self_attn.q_proj.weight.zeros", []float32{0}, 1),
		miniMaxM2F32RawTensor("model.layers.0.self_attn.q_proj.bias", []float32{0, 0, 0, 0}, 4),
	})

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if pack.AutoRound == nil || pack.AutoRound.TensorCount != 1 || !pack.AutoRound.NativeTensorMap() {
		t.Fatalf("AutoRound metadata = %+v, want one validated native tensor map", pack.AutoRound)
	}
	if pack.QuantBits != 4 || pack.QuantGroup != 32 || pack.QuantType != "W4A16" {
		t.Fatalf("quant metadata = bits:%d group:%d type:%q", pack.QuantBits, pack.QuantGroup, pack.QuantType)
	}
	if !pack.Valid() || !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedAutoRound) {
		t.Fatalf("pack validity native=%v valid=%v issues=%+v, want validated native AutoRound tensor map", pack.NativeLoadable, pack.Valid(), pack.Issues)
	}
}

func TestInspectModelPack_AutoRoundNativeExportedPackIsStagedLoadable_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 32000,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	projection := autoround.PackedProjection{
		Tensor: autoround.PackTensor{
			Name:        "model.layers.0.self_attn.q_proj.weight",
			Packed:      "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:      "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints:  "model.layers.0.self_attn.q_proj.weight.zeros",
			Shape:       []int32{1, 4},
			Bits:        2,
			GroupSize:   32,
			Symmetric:   true,
			PackedBytes: 1,
			Groups:      1,
			QMin:        -2,
			QMax:        1,
		},
		Weights: autoround.PackedWeights{
			Scheme:     autoround.SchemeW2A16,
			Format:     autoround.FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     []byte{0b11100100},
			Scales:     []float32{0.5},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
	}
	_, err := autoround.WriteNativePack(context.Background(), dir, autoround.PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Scheme:        autoround.SchemeW2A16,
		ExportFormat:  autoround.FormatAutoRound,
		Iters:         1000,
		NSamples:      512,
		SeqLen:        2048,
	}, []autoround.PackedProjection{projection})
	if err != nil {
		t.Fatalf("WriteNativePack() error = %v", err)
	}

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() || !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedAutoRound) {
		t.Fatalf("pack validity native=%v valid=%v issues=%+v, want staged native AutoRound pack", pack.NativeLoadable, pack.Valid(), pack.Issues)
	}
	if pack.AutoRound == nil || pack.AutoRound.TensorCount != 1 || !pack.AutoRound.NativeTensorMap() || pack.QuantType != string(autoround.SchemeW2A16) {
		t.Fatalf("AutoRound metadata = %+v quant=%q, want exported W2 native tensor map", pack.AutoRound, pack.QuantType)
	}
	if !modelPackHasCapability(pack, inference.CapabilityQuantization) {
		t.Fatalf("capabilities = %+v, want quantization capability", pack.Capabilities)
	}
}

func TestInspectModelPack_AutoRoundGGUFExportMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"vocab_size": 32000,
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "auto_round_config.json"), `{
		"bits": 4,
		"group_size": 256,
		"sym": true,
		"quant_method": "autoround",
		"packing_format": "gguf:q4_k_m"
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "qwen3"},
			{Key: "qwen3.context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(2048)},
		},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}}},
	)

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.AutoRound == nil || pack.AutoRound.Scheme != autoround.SchemeGGUFQ4KM || pack.QuantFamily != autoround.QuantFamilyAutoRound {
		t.Fatalf("AutoRound metadata = %+v quant family=%q, want GGUF export metadata", pack.AutoRound, pack.QuantFamily)
	}
}

func TestInspectModelPack_MiniMaxLayerSkeletonFromSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["MiniMaxM2ForCausalLM"],
		"model_type": "minimax_m2",
		"vocab_size": 32000,
		"hidden_size": 4,
		"intermediate_size": 4,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"max_position_embeddings": 2048,
		"num_local_experts": 3,
		"num_experts_per_tok": 2,
		"use_routing_bias": true
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "jang_config.json"), `{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"mxtq_bits": {"attention": 8, "routed_expert": 2},
		"quantization": {"method": "affine+mxtq", "group_size": 4, "bits_default": 2}
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "chat_template.jinja"), "{{ messages }}")

	cfg := m2.Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   4,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
		UseRoutingBias:     true,
	}
	plan, err := m2.BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
	}
	writeMiniMaxM2RawSafetensors(t, core.PathJoin(dir, "model.safetensors"), miniMaxM2SkeletonRawTensors(t, plan, false))

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	skel, _ := pack.MiniMaxM2LayerSkeleton.(*m2.LayerForwardSkeleton)
	if skel == nil {
		t.Fatalf("MiniMaxM2LayerSkeleton = nil, want safetensors-backed skeleton")
	}
	if len(skel.Attention) != 4 || skel.EstimatedBytes() != 108 {
		t.Fatalf("skeleton = %+v bytes=%d, want four attention tensors and 108 estimated bytes", skel, skel.EstimatedBytes())
	}
}

func TestInspectModelPack_MetadataOnlyArchitectureProfiles_Good(t *testing.T) {
	cases := []struct {
		name                 string
		config               string
		wantArchitecture     string
		wantParser           string
		wantMoE              bool
		wantEmbeddings       bool
		wantNative           bool
		wantChatTemplate     bool
		wantChatTemplateName string
	}{
		{
			name: "mixtral",
			config: `{
				"architectures": ["MixtralForCausalLM"],
				"vocab_size": 32000,
				"hidden_size": 4096,
				"num_hidden_layers": 32,
				"max_position_embeddings": 32768,
				"num_local_experts": 8,
				"num_experts_per_tok": 2
			}`,
			wantArchitecture: "mixtral",
			wantParser:       "mistral",
			wantMoE:          true,
			wantNative:       true,
		},
		{
			name: "bert",
			config: `{
				"architectures": ["BertModel"],
				"vocab_size": 30522,
				"hidden_size": 768,
				"num_hidden_layers": 12,
				"max_position_embeddings": 512
			}`,
			wantArchitecture: "bert",
			wantParser:       "generic",
			wantEmbeddings:   true,
			wantNative:       true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeModelPackFile(t, core.PathJoin(dir, "config.json"), tc.config)
			writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
			writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

			pack, err := Inspect(dir)
			if err != nil {
				t.Fatalf("Inspect() error = %v", err)
			}
			if !pack.Valid() {
				t.Fatalf("pack should be metadata-valid, issues = %+v", pack.Issues)
			}
			if pack.Architecture != tc.wantArchitecture || !pack.SupportedArchitecture {
				t.Fatalf("architecture = %q supported=%v, want %q supported", pack.Architecture, pack.SupportedArchitecture, tc.wantArchitecture)
			}
			if pack.NativeLoadable != tc.wantNative {
				t.Fatalf("runtime = native:%v issues:%+v, want native=%v", pack.NativeLoadable, pack.Issues, tc.wantNative)
			}
			if tc.wantNative && pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
				t.Fatalf("issues = %+v, native staged pack should not carry unsupported runtime", pack.Issues)
			}
			if !tc.wantNative && !pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
				t.Fatalf("issues = %+v, want metadata-only runtime gate", pack.Issues)
			}
			if pack.ArchitectureProfile == nil {
				t.Fatal("ArchitectureProfile = nil, want metadata profile")
			}
			if pack.ArchitectureProfile.ParserID != tc.wantParser || pack.ArchitectureProfile.MoE != tc.wantMoE || pack.ArchitectureProfile.Embeddings != tc.wantEmbeddings {
				t.Fatalf("profile = %+v, want parser/moe/embeddings %q/%v/%v", pack.ArchitectureProfile, tc.wantParser, tc.wantMoE, tc.wantEmbeddings)
			}
			if pack.HasChatTemplate != tc.wantChatTemplate {
				t.Fatalf("HasChatTemplate = %v, want %v", pack.HasChatTemplate, tc.wantChatTemplate)
			}
			if tc.wantChatTemplateName != "" && pack.ChatTemplate != tc.wantChatTemplateName {
				t.Fatalf("ChatTemplate = %q, want %q", pack.ChatTemplate, tc.wantChatTemplateName)
			}
		})
	}
}

func TestInspectModelPack_BertSentenceTransformerEmbeddings_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 384,
		"num_hidden_layers": 6,
		"max_position_embeddings": 512
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "sentence_bert_config.json"), `{"max_seq_length": 256}`)
	writeModelPackFile(t, core.PathJoin(dir, "modules.json"), `[
		{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
		{"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
		{"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
	]`)
	poolingDir := core.PathJoin(dir, "1_Pooling")
	if result := core.MkdirAll(poolingDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll(%s) error = %v", poolingDir, result.Value)
	}
	writeModelPackFile(t, core.PathJoin(poolingDir, "config.json"), `{
		"pooling_mode_cls_token": false,
		"pooling_mode_mean_tokens": true,
		"pooling_mode_max_tokens": false
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be metadata-valid, issues = %+v", pack.Issues)
	}
	if pack.Embedding == nil {
		t.Fatalf("Embedding = nil, want BERT embedding profile")
	}
	if pack.Embedding.Dimension != 384 || pack.Embedding.Pooling != "mean" || !pack.Embedding.Normalize || pack.Embedding.MaxSequenceLength != 256 {
		t.Fatalf("Embedding = %+v, want dim 384 mean pooling normalized max sequence 256", pack.Embedding)
	}
	if !modelPackHasCapability(pack, inference.CapabilityEmbeddings) {
		t.Fatalf("capabilities = %+v, want embeddings capability", pack.Capabilities)
	}
	if !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("runtime = native:%v issues:%+v, want staged native BERT encoder", pack.NativeLoadable, pack.Issues)
	}
}

func TestInspectModelPack_BertCrossEncoderRerank_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be metadata-valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "bert_rerank" || pack.ArchitectureProfile == nil || !pack.ArchitectureProfile.Rerank {
		t.Fatalf("architecture/profile = %q %+v, want bert_rerank profile", pack.Architecture, pack.ArchitectureProfile)
	}
	if pack.Rerank == nil || pack.Rerank.Method != "cross-encoder" || pack.Rerank.MaxSequenceLength != 512 {
		t.Fatalf("Rerank = %+v, want cross-encoder max sequence 512", pack.Rerank)
	}
	if !modelPackHasCapability(pack, inference.CapabilityRerank) {
		t.Fatalf("capabilities = %+v, want rerank capability", pack.Capabilities)
	}
	if !pack.NativeLoadable || pack.HasIssue(mp.ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("runtime = native:%v issues:%+v, want staged native BERT rerank scorer", pack.NativeLoadable, pack.Issues)
	}
}

func modelPackHasCapability(pack mp.ModelPack, id inference.CapabilityID) bool {
	for _, capability := range pack.Capabilities {
		if capability.ID == id {
			return true
		}
	}
	return false
}

func TestValidateModelPack_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma3"}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	pack, err := Validate(dir)
	if err == nil {
		t.Fatal("expected validation error for missing tokenizer")
	}
	if !pack.HasIssue(mp.ModelPackIssueMissingTokenizer) {
		t.Fatalf("issues = %+v, want missing tokenizer", pack.Issues)
	}
}

func TestValidateModelPack_QuantizationAndContext_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	pack, err := Validate(dir, mp.WithPackQuantization(8), mp.WithPackMaxContextLength(8192))
	if err == nil {
		t.Fatal("expected validation error for quantization/context mismatch")
	}
	if !pack.HasIssue(mp.ModelPackIssueQuantizationMismatch) || !pack.HasIssue(mp.ModelPackIssueContextTooLarge) {
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
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{127, 128}}},
	)

	pack, err := Validate(dir)
	if err == nil {
		t.Fatal("expected validation error for invalid GGUF tensor metadata")
	}
	if !pack.HasIssue(mp.ModelPackIssueInvalidGGUF) {
		t.Fatalf("issues = %+v, want invalid GGUF", pack.Issues)
	}
}

// TestValidateModelPack_Good validates a sound, native-loadable pack and gets it
// back error-free — the success path Validate is the strict wrapper for. A valid
// safetensors gemma-4 pack with matching quant/context opts must return no error
// and a pack that reports Valid(). The Bad/Ugly siblings above only exercise the
// reject branches; this pins the accept branch.
func TestValidateModelPack_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	pack, err := Validate(dir, mp.WithPackQuantization(4), mp.WithPackMaxContextLength(131072))
	if err != nil {
		t.Fatalf("Validate() error = %v, issues = %+v", err, pack.Issues)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if !pack.NativeLoadable || !pack.SupportedArchitecture {
		t.Fatalf("native=%v supported=%v, want a native-loadable supported pack", pack.NativeLoadable, pack.SupportedArchitecture)
	}
}

// TestSupportsArchitecture_Good reports true for architectures that carry a
// registered profile in dappco.re/go/mlx/profile — the loader-facing predicate
// every candidate model is screened by. The names are real registered profiles
// (gemma-4 text, qwen3 dense + MoE, llama), and the lookup is case-insensitive,
// so an upper-cased alias resolves the same as its canonical form.
func TestSupportsArchitecture_Good(t *testing.T) {
	for _, arch := range []string{"gemma4", "gemma4_text", "qwen3", "qwen3_moe", "llama", "QWEN3"} {
		if !SupportsArchitecture(arch) {
			t.Errorf("SupportsArchitecture(%q) = false, want true for a registered profile", arch)
		}
	}
}

// TestSupportsArchitecture_Bad reports false for architectures with no registered
// profile — an unknown name and the empty string. The predicate must not claim
// support it can't back with a profile.
func TestSupportsArchitecture_Bad(t *testing.T) {
	for _, arch := range []string{"totally_unknown_arch_xyz", "", "gpt2", "not-an-arch"} {
		if SupportsArchitecture(arch) {
			t.Errorf("SupportsArchitecture(%q) = true, want false for an unregistered profile", arch)
		}
	}
}

// --- autoRoundConfigIssuePath: the path an AutoRound parse-error issue points
// at. Two recognised config filenames, the auto_round one taking priority when
// both are listed. Driven directly against a synthetic dir index so both
// branches plus the nil-index fall-through are pinned without an Inspect run.

// TestAutoRoundConfigIssuePath_Good points at auto_round_config.json when the
// directory listing recorded it — the preferred filename when present.
func TestAutoRoundConfigIssuePath_Good(t *testing.T) {
	dir := &modelPackDirIndex{populated: true, autoRoundConfig: true}
	got := autoRoundConfigIssuePath("/models/pack", dir)
	want := core.PathJoin("/models/pack", autoround.PackConfigFileAutoRound)
	if got != want {
		t.Errorf("autoRoundConfigIssuePath: got %q want %q", got, want)
	}
}

// TestAutoRoundConfigIssuePath_Ugly falls back to quantization_config.json when
// only that filename was listed — the AutoRound metadata rode in on the generic
// quantization config rather than its own file.
func TestAutoRoundConfigIssuePath_Ugly(t *testing.T) {
	dir := &modelPackDirIndex{populated: true, quantConfig: true}
	got := autoRoundConfigIssuePath("/models/pack", dir)
	want := core.PathJoin("/models/pack", autoround.PackConfigFileQuantization)
	if got != want {
		t.Errorf("autoRoundConfigIssuePath: got %q want %q", got, want)
	}
}

// TestAutoRoundConfigIssuePath_Bad handles a nil index — the listing was never
// gathered (single-file entry path). has() returns true for a nil index so the
// caller would probe normally; the issue path defaults to the quantization
// filename rather than panicking on the nil deref.
func TestAutoRoundConfigIssuePath_Bad(t *testing.T) {
	got := autoRoundConfigIssuePath("/models/pack", nil)
	want := core.PathJoin("/models/pack", autoround.PackConfigFileQuantization)
	if got != want {
		t.Errorf("autoRoundConfigIssuePath(nil): got %q want %q", got, want)
	}
}

// --- modelPackUnsupportedRuntimeMessageFor: the warning text attached when a
// recognised architecture has no native runtime yet. The message specialises on
// the profile's shape (attached drafter, hybrid linear-attention, sparse expert,
// embeddings, rerank); a nil profile or an unspecialised one gets the generic
// line. Looked up by id from the live registry so the branch under test is the
// one the real Inspect path would hit.

// TestModelPackUnsupportedRuntimeMessageFor_Good takes the generic branch for a
// recognised generative architecture that carries none of the special shapes —
// the plain "native runtime loading is not implemented yet" line.
func TestModelPackUnsupportedRuntimeMessageFor_Good(t *testing.T) {
	prof, ok := profile.LookupArchitectureProfileRef("llama")
	if !ok {
		t.Skip("llama profile not registered")
	}
	got := modelPackUnsupportedRuntimeMessageFor(prof, "llama")
	want := "architecture is recognized, but native runtime loading is not implemented yet: llama"
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

// TestModelPackUnsupportedRuntimeMessageFor_Ugly walks the specialised branches
// — each profile shape produces its own tailored guidance, so a caller reading
// the warning knows why the pack can't load natively and what to do instead.
// The id-keyed cases resolve from the live registry; the flag-keyed cases use
// synthetic profile literals so the embeddings/rerank/MoE arms are pinned
// independent of which specific architectures happen to carry those flags
// without a native runtime.
func TestModelPackUnsupportedRuntimeMessageFor_Ugly(t *testing.T) {
	idCases := []struct {
		id       string
		contains string
	}{
		{"gemma4_assistant", "attached MTP drafter"},
		{"qwen3_6", "hybrid linear-attention"},
		{"qwen3_6_moe", "sparse expert"},
	}
	for _, tc := range idCases {
		t.Run(tc.id, func(t *testing.T) {
			prof, ok := profile.LookupArchitectureProfileRef(tc.id)
			if !ok {
				t.Skipf("%s profile not registered", tc.id)
			}
			got := modelPackUnsupportedRuntimeMessageFor(prof, tc.id)
			if !core.Contains(got, tc.contains) {
				t.Errorf("message for %s = %q, want it to contain %q", tc.id, got, tc.contains)
			}
			if !core.HasSuffix(got, tc.id) {
				t.Errorf("message for %s = %q, want it to end with the architecture name", tc.id, got)
			}
		})
	}

	flagCases := []struct {
		name     string
		prof     profile.ModelArchitectureProfile
		contains string
	}{
		{"embeddings", profile.ModelArchitectureProfile{ID: "some_encoder", Embeddings: true}, "embedding encoder"},
		{"rerank", profile.ModelArchitectureProfile{ID: "some_reranker", Rerank: true}, "rerank scorer"},
		{"moe", profile.ModelArchitectureProfile{ID: "some_moe", MoE: true}, "sparse expert runtime"},
	}
	for _, tc := range flagCases {
		t.Run(tc.name, func(t *testing.T) {
			prof := tc.prof
			got := modelPackUnsupportedRuntimeMessageFor(&prof, "arch_"+tc.name)
			if !core.Contains(got, tc.contains) {
				t.Errorf("message for %s = %q, want it to contain %q", tc.name, got, tc.contains)
			}
		})
	}
}

// TestModelPackUnsupportedRuntimeMessageFor_Bad takes the nil-profile branch:
// a nil profile (architecture has no profile entry) yields the generic line as
// the safe default rather than a nil deref on profile.ID.
func TestModelPackUnsupportedRuntimeMessageFor_Bad(t *testing.T) {
	got := modelPackUnsupportedRuntimeMessageFor(nil, "mystery_arch")
	want := "architecture is recognized, but native runtime loading is not implemented yet: mystery_arch"
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

// --- modelPackAlgorithmCapability: stamps the model architecture onto a shared
// algorithm capability. Registered algorithm ids (embeddings, quantization, …)
// resolve to the profile's capability; an id with no algorithm profile falls
// back to a planned capability. Both paths attach the architecture label, and
// an empty architecture must not write an empty label.

// TestModelPackAlgorithmCapability_Good resolves a registered algorithm id and
// stamps the architecture label onto the profile-supplied capability.
func TestModelPackAlgorithmCapability_Good(t *testing.T) {
	cap := modelPackAlgorithmCapability(inference.CapabilityEmbeddings, "bert")
	if cap.ID != inference.CapabilityEmbeddings {
		t.Errorf("ID: got %q want %q", cap.ID, inference.CapabilityEmbeddings)
	}
	if cap.Labels["architecture"] != "bert" {
		t.Errorf("architecture label: got %q want %q", cap.Labels["architecture"], "bert")
	}
}

// TestModelPackAlgorithmCapability_Ugly drives the planned-capability fallback:
// CapabilityGenerate has no registered algorithm profile, so the helper mints a
// planned capability and still attaches the architecture label.
func TestModelPackAlgorithmCapability_Ugly(t *testing.T) {
	cap := modelPackAlgorithmCapability(inference.CapabilityGenerate, "qwen3")
	if cap.ID != inference.CapabilityGenerate {
		t.Errorf("ID: got %q want %q", cap.ID, inference.CapabilityGenerate)
	}
	if cap.Status != inference.CapabilityStatusPlanned {
		t.Errorf("Status: got %q want %q", cap.Status, inference.CapabilityStatusPlanned)
	}
	if cap.Labels["architecture"] != "qwen3" {
		t.Errorf("architecture label: got %q want %q", cap.Labels["architecture"], "qwen3")
	}
}

// TestModelPackAlgorithmCapability_Bad passes an empty architecture on both the
// registered-profile and planned-fallback arms — the helper must not write an
// "architecture" label key with an empty value (it would pollute the capability
// metadata with a meaningless entry). CapabilityEmbeddings resolves a profile;
// CapabilityGenerate takes the planned fallback. Neither must stamp a label.
func TestModelPackAlgorithmCapability_Bad(t *testing.T) {
	for _, id := range []inference.CapabilityID{inference.CapabilityEmbeddings, inference.CapabilityGenerate} {
		cap := modelPackAlgorithmCapability(id, "")
		if _, present := cap.Labels["architecture"]; present {
			t.Errorf("%s: architecture label present for empty architecture: %v", id, cap.Labels)
		}
	}
}

// --- readPoolingConfig: maps a sentence-transformers Pooling config.json mode
// flag to the canonical pooling name. Driven against synthetic config bodies in
// a tempdir so each mode branch and the unreadable/empty fall-throughs are
// exercised without a full embedding-model Inspect.

// TestReadPoolingConfig_Good walks each recognised mode flag → canonical name.
// mean wins over cls when both are set (the precedence the switch encodes), so
// the cases set exactly one flag each to pin the branch under test.
func TestReadPoolingConfig_Good(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"mean", `{"pooling_mode_mean_tokens":true}`, "mean"},
		{"cls", `{"pooling_mode_cls_token":true}`, "cls"},
		{"max", `{"pooling_mode_max_tokens":true}`, "max"},
		{"weighted_mean", `{"pooling_mode_weightedmean_tokens":true}`, "weighted_mean"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), "config.json")
			writeModelPackFile(t, path, tc.body)
			got, ok := readPoolingConfig(path)
			if !ok || got != tc.want {
				t.Errorf("readPoolingConfig(%s) = (%q,%v), want (%q,true)", tc.name, got, ok, tc.want)
			}
		})
	}
}

// TestReadPoolingConfig_Ugly returns ok=false when the config parses cleanly but
// declares no recognised mode — the caller then falls through to the next
// pooling candidate rather than adopting a bogus default.
func TestReadPoolingConfig_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "config.json")
	writeModelPackFile(t, path, `{"word_embedding_dimension":768}`)
	if got, ok := readPoolingConfig(path); ok {
		t.Errorf("readPoolingConfig with no mode flag = (%q,true), want ok=false", got)
	}
}

// TestReadPoolingConfig_Bad returns ok=false on two distinct failure modes that
// share one fall-through contract for the caller: a path that does not exist
// (unreadable branch) and a present-but-malformed body (parse-failure branch).
func TestReadPoolingConfig_Bad(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "missing", "config.json")
	if got, ok := readPoolingConfig(missing); ok {
		t.Errorf("readPoolingConfig(missing) = (%q,true), want ok=false", got)
	}

	malformed := core.PathJoin(t.TempDir(), "config.json")
	writeModelPackFile(t, malformed, `{"pooling_mode_mean_tokens": tru`) // truncated JSON
	if got, ok := readPoolingConfig(malformed); ok {
		t.Errorf("readPoolingConfig(malformed) = (%q,true), want ok=false", got)
	}
}

// --- readSentenceBertMaxSequence: reads max_seq_length from
// sentence_bert_config.json when the listing recorded it. Synthetic tempdir +
// dir index exercise the present/absent/unreadable boundaries.

// TestReadSentenceBertMaxSequence_Good reads a positive max_seq_length from a
// recorded sentence_bert_config.json.
func TestReadSentenceBertMaxSequence_Good(t *testing.T) {
	root := t.TempDir()
	writeModelPackFile(t, core.PathJoin(root, "sentence_bert_config.json"), `{"max_seq_length":512}`)
	dir := &modelPackDirIndex{populated: true, sentenceBert: true}
	got, ok := readSentenceBertMaxSequence(root, dir)
	if !ok || got != 512 {
		t.Errorf("readSentenceBertMaxSequence = (%d,%v), want (512,true)", got, ok)
	}
}

// TestReadSentenceBertMaxSequence_Bad returns ok=false on two failure modes: the
// listing recorded the file but it is not actually readable (unreadable branch),
// and a present-but-malformed body (parse-failure branch). Both fall through to
// the config-derived default rather than surfacing a bogus sequence length.
func TestReadSentenceBertMaxSequence_Bad(t *testing.T) {
	unreadable := t.TempDir() // recorded present, but no file written
	dir := &modelPackDirIndex{populated: true, sentenceBert: true}
	if got, ok := readSentenceBertMaxSequence(unreadable, dir); ok {
		t.Errorf("readSentenceBertMaxSequence(unreadable) = (%d,true), want ok=false", got)
	}

	malformed := t.TempDir()
	writeModelPackFile(t, core.PathJoin(malformed, "sentence_bert_config.json"), `{"max_seq_length": 51`) // truncated
	if got, ok := readSentenceBertMaxSequence(malformed, dir); ok {
		t.Errorf("readSentenceBertMaxSequence(malformed) = (%d,true), want ok=false", got)
	}
}

// --- readSentenceTransformerPooling: resolves the pooling mode either by the
// fast-path recorded "*_Pooling" subdir or the glob fallback. The white-box
// tests below pin the glob-fallback branch the fast-path tests skip.

// TestReadSentenceTransformerPooling_Ugly takes the glob fallback: the dir index
// recorded no poolingDir (single-file entry path), so the helper globs
// "*_Pooling/config.json" and resolves the mode from the matched file.
func TestReadSentenceTransformerPooling_Ugly(t *testing.T) {
	root := t.TempDir()
	poolDir := core.PathJoin(root, "1_Pooling")
	if r := core.MkdirAll(poolDir, 0o755); !r.OK {
		t.Fatalf("mkdir pooling: %v", r.Value)
	}
	writeModelPackFile(t, core.PathJoin(poolDir, "config.json"), `{"pooling_mode_mean_tokens":true}`)
	dir := &modelPackDirIndex{populated: true} // poolingDir empty → glob path
	got, ok := readSentenceTransformerPooling(root, dir)
	if !ok || got != "mean" {
		t.Errorf("readSentenceTransformerPooling (glob) = (%q,%v), want (mean,true)", got, ok)
	}
}

// TestReadSentenceTransformerPooling_Bad returns ok=false when neither the
// recorded subdir nor the glob finds a parseable Pooling config — the embedding
// inspector then keeps its transformers-default "cls" pooling.
func TestReadSentenceTransformerPooling_Bad(t *testing.T) {
	root := t.TempDir() // no *_Pooling dir at all
	dir := &modelPackDirIndex{populated: true}
	if got, ok := readSentenceTransformerPooling(root, dir); ok {
		t.Errorf("readSentenceTransformerPooling (none) = (%q,true), want ok=false", got)
	}
}
