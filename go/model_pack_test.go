// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
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

func TestInspectModelPack_WeightAndConfigEdgeCases_Bad(t *testing.T) {
	t.Run("mixed_weights", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
		writeModelPackFile(t, core.PathJoin(dir, "model.gguf"), "stub")

		pack, err := InspectModelPack(dir, WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("InspectModelPack() error = %v", err)
		}
		if pack.Format != ModelPackFormatMixed || !pack.HasIssue(ModelPackIssueMixedWeightFormats) {
			t.Fatalf("pack = %+v, want mixed weight issue", pack)
		}
	})

	t.Run("multiple_gguf", func(t *testing.T) {
		dir := t.TempDir()
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
		writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(dir, "a.gguf"), "stub")
		writeModelPackFile(t, core.PathJoin(dir, "b.gguf"), "stub")

		pack, err := InspectModelPack(dir, WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("InspectModelPack() error = %v", err)
		}
		if pack.Format != ModelPackFormatGGUF || !pack.HasIssue(ModelPackIssueMultipleGGUF) {
			t.Fatalf("pack = %+v, want multiple GGUF issue", pack)
		}
	})

	t.Run("missing_and_invalid_config", func(t *testing.T) {
		missing := t.TempDir()
		writeModelPackFile(t, core.PathJoin(missing, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(missing, "model.safetensors"), "stub")
		pack, err := InspectModelPack(missing, WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("InspectModelPack(missing config) error = %v", err)
		}
		if !pack.HasIssue(ModelPackIssueMissingConfig) || !pack.HasIssue(ModelPackIssueMissingArchitecture) {
			t.Fatalf("issues = %+v, want missing config and architecture", pack.Issues)
		}

		invalid := t.TempDir()
		writeModelPackFile(t, core.PathJoin(invalid, "config.json"), "{")
		writeModelPackFile(t, core.PathJoin(invalid, "tokenizer.json"), modelPackTokenizerJSON)
		writeModelPackFile(t, core.PathJoin(invalid, "model.safetensors"), "stub")
		pack, err = InspectModelPack(invalid, WithPackRequireChatTemplate(false))
		if err != nil {
			t.Fatalf("InspectModelPack(invalid config) error = %v", err)
		}
		if !pack.HasIssue(ModelPackIssueInvalidConfig) {
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

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.Architecture != "minimax_m2" || !pack.SupportedArchitecture {
		t.Fatalf("architecture = %q supported=%v, want supported minimax_m2", pack.Architecture, pack.SupportedArchitecture)
	}
	if pack.NativeLoadable || !pack.HasIssue(ModelPackIssueUnsupportedRuntime) {
		t.Fatalf("runtime gate = native:%v issues:%+v, want recognised but kernel-gated", pack.NativeLoadable, pack.Issues)
	}
	if pack.ChatTemplateSource != ModelPackChatTemplateJinja || !pack.HasChatTemplate {
		t.Fatalf("chat template = source:%q has:%v, want chat_template.jinja", pack.ChatTemplateSource, pack.HasChatTemplate)
	}
	if pack.QuantBits != 2 || pack.QuantGroup != 64 || pack.QuantType != "jangtq" || pack.QuantFamily != "jang" {
		t.Fatalf("quant metadata = bits:%d group:%d type:%q family:%q", pack.QuantBits, pack.QuantGroup, pack.QuantType, pack.QuantFamily)
	}
	if pack.JANG == nil || pack.JANG.Profile != "JANGTQ" || pack.JANG.RoutedExpertBits != 2 || !pack.JANG.Capabilities.SupportsThinking {
		t.Fatalf("JANG metadata = %+v, want JANGTQ routed expert metadata", pack.JANG)
	}
	if pack.PackedQuantization == nil || pack.PackedQuantization.Format != "mxtq" || pack.PackedQuantization.RoleBits[string(JANGTensorRoleRoutedExpert)] != 2 {
		t.Fatalf("packed quantization = %+v, want MXTQ routed expert profile", pack.PackedQuantization)
	}
	if pack.MiniMaxM2 == nil || pack.MiniMaxM2.Config.NumLocalExperts != 256 || pack.MiniMaxM2.Config.NumExpertsPerToken != 8 {
		t.Fatalf("MiniMaxM2 plan = %+v, want expert routing config", pack.MiniMaxM2)
	}
	specs, err := pack.MiniMaxM2.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("MiniMaxM2.LayerTensorSpecs() error = %v", err)
	}
	if expert := findMiniMaxM2Spec(specs, MiniMaxM2TensorRoleExpertDown); expert.Packed == nil || expert.Packed.Bits != 2 {
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

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if pack.Codebook == nil || pack.Codebook.Format != CodebookFormatVQ || len(pack.Codebook.Tensors) != 1 {
		t.Fatalf("codebook profile = %+v, want VQ model-pack feature flag", pack.Codebook)
	}
	if pack.NativeLoadable || pack.Valid() || !pack.HasIssue(ModelPackIssueUnsupportedCodebook) {
		t.Fatalf("pack loadability = native:%v valid:%v issues:%+v, want clear unsupported codebook issue", pack.NativeLoadable, pack.Valid(), pack.Issues)
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

	cfg := MiniMaxM2Config{
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
	plan, err := BuildMiniMaxM2TensorPlan(cfg, &JANGQuantizationInfo{
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

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
	}
	if !pack.Valid() {
		t.Fatalf("pack should be valid, issues = %+v", pack.Issues)
	}
	if pack.MiniMaxM2LayerSkeleton == nil {
		t.Fatalf("MiniMaxM2LayerSkeleton = nil, want safetensors-backed skeleton")
	}
	if len(pack.MiniMaxM2LayerSkeleton.Attention) != 4 || pack.MiniMaxM2LayerSkeleton.EstimatedBytes() != 108 {
		t.Fatalf("skeleton = %+v bytes=%d, want four attention tensors and 108 estimated bytes", pack.MiniMaxM2LayerSkeleton, pack.MiniMaxM2LayerSkeleton.EstimatedBytes())
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
			wantArchitecture:     "mixtral",
			wantParser:           "mistral",
			wantMoE:              true,
			wantChatTemplate:     true,
			wantChatTemplateName: "mistral",
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
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeModelPackFile(t, core.PathJoin(dir, "config.json"), tc.config)
			writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
			writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

			pack, err := InspectModelPack(dir)
			if err != nil {
				t.Fatalf("InspectModelPack() error = %v", err)
			}
			if !pack.Valid() {
				t.Fatalf("pack should be metadata-valid, issues = %+v", pack.Issues)
			}
			if pack.Architecture != tc.wantArchitecture || !pack.SupportedArchitecture {
				t.Fatalf("architecture = %q supported=%v, want %q supported", pack.Architecture, pack.SupportedArchitecture, tc.wantArchitecture)
			}
			if pack.NativeLoadable || !pack.HasIssue(ModelPackIssueUnsupportedRuntime) {
				t.Fatalf("runtime = native:%v issues:%+v, want metadata-only runtime gate", pack.NativeLoadable, pack.Issues)
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

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
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

	pack, err := InspectModelPack(dir)
	if err != nil {
		t.Fatalf("InspectModelPack() error = %v", err)
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

func modelPackHasCapability(pack ModelPack, id inference.CapabilityID) bool {
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
