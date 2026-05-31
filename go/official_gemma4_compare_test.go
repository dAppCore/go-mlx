// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestOfficialGemma4E2BControlComparison_Good(t *testing.T) {
	targetLock, targetDir := officialGemma4ComparisonSnapshot(t, OfficialGemma4E2BRoleTarget, 6, 512, "proportional")
	controlDir := officialGemma4ArchivedControlSnapshot(t, 4, 512, "proportional")

	report, err := CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir, targetLock)
	if err != nil {
		t.Fatalf("CompareOfficialGemma4E2BControlSnapshots() error = %v", err)
	}
	if !report.Compatible || len(report.Issues) != 0 {
		t.Fatalf("report compatibility = %v issues=%+v, want clean metadata match", report.Compatible, report.Issues)
	}
	if report.Target.ModelID != "google/gemma-4-E2B-it" || report.Control.ModelID != ProductionLaneArchivedBaselineModelID {
		t.Fatalf("model IDs = target:%q control:%q", report.Target.ModelID, report.Control.ModelID)
	}
	if report.Target.QuantBits != 6 || report.Control.QuantBits != 4 {
		t.Fatalf("quant bits = target:%d control:%d, want q6 official target vs q4 control", report.Target.QuantBits, report.Control.QuantBits)
	}
	if report.Target.ContextLength != ProductionLaneHyperLongContextLength || report.Control.ContextLength != ProductionLaneHyperLongContextLength {
		t.Fatalf("context = target:%d control:%d, want 128Ki match", report.Target.ContextLength, report.Control.ContextLength)
	}
	if report.Target.FullAttentionLayers != 7 || report.Target.SlidingAttentionLayers != 28 || report.Target.FullAttentionInterval != 5 {
		t.Fatalf("target attention = %+v, want 28 sliding + 7 full with every fifth layer full", report.Target)
	}
	if !report.Target.ProportionalRoPE || report.Target.FullRoPETheta != 1000000 || report.Target.FullRoPEType != "proportional" || report.Target.SlidingRoPETheta != 10000 {
		t.Fatalf("target rope = %+v, want Gemma 4 p-RoPE global plus default local RoPE", report.Target)
	}
	if !report.Target.PerLayerInputs || report.Target.HiddenSizePerLayerInput != 256 || report.Target.VocabSizePerLayerInput != 262144 {
		t.Fatalf("target PLE = %+v, want per-layer input fields recorded", report.Target)
	}
	if report.Target.NumKVSharedLayers != 20 {
		t.Fatalf("target shared KV = %d, want 20", report.Target.NumKVSharedLayers)
	}
	if !report.Target.HasThinkingToken || !report.Target.StripsThinking || !report.Target.HasThoughtChannelMarkers {
		t.Fatalf("target template markers = %+v, want thinking/no-thinking contract markers", report.Target)
	}
	if !report.ChatTemplateCompatible || !report.AttentionCompatible || !report.RoPECompatible || !report.SharedKVCompatible || !report.PerLayerInputCompatible {
		t.Fatalf("compatibility flags = %+v", report)
	}
	if !report.RetainedStateCompatible || !report.PromptCacheCompatible {
		t.Fatalf("state/cache compatibility flags = retained:%v prompt:%v, want both compatible", report.RetainedStateCompatible, report.PromptCacheCompatible)
	}
}

func TestOfficialGemma4E2BControlComparison_RejectsControlRoPEDrift_Bad(t *testing.T) {
	targetLock, targetDir := officialGemma4ComparisonSnapshot(t, OfficialGemma4E2BRoleTarget, 6, 512, "proportional")
	controlDir := officialGemma4ArchivedControlSnapshot(t, 4, 512, "default")

	report, err := CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir, targetLock)
	if err == nil {
		t.Fatal("CompareOfficialGemma4E2BControlSnapshots(rope drift) error = nil")
	}
	if report.Compatible || !containsOfficialGemma4ComparisonIssue(report.Issues, "p-RoPE") {
		t.Fatalf("report = %+v, want incompatible p-RoPE drift", report)
	}
}

func TestOfficialGemma4E2BControlComparison_RejectsControlWindowDrift_Ugly(t *testing.T) {
	targetLock, targetDir := officialGemma4ComparisonSnapshot(t, OfficialGemma4E2BRoleTarget, 6, 512, "proportional")
	controlDir := officialGemma4ArchivedControlSnapshot(t, 4, 1024, "proportional")

	report, err := CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir, targetLock)
	if err == nil {
		t.Fatal("CompareOfficialGemma4E2BControlSnapshots(window drift) error = nil")
	}
	if report.Compatible || !containsOfficialGemma4ComparisonIssue(report.Issues, "sliding window") {
		t.Fatalf("report = %+v, want incompatible sliding-window drift", report)
	}
	if report.RetainedStateCompatible || report.PromptCacheCompatible {
		t.Fatalf("state/cache compatibility = retained:%v prompt:%v, want both false after K/V window drift", report.RetainedStateCompatible, report.PromptCacheCompatible)
	}
}

func containsOfficialGemma4ComparisonIssue(issues []string, fragment string) bool {
	for _, issue := range issues {
		if core.Contains(issue, fragment) {
			return true
		}
	}
	return false
}

func officialGemma4ArchivedControlSnapshot(t *testing.T, quantBits, slidingWindow int, fullRoPEType string) string {
	t.Helper()
	lock, dir := officialGemma4ComparisonSnapshot(t, "control", quantBits, slidingWindow, fullRoPEType)
	_ = lock
	return dir
}

func officialGemma4ComparisonSnapshot(t *testing.T, role string, quantBits, slidingWindow int, fullRoPEType string) (OfficialGemma4E2BLock, string) {
	t.Helper()
	modelID := "google/gemma-4-E2B-it"
	revision := "test-official-compare"
	if role == "control" {
		modelID = ProductionLaneArchivedBaselineModelID
		revision = "test-archived-q4-control"
	}
	config := []byte(core.Sprintf(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"vocab_size_per_layer_input": 262144,
			"hidden_size": 1536,
			"hidden_size_per_layer_input": 256,
			"num_hidden_layers": 35,
			"num_attention_heads": 8,
			"num_key_value_heads": 1,
			"num_kv_shared_layers": 20,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 131072,
			"sliding_window": %d,
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "%s"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
		},
		"quantization_config": {"bits": %d, "group_size": 64}
	}`, slidingWindow, fullRoPEType, quantBits))
	tokenizer := []byte(`{"model":{"type":"BPE","vocab":{"h":0},"merges":[]},"added_tokens":[{"id":100,"content":"<bos>","special":true},{"id":101,"content":"<eos>","special":true}]}`)
	tokenizerConfig := []byte(`{"model_max_length": 131072}`)
	generationConfig := []byte(`{"max_new_tokens": 8192}`)
	chatTemplate := []byte(`{{ bos_token }}{% if enable_thinking %}<|think|>{% endif %}<|channel>thought
{{ reasoning | default('') }}<channel|>{% for message in messages %}{{ message["content"] | replace('<|channel>thought', '') }}{% endfor %}`)
	weights := []byte("weights")
	lock := OfficialGemma4E2BLock{
		Role:                   role,
		ModelID:                modelID,
		Revision:               revision,
		Architecture:           "Gemma4ForConditionalGeneration",
		ModelType:              "gemma4",
		ConfigSHA256:           core.SHA256Hex(config),
		TokenizerSHA256:        core.SHA256Hex(tokenizer),
		TokenizerConfigSHA256:  core.SHA256Hex(tokenizerConfig),
		GenerationConfigSHA256: core.SHA256Hex(generationConfig),
		ChatTemplateSHA256:     core.SHA256Hex(chatTemplate),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex(weights),
		WeightBytes:            uint64(len(weights)),
	}
	dir := core.PathJoin(t.TempDir(), revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4TestFile(t, dir, "config.json", config)
	writeOfficialGemma4TestFile(t, dir, "tokenizer.json", tokenizer)
	writeOfficialGemma4TestFile(t, dir, "tokenizer_config.json", tokenizerConfig)
	writeOfficialGemma4TestFile(t, dir, "generation_config.json", generationConfig)
	writeOfficialGemma4TestFile(t, dir, "chat_template.jinja", chatTemplate)
	writeOfficialGemma4TestFile(t, dir, lock.WeightFile, weights)
	return lock, dir
}
