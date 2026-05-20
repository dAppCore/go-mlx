// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestGemma4Assistant_LoadGemma4Assistant_Good(t *testing.T) {
	coverageTokens := "Gemma4Assistant LoadGemma4Assistant"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	dir := t.TempDir()
	writeGemma4AssistantConfig(t, dir, true)
	writeMinimalTokenizer(t, dir)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4AssistantTinyWeights(true)); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4Assistant(dir)
	if err != nil {
		t.Fatalf("LoadGemma4Assistant: %v", err)
	}
	defer model.Close()

	if model.ModelType() != "gemma4_assistant" || model.NumLayers() != 2 || model.Tokenizer() == nil {
		t.Fatalf("assistant metadata = %s/%d/%v", model.ModelType(), model.NumLayers(), model.Tokenizer())
	}
	if !model.UseOrderedEmbeddings || model.MaskedCentroids == nil || model.TokenOrdering == nil {
		t.Fatalf("ordered embedding tensors not loaded: centroids=%v ordering=%v", model.MaskedCentroids, model.TokenOrdering)
	}
	if model.PreProjection.Weight.Shape()[1] != 16 || model.PostProjection.Weight.Shape()[0] != 8 {
		t.Fatalf("projection shapes = %v/%v", model.PreProjection.Weight.Shape(), model.PostProjection.Weight.Shape())
	}
}

func TestGemma4Assistant_LoadGemma4AssistantPair_Good(t *testing.T) {
	coverageTokens := "Gemma4Assistant LoadGemma4AssistantPair"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	targetDir := t.TempDir()
	writeGemma4AssistantTargetConfig(t, targetDir)
	writeMinimalTokenizer(t, targetDir)
	if err := SaveSafetensors(core.JoinPath(targetDir, "model.safetensors"), gemma4AssistantTargetTinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors target: %v", err)
	}

	assistantDir := t.TempDir()
	writeGemma4AssistantConfig(t, assistantDir, true)
	writeMinimalTokenizer(t, assistantDir)
	if err := SaveSafetensors(core.JoinPath(assistantDir, "model.safetensors"), gemma4AssistantTinyWeights(true)); err != nil {
		t.Fatalf("SaveSafetensors assistant: %v", err)
	}

	pair, err := LoadGemma4AssistantPair(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPair: %v", err)
	}
	defer pair.Close()

	if pair.Target == nil || pair.Assistant == nil {
		t.Fatalf("pair = %+v, want target and assistant", pair)
	}
	if pair.Target.Cfg.HiddenSize != pair.Assistant.BackboneHiddenSize {
		t.Fatalf("hidden/backbone = %d/%d, want match", pair.Target.Cfg.HiddenSize, pair.Assistant.BackboneHiddenSize)
	}
}

func TestGemma4Assistant_AttachGemma4Assistant_Bad(t *testing.T) {
	coverageTokens := "Gemma4Assistant AttachGemma4Assistant Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}

	target := &Gemma4Model{Cfg: &Gemma4TextConfig{HiddenSize: 12, VocabSize: 10}}
	assistant := &Gemma4AssistantModel{Cfg: &Gemma4TextConfig{VocabSize: 10}, BackboneHiddenSize: 8}
	_, err := AttachGemma4Assistant(target, assistant)
	if err == nil {
		t.Fatal("AttachGemma4Assistant() error = nil, want hidden-size mismatch")
	}
	if !core.Contains(err.Error(), "backbone_hidden_size") {
		t.Fatalf("AttachGemma4Assistant() error = %v, want backbone_hidden_size", err)
	}
}

func TestGemma4Assistant_LoadLocalAssistantPack_Good(t *testing.T) {
	coverageTokens := "Gemma4Assistant LoadLocalAssistantPack"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	modelPath := core.Trim(core.Env("GO_MLX_GEMMA4_ASSISTANT_MODEL"))
	if modelPath == "" {
		t.Skip("set GO_MLX_GEMMA4_ASSISTANT_MODEL to run the local assistant pack smoke")
	}
	model, err := LoadGemma4Assistant(modelPath)
	if err != nil {
		t.Fatalf("LoadGemma4Assistant(%s): %v", modelPath, err)
	}
	defer model.Close()
	if model.ModelType() != "gemma4_assistant" || model.NumLayers() != 4 {
		t.Fatalf("assistant metadata = %s/%d, want gemma4_assistant/4", model.ModelType(), model.NumLayers())
	}
	if model.BackboneHiddenSize <= 0 || model.PreProjection == nil || model.PostProjection == nil {
		t.Fatalf("assistant projections/backbone not loaded: backbone=%d pre=%v post=%v", model.BackboneHiddenSize, model.PreProjection, model.PostProjection)
	}
}

func TestGemma4Assistant_LoadLocalAssistantPair_Good(t *testing.T) {
	coverageTokens := "Gemma4Assistant LoadLocalAssistantPair"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	targetPath := core.Trim(core.Env("GO_MLX_GEMMA4_TARGET_MODEL"))
	assistantPath := core.Trim(core.Env("GO_MLX_GEMMA4_ASSISTANT_MODEL"))
	if targetPath == "" || assistantPath == "" {
		t.Skip("set GO_MLX_GEMMA4_TARGET_MODEL and GO_MLX_GEMMA4_ASSISTANT_MODEL to run the local target+assistant smoke")
	}
	pair, err := LoadGemma4AssistantPair(targetPath, assistantPath)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPair(%s, %s): %v", targetPath, assistantPath, err)
	}
	defer pair.Close()
	if pair.Target == nil || pair.Assistant == nil {
		t.Fatalf("pair = %+v, want target and assistant", pair)
	}
}

func TestGemma4Assistant_LoadGemma4Assistant_Bad(t *testing.T) {
	coverageTokens := "Gemma4Assistant LoadGemma4Assistant Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	dir := t.TempDir()
	writeGemma4AssistantConfig(t, dir, false)
	writeMinimalTokenizer(t, dir)
	weights := gemma4AssistantTinyWeights(false)
	Free(weights["post_projection.weight"])
	delete(weights, "post_projection.weight")
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	_, err := LoadGemma4Assistant(dir)
	if err == nil {
		t.Fatal("LoadGemma4Assistant() error = nil, want missing post_projection")
	}
	if !core.Contains(err.Error(), "post_projection.weight") {
		t.Fatalf("LoadGemma4Assistant() error = %v, want post_projection.weight", err)
	}
}

func TestGemma4Assistant_ParseConfig_Ugly(t *testing.T) {
	coverageTokens := "Gemma4Assistant ParseConfig Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	_, err := parseGemma4AssistantConfig([]byte(`{
		"model_type": "gemma4_assistant",
		"backbone_hidden_size": 0,
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 4,
			"num_hidden_layers": 1,
			"intermediate_size": 8,
			"num_attention_heads": 1,
			"num_key_value_heads": 1,
			"head_dim": 4,
			"vocab_size": 10,
			"rms_norm_eps": 1e-6
		}
	}`))
	if err == nil {
		t.Fatal("parseGemma4AssistantConfig() error = nil, want invalid backbone_hidden_size")
	}
	if !core.Contains(err.Error(), "backbone_hidden_size") {
		t.Fatalf("parseGemma4AssistantConfig() error = %v, want backbone_hidden_size", err)
	}
}

func writeGemma4AssistantTargetConfig(t *testing.T, dir string) {
	t.Helper()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 4,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"layer_types": ["sliding_attention", "full_attention"],
		"rope_parameters": {
			"sliding_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000, "rope_type": "default"},
			"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000, "rope_type": "default"}
		}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write target config.json: %v", err)
	}
}

func writeGemma4AssistantConfig(t *testing.T, dir string, ordered bool) {
	t.Helper()
	orderedText := "false"
	if ordered {
		orderedText = "true"
	}
	config := `{
		"architectures": ["Gemma4AssistantForCausalLM"],
		"model_type": "gemma4_assistant",
		"backbone_hidden_size": 8,
		"num_centroids": 3,
		"centroid_intermediate_top_k": 2,
		"use_ordered_embeddings": ` + orderedText + `,
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 4,
			"num_hidden_layers": 2,
			"intermediate_size": 8,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 4,
			"global_head_dim": 4,
			"hidden_size_per_layer_input": 0,
			"vocab_size": 10,
			"vocab_size_per_layer_input": 0,
			"rms_norm_eps": 1e-6,
			"layer_types": ["sliding_attention", "full_attention"],
			"rope_parameters": {
				"sliding_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000, "rope_type": "default"},
				"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000, "rope_type": "default"}
			}
		}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

func gemma4AssistantTargetTinyWeights() map[string]*Array {
	weights := map[string]*Array{
		"model.embed_tokens.weight": seqArray(0.01, 10, 8),
		"model.norm.weight":         seqArray(0.02, 8),
	}
	for idx := 0; idx < 2; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		weights[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx), 8)
		weights[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx), 8)
		weights[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx), 8)
		weights[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx), 8)
		weights[prefix+".layer_scalar"] = FromValues([]float32{1}, 1)
		weights[prefix+".self_attn.q_proj.weight"] = seqArray(0.10+float32(idx), 8, 8)
		weights[prefix+".self_attn.k_proj.weight"] = seqArray(0.20+float32(idx), 4, 8)
		weights[prefix+".self_attn.v_proj.weight"] = seqArray(0.30+float32(idx), 4, 8)
		weights[prefix+".self_attn.o_proj.weight"] = seqArray(0.40+float32(idx), 8, 8)
		weights[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx), 4)
		weights[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx), 4)
		weights[prefix+".mlp.gate_proj.weight"] = seqArray(0.70+float32(idx), 16, 8)
		weights[prefix+".mlp.up_proj.weight"] = seqArray(0.80+float32(idx), 16, 8)
		weights[prefix+".mlp.down_proj.weight"] = seqArray(0.90+float32(idx), 8, 16)
	}
	return weights
}

func gemma4AssistantTinyWeights(ordered bool) map[string]*Array {
	weights := map[string]*Array{
		"model.embed_tokens.weight": seqArray(0.01, 10, 4),
		"model.norm.weight":         seqArray(0.02, 4),
		"pre_projection.weight":     seqArray(0.03, 4, 16),
		"post_projection.weight":    seqArray(0.04, 8, 4),
	}
	if ordered {
		weights["masked_embedding.centroids.weight"] = seqArray(0.05, 3, 4)
		weights["masked_embedding.token_ordering"] = FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10)
	}
	for idx := 0; idx < 2; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		weights[prefix+".input_layernorm.weight"] = seqArray(0.10+float32(idx), 4)
		weights[prefix+".post_attention_layernorm.weight"] = seqArray(0.11+float32(idx), 4)
		weights[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.12+float32(idx), 4)
		weights[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.13+float32(idx), 4)
		weights[prefix+".layer_scalar"] = FromValues([]float32{1}, 1)
		weights[prefix+".self_attn.q_proj.weight"] = seqArray(0.20+float32(idx), 8, 4)
		weights[prefix+".self_attn.o_proj.weight"] = seqArray(0.21+float32(idx), 4, 8)
		weights[prefix+".self_attn.q_norm.weight"] = seqArray(0.22+float32(idx), 4)
		weights[prefix+".mlp.gate_proj.weight"] = seqArray(0.30+float32(idx), 8, 4)
		weights[prefix+".mlp.up_proj.weight"] = seqArray(0.31+float32(idx), 8, 4)
		weights[prefix+".mlp.down_proj.weight"] = seqArray(0.32+float32(idx), 4, 8)
	}
	return weights
}
