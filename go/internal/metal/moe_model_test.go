// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestModel_LoadModel_Qwen3MoEFullLoad_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	for e := 0; e < 2; e++ {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(qwen3_moe) error = %v", err)
	}
	if model.ModelType() != "qwen3_moe" {
		t.Fatalf("ModelType() = %q, want qwen3_moe", model.ModelType())
	}
	if _, ok := model.(*Qwen3MoEModel); !ok {
		t.Fatalf("model type = %T, want *Qwen3MoEModel", model)
	}

	info := (&Model{model: model, tokenizer: model.Tokenizer(), modelType: model.ModelType()}).Info()
	if info.VocabSize != 5 || info.HiddenSize != 8 || info.NumLayers != 1 {
		t.Fatalf("Info() = %+v, want vocab=5 hidden=8 layers=1", info)
	}
}

func TestModel_LoadModel_MixtralFullLoad_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "mixtral",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 1,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	for e := 0; e < 2; e++ {
		p := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", e)
		weights[p+".w1.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".w2.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".w3.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.block_sparse_moe.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(mixtral) error = %v", err)
	}
	if model.ModelType() != "mixtral" {
		t.Fatalf("ModelType() = %q, want mixtral", model.ModelType())
	}
	if _, ok := model.(*MixtralModel); !ok {
		t.Fatalf("model type = %T, want *MixtralModel", model)
	}
}

func TestModel_LoadModel_KimiFullLoad_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 1,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	for e := 0; e < 2; e++ {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(kimi) error = %v", err)
	}
	if model.ModelType() != "kimi" {
		t.Fatalf("ModelType() = %q, want kimi", model.ModelType())
	}
	if _, ok := model.(*KimiModel); !ok {
		t.Fatalf("model type = %T, want *KimiModel", model)
	}
}

func TestModel_Generate_Qwen3MoEDiagnostic_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 0,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	for e := 0; e < 2; e++ {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3_moe) error = %v", err)
	}
	defer model.Close()

	var genCount int
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 2}) {
		genCount++
	}
	if genCount != 2 {
		t.Fatalf("generated %d token(s), want 2 with native sparse-expert decode", genCount)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil after native sparse-expert decode", err)
	}
}

func TestModel_Generate_SharedMoEFamilies_Good(t *testing.T) {
	coverageTokens := "Model Generate SharedMoEFamilies"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cases := []struct {
		name        string
		modelType   string
		writeConfig func(t *testing.T, dir string)
		build       func() map[string]*Array
	}{
		{
			name:      "mixtral",
			modelType: "mixtral",
			writeConfig: func(t *testing.T, dir string) {
				t.Helper()
				_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
					"model_type": "mixtral",
					"hidden_size": 8,
					"num_hidden_layers": 1,
					"num_attention_heads": 2,
					"num_key_value_heads": 2,
					"head_dim": 4,
					"vocab_size": 5,
					"max_position_embeddings": 32,
					"rms_norm_eps": 1e-6,
					"rope_theta": 1000000,
					"decoder_sparse_step": 1,
					"num_local_experts": 2,
					"num_experts_per_tok": 2
				}`)
			},
			build: func() map[string]*Array {
				return tinyMixtralMoEWeights(8, 16, 2, 5)
			},
		},
		{
			name:      "kimi",
			modelType: "kimi",
			writeConfig: func(t *testing.T, dir string) {
				t.Helper()
				_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
					"model_type": "kimi",
					"hidden_size": 8,
					"num_hidden_layers": 1,
					"num_attention_heads": 2,
					"num_key_value_heads": 2,
					"head_dim": 4,
					"vocab_size": 5,
					"max_position_embeddings": 32,
					"rms_norm_eps": 1e-6,
					"rope_theta": 1000000,
					"decoder_sparse_step": 1,
					"num_local_experts": 2,
					"num_experts_per_tok": 2
				}`)
			},
			build: func() map[string]*Array {
				return tinyQwenStyleMoEWeights(8, 16, 2, 5)
			},
		},
		{
			name:      "gpt_oss",
			modelType: "gpt_oss",
			writeConfig: func(t *testing.T, dir string) {
				t.Helper()
				_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
					"model_type": "gpt_oss",
					"hidden_size": 8,
					"num_hidden_layers": 1,
					"num_attention_heads": 2,
					"num_key_value_heads": 2,
					"head_dim": 4,
					"vocab_size": 6,
					"max_position_embeddings": 32,
					"rms_norm_eps": 1e-6,
					"rope_theta": 1000000,
					"decoder_sparse_step": 1,
					"num_local_experts": 2,
					"num_experts_per_tok": 2
				}`)
			},
			build: func() map[string]*Array {
				return tinyQwenStyleMoEWeights(8, 16, 2, 6)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.writeConfig(t, dir)
			writeMinimalTokenizer(t, dir)

			weights := tc.build()
			defer freeArrayMap(weights)
			if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
				t.Fatalf("SaveSafetensors: %v", err)
			}

			model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
			if err != nil {
				t.Fatalf("LoadAndInit(%s) error = %v", tc.modelType, err)
			}
			defer model.Close()

			var genCount int
			for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 2}) {
				genCount++
			}
			if genCount != 2 {
				t.Fatalf("generated %d token(s), want 2 with native shared sparse-expert decode", genCount)
			}
			if err := model.Err(); err != nil {
				t.Fatalf("Err() = %v, want nil after native shared sparse-expert decode", err)
			}
		})
	}
}

func tinyMoEDecoderWeights(hidden, intermediate, experts, vocab int32) map[string]*Array {
	h := int(hidden)
	i := int(intermediate)
	v := int(vocab)
	return map[string]*Array{
		"model.embed_tokens.weight":                      seqArray(0.01, v, h),
		"model.layers.0.input_layernorm.weight":          seqArray(0.02, h),
		"model.layers.0.post_attention_layernorm.weight": seqArray(0.03, h),
		"model.layers.0.self_attn.q_proj.weight":         seqArray(0.04, h, h),
		"model.layers.0.self_attn.k_proj.weight":         seqArray(0.05, h, h),
		"model.layers.0.self_attn.v_proj.weight":         seqArray(0.06, h, h),
		"model.layers.0.self_attn.o_proj.weight":         seqArray(0.07, h, h),
		"model.layers.0.mlp.gate_proj.weight":            seqArray(0.08, i, h),
		"model.layers.0.mlp.up_proj.weight":              seqArray(0.09, i, h),
		"model.layers.0.mlp.down_proj.weight":            seqArray(0.10, h, i),
		"model.norm.weight":                              seqArray(0.11, h),
		"lm_head.weight":                                 seqArray(0.12, v, h),
	}
}

func tinyQwenStyleMoEWeights(hidden, intermediate, experts, vocab int32) map[string]*Array {
	weights := tinyMoEDecoderWeights(hidden, intermediate, experts, vocab)
	for e := int32(0); e < experts; e++ {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, int(intermediate), int(hidden))
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, int(intermediate), int(hidden))
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, int(hidden), int(intermediate))
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, int(experts), int(hidden))
	return weights
}

func tinyMixtralMoEWeights(hidden, intermediate, experts, vocab int32) map[string]*Array {
	weights := tinyMoEDecoderWeights(hidden, intermediate, experts, vocab)
	for e := int32(0); e < experts; e++ {
		p := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", e)
		weights[p+".w1.weight"] = seqArray(0.30+float32(e)*0.03, int(intermediate), int(hidden))
		weights[p+".w2.weight"] = seqArray(0.31+float32(e)*0.03, int(hidden), int(intermediate))
		weights[p+".w3.weight"] = seqArray(0.32+float32(e)*0.03, int(intermediate), int(hidden))
	}
	weights["model.layers.0.block_sparse_moe.gate.weight"] = seqArray(0.20, int(experts), int(hidden))
	return weights
}

func TestModel_LoadModel_GptOssFullLoad_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gpt_oss",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 6,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 1,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 6)
	for e := 0; e < 2; e++ {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := loadModel(dir)
	if err != nil {
		t.Fatalf("loadModel(gpt_oss) error = %v", err)
	}
	if model.ModelType() != "gpt_oss" {
		t.Fatalf("ModelType() = %q, want gpt_oss", model.ModelType())
	}
	if _, ok := model.(*GptOssModel); !ok {
		t.Fatalf("model type = %T, want *GptOssModel", model)
	}
}
