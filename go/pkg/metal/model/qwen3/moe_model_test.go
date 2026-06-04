// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
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
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3MoE(dir)
	if err != nil {
		t.Fatalf("LoadQwen3MoE(qwen3_moe) error = %v", err)
	}
	if model.ModelType() != "qwen3_moe" {
		t.Fatalf("ModelType() = %q, want qwen3_moe", model.ModelType())
	}

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 5 || info.HiddenSize != 8 || info.NumLayers != 1 {
		t.Fatalf("Info() = %+v, want vocab=5 hidden=8 layers=1", info)
	}
}

func TestModel_LoadModel_Qwen3MoEModelTypeDispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000,
		"max_position_embeddings": 32768,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 384,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := LoadQwen3MoE(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for qwen3_moe without safetensors")
	}
	if !core.Contains(err.Error(), "qwen3_moe") {
		t.Fatalf("error = %v, should contain qwen3_moe", err)
	}
}

// Kimi full-load coverage travels with the model in package metal/model/kimi.
// Mixtral full-load coverage travels with the model in package
// metal/model/mixtral.

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
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := metal.LoadAndInit(dir, metal.LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3_moe) error = %v", err)
	}
	defer model.Close()

	var genCount int
	for range model.Generate(context.Background(), "hello", metal.GenerateConfig{MaxTokens: 2}) {
		genCount++
	}
	if genCount != 2 {
		t.Fatalf("generated %d token(s), want 2 with native sparse-expert decode", genCount)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil after native sparse-expert decode", err)
	}
}

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	model := &Qwen3MoEModel{
		Layers: []*Qwen3MoEDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &Qwen3MoEBlock{
				Router:        router,
				Experts:       []*Qwen3MoEExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !model.MoETextRuntimeAvailable() {
		t.Fatal("Qwen3MoEModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := model.MoETextDecodeFamily(); got != "qwen3_moe" {
		t.Fatalf("MoETextDecodeFamily() = %q, want qwen3_moe", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&Qwen3MoEModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty Qwen3MoEModel.MoETextRuntimeAvailable() = true, want false")
	}
	incomplete := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete Qwen3MoEModel.MoETextRuntimeAvailable() = true, want false")
	}
}

func tinyMoEDecoderWeights(hidden, intermediate, experts, vocab int32) map[string]*metal.Array {
	h := int(hidden)
	i := int(intermediate)
	v := int(vocab)
	return map[string]*metal.Array{
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

// GPT-OSS full-load coverage travels with package metal/model/gptoss.

func seqArray(start float32, shape ...int) *metal.Array {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	values := make([]float32, total)
	for i := range values {
		values[i] = start + float32(i)*0.01
	}
	return metal.FromValues(values, shape...)
}

func moeReadyRuntimeParts(t *testing.T) (*metal.MoERouter, *metal.MoESwiGLUExperts, func()) {
	t.Helper()
	routerWeight := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	gate := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil)}
	up := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{0.5, 0, 0, 0.5}, 2, 2), nil)}
	down := []*metal.Linear{metal.NewLinear(metal.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil)}
	experts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		t.Fatal("NewMoESwiGLUExpertsFromLinears() ok = false, want true")
	}
	metal.Materialize(routerWeight)
	cleanup := func() {
		metal.Free(routerWeight)
		metal.FreeMoESwiGLUExperts(experts)
	}
	return &metal.MoERouter{Weight: routerWeight}, experts, cleanup
}

func freeArrayMap(arrays map[string]*metal.Array) {
	for _, array := range arrays {
		metal.Free(array)
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
