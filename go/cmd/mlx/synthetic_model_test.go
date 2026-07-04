// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// requireSyntheticRuntime gates the synthetic-model command tests on the same
// guard the architecture suites use: a real Metal device built with
// -tags metal_runtime. The synthetic models are TINY (1-2 layers, hidden 64) —
// they load through the REAL model path so the command runners exercise their
// genuine load+generate code, never a stub seam, and never a multi-GB
// checkpoint (AX-11: synthetic micro-load, not a live model load).
func requireSyntheticRuntime(t *testing.T) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// synthArray builds a float32 array of the given shape filled deterministically
// (start + 0.01·i), matching the gemma3/qwen3 fixture convention so the loaded
// weights are reproducible.
func synthArray(start float32, shape ...int) *metal.Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	v := make([]float32, total)
	for i := range v {
		v[i] = start + float32(i)*0.01
	}
	return metal.FromValues(v, shape...)
}

const synthGemma3ConfigJSON = `{
	"model_type": "gemma3",
	"hidden_size": 64,
	"num_hidden_layers": 1,
	"intermediate_size": 128,
	"num_attention_heads": 2,
	"num_key_value_heads": 1,
	"head_dim": 32,
	"vocab_size": 100,
	"rms_norm_eps": 1e-6
}`

const synthTokenizerJSON = `{
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

const synthGemma4TextConfigJSON = `{
	"model_type": "gemma4_text",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 1,
	"num_key_value_heads": 1,
	"head_dim": 4,
	"global_head_dim": 8,
	"vocab_size": 10,
	"max_position_embeddings": 16,
	"rms_norm_eps": 1e-6,
	"sliding_window": 4,
	"sliding_window_pattern": 2,
	"num_kv_shared_layers": 0,
	"hidden_size_per_layer_input": 0,
	"use_double_wide_mlp": false,
	"layer_types": ["sliding_attention", "full_attention"]
}`

// writeSyntheticGemma4TextModel writes a complete, loadable 2-layer gemma4_text
// checkpoint into a fresh temp dir and returns the dir.
func writeSyntheticGemma4TextModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), synthGemma4TextConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), synthTokenizerJSON); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
	w := synthGemma4TrunkWeights()
	defer func() {
		for _, a := range w {
			metal.Free(a)
		}
	}()
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), w); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	return dir
}

// writeSyntheticGemma3Model writes a complete, loadable 1-layer gemma3 model
// (config.json + tokenizer.json + model.safetensors with untied lm_head) into
// dir. It returns dir. The model is just large enough to load through
// mlx.LoadModelAsTextModel / mlx.LoadModel and decode a handful of tokens.
func writeSyntheticGemma3Model(t *testing.T, dir string) string {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), synthGemma3ConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), synthTokenizerJSON); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
	const (
		h     = 64
		inter = 128
		v     = 100
		qOut  = 64 // num_attention_heads(2) * head_dim(32)
		kOut  = 32 // num_key_value_heads(1) * head_dim(32)
		headD = 32
	)
	w := map[string]*metal.Array{
		"model.embed_tokens.weight":                        synthArray(0.01, v, h),
		"model.layers.0.input_layernorm.weight":            synthArray(0.02, h),
		"model.layers.0.post_attention_layernorm.weight":   synthArray(0.03, h),
		"model.layers.0.pre_feedforward_layernorm.weight":  synthArray(0.04, h),
		"model.layers.0.post_feedforward_layernorm.weight": synthArray(0.05, h),
		"model.layers.0.self_attn.q_proj.weight":           synthArray(0.06, qOut, h),
		"model.layers.0.self_attn.k_proj.weight":           synthArray(0.07, kOut, h),
		"model.layers.0.self_attn.v_proj.weight":           synthArray(0.08, kOut, h),
		"model.layers.0.self_attn.o_proj.weight":           synthArray(0.09, h, qOut),
		"model.layers.0.self_attn.q_norm.weight":           synthArray(0.10, headD),
		"model.layers.0.self_attn.k_norm.weight":           synthArray(0.11, headD),
		"model.layers.0.mlp.gate_proj.weight":              synthArray(0.12, inter, h),
		"model.layers.0.mlp.up_proj.weight":                synthArray(0.13, inter, h),
		"model.layers.0.mlp.down_proj.weight":              synthArray(0.14, h, inter),
		"model.norm.weight":                                synthArray(0.15, h),
		"lm_head.weight":                                   synthArray(0.16, v, h),
	}
	defer func() {
		for _, a := range w {
			metal.Free(a)
		}
	}()
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), w); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	return dir
}

const synthDiffusionConfigJSON = `{
	"model_type": "diffusion_gemma",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 1,
	"num_key_value_heads": 1,
	"head_dim": 4,
	"global_head_dim": 8,
	"vocab_size": 10,
	"max_position_embeddings": 16,
	"rms_norm_eps": 1e-6,
	"sliding_window": 4,
	"sliding_window_pattern": 2,
	"num_kv_shared_layers": 0,
	"hidden_size_per_layer_input": 0,
	"use_double_wide_mlp": false,
	"canvas_length": 4,
	"eos_token_id": 1,
	"layer_types": ["sliding_attention", "full_attention"]
}`

// writeSyntheticDiffusionModel writes a complete, loadable 2-layer
// diffusion_gemma checkpoint (the weight-tied trunk + the self-conditioning
// block + encoder-role per-layer scalars the diffusion loader collects) into
// dir. Mirrors the gemma4 package's diffusion fixture so runDiffuseCommand
// loads through gemma4.LoadDiffusionGemma on a tiny synthetic.
func writeSyntheticDiffusionModel(t *testing.T, dir string) string {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), synthDiffusionConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), synthTokenizerJSON); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
	w := synthGemma4TrunkWeights()
	// Diffusion extras the loader extracts on top of the trunk.
	w["self_conditioning.pre_norm.weight"] = synthArray(0.02, 8)
	w["self_conditioning.gate_proj.weight"] = synthArray(0.7, 16, 8)
	w["self_conditioning.up_proj.weight"] = synthArray(0.8, 16, 8)
	w["self_conditioning.down_proj.weight"] = synthArray(0.9, 8, 16)
	w["model.encoder.language_model.layers.0.layer_scalar"] = metal.FromValues([]float32{1}, 1)
	w["model.encoder.language_model.layers.1.layer_scalar"] = metal.FromValues([]float32{1}, 1)
	defer func() {
		for _, a := range w {
			metal.Free(a)
		}
	}()
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), w); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	return dir
}

// synthGemma4TrunkWeights builds the shared 2-layer gemma4 trunk (hidden 8) the
// diffusion synthetic rides on — one sliding-attention layer (head_dim 4) and
// one full-attention layer (head_dim 8). Mirrors gemma4TinyWeights in the
// gemma4 package, which is unexported there.
func synthGemma4TrunkWeights() map[string]*metal.Array {
	w := map[string]*metal.Array{
		"model.embed_tokens.weight": synthArray(0.01, 10, 8),
		"model.norm.weight":         synthArray(0.02, 8),
	}
	addLayer := func(idx int, sliding bool) {
		prefix := core.Sprintf("model.layers.%d", idx)
		headDim, oIn := 8, 8
		if sliding {
			headDim, oIn = 4, 4
		}
		w[prefix+".input_layernorm.weight"] = synthArray(0.03+float32(idx), 8)
		w[prefix+".post_attention_layernorm.weight"] = synthArray(0.04+float32(idx), 8)
		w[prefix+".pre_feedforward_layernorm.weight"] = synthArray(0.05+float32(idx), 8)
		w[prefix+".post_feedforward_layernorm.weight"] = synthArray(0.06+float32(idx), 8)
		w[prefix+".layer_scalar"] = metal.FromValues([]float32{1}, 1)
		w[prefix+".self_attn.q_proj.weight"] = synthArray(0.10+float32(idx), headDim, 8)
		w[prefix+".self_attn.k_proj.weight"] = synthArray(0.20+float32(idx), headDim, 8)
		w[prefix+".self_attn.v_proj.weight"] = synthArray(0.30+float32(idx), headDim, 8)
		w[prefix+".self_attn.o_proj.weight"] = synthArray(0.40+float32(idx), 8, oIn)
		w[prefix+".self_attn.q_norm.weight"] = synthArray(0.50+float32(idx), headDim)
		w[prefix+".self_attn.k_norm.weight"] = synthArray(0.60+float32(idx), headDim)
		w[prefix+".mlp.gate_proj.weight"] = synthArray(0.70+float32(idx), 16, 8)
		w[prefix+".mlp.up_proj.weight"] = synthArray(0.80+float32(idx), 16, 8)
		w[prefix+".mlp.down_proj.weight"] = synthArray(0.90+float32(idx), 8, 16)
	}
	addLayer(0, true)
	addLayer(1, false)
	return w
}
