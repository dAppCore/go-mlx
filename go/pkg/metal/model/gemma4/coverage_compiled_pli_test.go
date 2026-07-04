// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the per-layer-input (PLI) arm of the compiled layer closure. A
// quantised model with hidden_size_per_layer_input set carries a PLI gate +
// projection + norm per layer; driving its compiled decode with a PLI tensor
// covers the pli branch of gemma4CompiledLayerStep and the presence checks in
// compiledDecodeForward / buildGemma4CompiledLayerState. Helpers prefixed cov4pli.

// cov4pliBuildModel writes and loads a quantised 2-layer PLI model. The per-layer
// input dim is 32 (a multiple of the group size so the projection quantises).
func cov4pliBuildModel(t *testing.T) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)
	const H, HD, PLI = 32, 32, 32

	dir := t.TempDir()
	config := core.Sprintf(`{
		"model_type": "gemma4_text",
		"hidden_size": %d,
		"num_hidden_layers": 2,
		"intermediate_size": 64,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": %d,
		"global_head_dim": %d,
		"vocab_size": 10,
		"vocab_size_per_layer_input": 10,
		"hidden_size_per_layer_input": %d,
		"max_position_embeddings": 64,
		"rms_norm_eps": 1e-6,
		"sliding_window": 16,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"],
		"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}
	}`, H, HD, HD, PLI)
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	w := map[string]*metal.Array{
		"model.embed_tokens.weight": seqArray(0.001, 10, H),
		"model.norm.weight":         seqArray(0.02, H),
		// per-layer-input trunk (norm is plain; embeddings + projection are bf16).
		"model.embed_tokens_per_layer.weight":     seqArray(0.0011, 10, PLI),
		"model.per_layer_model_projection.weight": seqArray(0.0012, 2*PLI, H),
		"model.per_layer_projection_norm.weight":  seqArray(0.0013, PLI),
	}
	quant := func(name string, arr *metal.Array) {
		wq, s, b := cov4clQuantize(t, arr)
		w[name+".weight"], w[name+".scales"], w[name+".biases"] = wq, s, b
		metal.Free(arr)
	}
	for idx := 0; idx < 2; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		w[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx)*0.001, H)
		w[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx)*0.001, H)
		w[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx)*0.001, H)
		w[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx)*0.001, H)
		w[prefix+".post_per_layer_input_norm.weight"] = seqArray(0.07+float32(idx)*0.001, H)
		w[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx)*0.001, HD)
		w[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx)*0.001, HD)
		quant(prefix+".self_attn.q_proj", seqArray(0.001*float32(idx+1), HD, H))
		quant(prefix+".self_attn.k_proj", seqArray(0.002*float32(idx+1), HD, H))
		quant(prefix+".self_attn.v_proj", seqArray(0.003*float32(idx+1), HD, H))
		quant(prefix+".self_attn.o_proj", seqArray(0.004*float32(idx+1), H, HD))
		quant(prefix+".mlp.gate_proj", seqArray(0.005*float32(idx+1), 64, H))
		quant(prefix+".mlp.up_proj", seqArray(0.006*float32(idx+1), 64, H))
		quant(prefix+".mlp.down_proj", seqArray(0.007*float32(idx+1), H, 64))
		// PLI gate [pliDim, hidden] and projection [hidden, pliDim].
		quant(prefix+".per_layer_input_gate", seqArray(0.008*float32(idx+1), PLI, H))
		quant(prefix+".per_layer_projection", seqArray(0.009*float32(idx+1), H, PLI))
	}
	cov4SaveAndFree(t, core.JoinPath(dir, "model.safetensors"), w)
	m, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	t.Cleanup(func() { closeGemma4(m) })
	return m
}

// TestCompiledLayer_PerLayerInputDecode_Good drives the PLI arm of the compiled
// closure: the layer carries a per-layer-input gate/projection, so a PLI tensor
// is supplied and the gated-projection block runs inside the trace.
func TestCompiledLayer_PerLayerInputDecode_Good(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4pliBuildModel(t)
	if m.Cfg.HiddenSizePerLayerInput != 32 {
		t.Fatalf("HiddenSizePerLayerInput = %d, want 32 (PLI not wired)", m.Cfg.HiddenSizePerLayerInput)
	}
	layer := m.Layers[1]
	if layer.PerLayerInputGate == nil || layer.PerLayerProjection == nil {
		t.Fatal("layer 1 missing PLI gate/projection")
	}
	cache := cov4clPrimeFixed(t, 16, 32, 1)

	before := CompiledLayerDecodeHits()
	x := seqArray(0.02, 1, 1, 32)
	pli := seqArray(0.05, 1, 1, 32) // one token's per-layer input
	out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, pli, sharedKV{}, m.Cfg)
	if !ok {
		t.Fatal("PLI owner declined the compiled path")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval PLI hidden: %v", err)
	}
	metal.Free(out, x, pli)
	if CompiledLayerDecodeHits() == before {
		t.Fatal("PLI compiled closure did not serve")
	}
}

// TestCompiledLayer_PerLayerInputPresenceMismatch_Bad covers the decline branch
// where the layer expects a PLI tensor but none is supplied (or vice versa).
func TestCompiledLayer_PerLayerInputPresenceMismatch_Bad(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4pliBuildModel(t)
	layer := m.Layers[1]
	cache := cov4clPrimeFixed(t, 16, 32, 1)

	// PLI layer but nil pli tensor → presence mismatch decline.
	x := seqArray(0.02, 1, 1, 32)
	defer metal.Free(x)
	if _, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg); ok {
		t.Fatal("PLI layer with no pli tensor must decline")
	}
}

// Note: the uncompiled per-layer-input forward path (decoder_layer.go / perlayer.go)
// is exercised by the existing TestGemma4_LoadAndForwardPerLayerInputModel_Good in
// model_test.go. Driving a second PLI Forward here interacted badly with a later
// safetensors load in the same process (an MLX loader-state quirk surfacing as
// "expected a non-empty mlx_array"), so this file covers only the PLI compiled
// arm, which is the gap that existing test leaves.
