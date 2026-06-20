// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the MoE expert/router forward (experts.go, router.go) and the MoE
// arm of the compiled layer closure. A quantised MoE model is built with MLX's
// real affine Quantize (group size 32) so GatherQMM serves the expert lanes —
// confirmed present for gs=32 in this metallib. Helpers prefixed cov4moe.

// cov4moeQuantSwitch quantises a dense [experts*out, in] weight to affine q4 and
// reshapes the three entries into the [experts, out, *] switch-linear layout.
func cov4moeQuantSwitch(t *testing.T, experts, out, in int, seed float32) (wq, scales, biases *metal.Array) {
	t.Helper()
	dense := seqArray(seed, experts*out, in)
	q, s, b, err := metal.Quantize(dense, 32, 4, "affine")
	if err != nil {
		t.Fatalf("Quantize switch: %v", err)
	}
	metal.Materialize(q, s, b)
	metal.Free(dense)
	qShape := q.Shape()
	sShape := s.Shape()
	wq = metal.Reshape(q, int32(experts), int32(out), qShape[1])
	scales = metal.Reshape(s, int32(experts), int32(out), sShape[1])
	biases = metal.Reshape(b, int32(experts), int32(out), sShape[1])
	metal.Materialize(wq, scales, biases)
	return wq, scales, biases
}

// cov4moeBuildModel writes and loads a quantised MoE Gemma 4 model: two layers,
// dense projections + router + experts all affine-q4 at group size 32. moeInt is
// the per-expert intermediate; fused selects the gate_up_proj fused expert layout.
func cov4moeBuildModel(t *testing.T, experts, topK, moeInt int, fused bool) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)
	// hidden 64 so the router proj (forced to group size 64 by the loader) fits.
	const H, HD = 64, 32

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
		"max_position_embeddings": 64,
		"rms_norm_eps": 1e-6,
		"sliding_window": 16,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"],
		"enable_moe_block": true,
		"num_experts": %d,
		"top_k_experts": %d,
		"moe_intermediate_size": %d,
		"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}
	}`, H, HD, HD, experts, topK, moeInt)
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	w := map[string]*metal.Array{
		"model.embed_tokens.weight": seqArray(0.001, 10, H),
		"model.norm.weight":         seqArray(0.02, H),
	}
	quant := func(name string, arr *metal.Array) {
		wq, s, b := cov4clQuantize(t, arr)
		w[name+".weight"], w[name+".scales"], w[name+".biases"] = wq, s, b
		metal.Free(arr)
	}
	// the loader forces the router proj to group size 64 / bits 8, so quantise it
	// to match (otherwise the packed weight is unpacked at the wrong stride).
	quantRouter := func(name string, arr *metal.Array) {
		q, s, b, err := metal.Quantize(arr, 64, 8, "affine")
		if err != nil {
			t.Fatalf("Quantize router: %v", err)
		}
		metal.Materialize(q, s, b)
		w[name+".weight"], w[name+".scales"], w[name+".biases"] = q, s, b
		metal.Free(arr)
	}
	switchW := func(base string, e, out, in int, seed float32) {
		wq, s, b := cov4moeQuantSwitch(t, e, out, in, seed)
		w[base+".weight"], w[base+".scales"], w[base+".biases"] = wq, s, b
	}
	for idx := 0; idx < 2; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		w[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx)*0.001, H)
		w[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx)*0.001, H)
		w[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx)*0.001, H)
		w[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx)*0.001, H)
		w[prefix+".pre_feedforward_layernorm_2.weight"] = seqArray(0.055+float32(idx)*0.001, H)
		w[prefix+".post_feedforward_layernorm_1.weight"] = seqArray(0.045+float32(idx)*0.001, H)
		w[prefix+".post_feedforward_layernorm_2.weight"] = seqArray(0.065+float32(idx)*0.001, H)
		w[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx)*0.001, HD)
		w[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx)*0.001, HD)
		quant(prefix+".self_attn.q_proj", seqArray(0.001*float32(idx+1), HD, H))
		quant(prefix+".self_attn.k_proj", seqArray(0.002*float32(idx+1), HD, H))
		quant(prefix+".self_attn.v_proj", seqArray(0.003*float32(idx+1), HD, H))
		quant(prefix+".self_attn.o_proj", seqArray(0.004*float32(idx+1), H, HD))
		// local MLP lane (still present alongside the experts).
		quant(prefix+".mlp.gate_proj", seqArray(0.005*float32(idx+1), 64, H))
		quant(prefix+".mlp.up_proj", seqArray(0.006*float32(idx+1), 64, H))
		quant(prefix+".mlp.down_proj", seqArray(0.007*float32(idx+1), H, 64))
		// router proj: a plain quantised Linear with weight [experts, hidden].
		quantRouter(prefix+".router.proj", seqArray(0.01*float32(idx+1), experts, H))
		// experts.
		if fused {
			switchW(prefix+".experts.switch_glu.gate_up_proj", experts, 2*moeInt, H, 0.02*float32(idx+1))
		} else {
			switchW(prefix+".experts.gate_proj", experts, moeInt, H, 0.02*float32(idx+1))
			switchW(prefix+".experts.up_proj", experts, moeInt, H, 0.03*float32(idx+1))
		}
		switchW(prefix+".experts.down_proj", experts, H, moeInt, 0.04*float32(idx+1))
	}
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), w); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	m, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	t.Cleanup(func() { closeGemma4(m) })
	return m
}

// TestMoE_UncompiledForward_Good runs the MoE model's normal (uncompiled) forward
// so Gemma4Experts.forward and Gemma4Router.forward serve the dual-branch FFN.
func TestMoE_UncompiledForward_Good(t *testing.T) {
	m := cov4moeBuildModel(t, 4, 2, 32, false)
	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := m.NewCache()
	logits := m.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval MoE logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()
	if shape := logits.Shape(); len(shape) != 3 || shape[2] != 10 {
		t.Fatalf("MoE logits shape = %v, want [...10]", shape)
	}
}

// TestMoE_FusedGateUpForward_Good runs the MoE model with the fused gate_up_proj
// expert layout (single-token decode triggers the fused split path).
func TestMoE_FusedGateUpForward_Good(t *testing.T) {
	m := cov4moeBuildModel(t, 4, 2, 32, true)
	// single token so gemma4UseFusedExpertGateUp picks the fused lane.
	tokens := metal.FromValues([]int32{3}, 1, 1)
	caches := m.NewCache()
	logits := m.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval fused MoE logits: %v", err)
	}
	metal.Free(tokens, logits)
	metal.FreeCaches(caches)
}

// TestMoE_CompiledLayerDecode_Good drives the MoE arm of the compiled layer
// closure: a quantised MoE owner layer compiled through a fixed cache.
func TestMoE_CompiledLayerDecode_Good(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4moeBuildModel(t, 4, 2, 32, false)
	layer := m.Layers[1] // full-attention MoE owner
	cache := cov4clPrimeFixed(t, 16, 32, 1)

	before := CompiledLayerDecodeHits()
	x := seqArray(0.02, 1, 1, 64)
	out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	if !ok {
		t.Fatal("MoE owner declined the compiled path")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval MoE compiled hidden: %v", err)
	}
	metal.Free(out, x)
	if CompiledLayerDecodeHits() == before {
		t.Fatal("MoE compiled closure did not serve")
	}
}

// TestMoE_RouterForwardFallback_Good drives Gemma4Router.forward directly with a
// shape NativeMoERouterTopK cannot fuse, exercising the generic argpartition +
// softmax + per-expert-scale fallback arm.
func TestMoE_RouterForwardFallback_Good(t *testing.T) {
	requireMetalRuntime(t)
	const experts = 4
	// router proj [E, H]; scores come out [..., E].
	dense := seqArray(0.05, experts, 32)
	wq, s, b, err := metal.Quantize(dense, 32, 4, "affine")
	if err != nil {
		t.Fatalf("Quantize router proj: %v", err)
	}
	metal.Materialize(wq, s, b)
	metal.Free(dense)
	proj := metal.NewQuantizedLinearWithMode(wq, s, b, nil, 32, 4, "affine")
	defer metal.FreeLinear(proj)

	r := &Gemma4Router{
		Proj:           proj,
		Scale:          gemma4Ones([]int32{32}),
		PerExpertScale: gemma4Ones([]int32{experts}),
		RootSize:       0.5,
		TopK:           2,
		Eps:            1e-6,
	}
	defer metal.Free(r.Scale, r.PerExpertScale)

	x := seqArray(0.01, 1, 3, 32) // multi-token so the native fused top-k declines
	defer metal.Free(x)
	idx, w := r.forward(x)
	defer metal.Free(idx, w)
	metal.Materialize(idx, w)
	if !idx.Valid() || !w.Valid() {
		t.Fatal("router forward produced invalid top-k outputs")
	}
}
