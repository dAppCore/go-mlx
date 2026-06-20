// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage tests for the whole-layer compiled decode path (compiled_layer.go).
//
// The compiled replay closure (gemma4CompiledLayerStep) only runs on a layer
// whose projections are affine-quantised, so these tests build a tiny quantised
// Gemma 4 model with MLX's real Quantize (group size 32, the smallest MLX
// supports) and drive a loaded layer's compiledDecodeForward directly through
// each regime: owner pre-capacity, owner post-capacity (sliding rotate), and the
// trace-decline ladder. Every helper is prefixed cov4cl to stay unique in the
// package test scope.

// cov4clQuantize converts a float32 weight to affine q4 and materialises the
// three safetensor entries.
func cov4clQuantize(t *testing.T, w *metal.Array) (wq, scales, biases *metal.Array) {
	t.Helper()
	q, s, b, err := metal.Quantize(w, 32, 4, "affine")
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	metal.Materialize(q, s, b)
	return q, s, b
}

// cov4clQuantConfig holds the dimensions a quantised tiny model is built at. The
// projection input dims (hidden, intermediate) must each be a multiple of 32.
type cov4clQuantConfig struct {
	hidden       int
	intermediate int
	headDim      int
	layers       int
	sliding      int
	pattern      int
	numShared    int
	kEqV         bool
}

// cov4clBuildQuantModel writes a quantised tiny Gemma 4 model to a temp dir and
// loads it. The weights are deterministic seqArray fills quantised at load time.
func cov4clBuildQuantModel(t *testing.T, qc cov4clQuantConfig) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)
	dir := t.TempDir()

	layerTypes := make([]string, qc.layers)
	for i := range layerTypes {
		lt := "full_attention"
		if qc.pattern > 1 && (i+1)%qc.pattern != 0 {
			lt = "sliding_attention"
		}
		if i == qc.layers-1 {
			lt = "full_attention"
		}
		layerTypes[i] = core.Sprintf("%q", lt)
	}
	ltJSON := ""
	for i, lt := range layerTypes {
		if i > 0 {
			ltJSON += ","
		}
		ltJSON += lt
	}

	config := core.Sprintf(`{
		"model_type": "gemma4_text",
		"hidden_size": %d,
		"num_hidden_layers": %d,
		"intermediate_size": %d,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": %d,
		"global_head_dim": %d,
		"vocab_size": 10,
		"max_position_embeddings": 64,
		"rms_norm_eps": 1e-6,
		"sliding_window": %d,
		"sliding_window_pattern": %d,
		"num_kv_shared_layers": %d,
		"hidden_size_per_layer_input": 0,
		"attention_k_eq_v": %t,
		"use_double_wide_mlp": false,
		"layer_types": [%s],
		"quantization": {"group_size": 32, "bits": 4, "mode": "affine"}
	}`, qc.hidden, qc.layers, qc.intermediate, qc.headDim, qc.headDim, qc.sliding, qc.pattern, qc.numShared, qc.kEqV, ltJSON)
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	H, I, HD := qc.hidden, qc.intermediate, qc.headDim
	w := map[string]*metal.Array{
		"model.embed_tokens.weight": seqArray(0.001, 10, H),
		"model.norm.weight":         seqArray(0.02, H),
	}
	quant := func(name string, arr *metal.Array) {
		wq, s, b := cov4clQuantize(t, arr)
		w[name+".weight"] = wq
		w[name+".scales"] = s
		w[name+".biases"] = b
		metal.Free(arr)
	}
	for idx := 0; idx < qc.layers; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		w[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx)*0.001, H)
		w[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx)*0.001, H)
		w[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx)*0.001, H)
		w[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx)*0.001, H)
		w[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx)*0.001, HD)
		quant(prefix+".self_attn.q_proj", seqArray(0.001*float32(idx+1), HD, H))
		// shared (consumer) layers omit k/v — modelled by numShared from the tail.
		isShared := qc.numShared > 0 && idx >= qc.layers-qc.numShared
		if !isShared {
			w[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx)*0.001, HD)
			quant(prefix+".self_attn.k_proj", seqArray(0.002*float32(idx+1), HD, H))
			if !qc.kEqV {
				quant(prefix+".self_attn.v_proj", seqArray(0.003*float32(idx+1), HD, H))
			}
		}
		quant(prefix+".self_attn.o_proj", seqArray(0.004*float32(idx+1), H, HD))
		quant(prefix+".mlp.gate_proj", seqArray(0.005*float32(idx+1), I, H))
		quant(prefix+".mlp.up_proj", seqArray(0.006*float32(idx+1), I, H))
		quant(prefix+".mlp.down_proj", seqArray(0.007*float32(idx+1), H, I))
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

// cov4clPrimeFixed returns a FixedKVCache whose storage is allocated and offset
// advanced to fill, by replaying prefill tokens through Update.
func cov4clPrimeFixed(t *testing.T, maxSize, headDim, fill int) *metal.FixedKVCache {
	t.Helper()
	cache := metal.NewFixedKVCache(maxSize)
	for i := 0; i < fill; i++ {
		k := seqArray(0.1+0.01*float32(i), 1, 1, 1, headDim)
		v := seqArray(0.2+0.01*float32(i), 1, 1, 1, headDim)
		gotK, gotV := cache.Update(k, v, 1)
		if err := metal.Eval(gotK, gotV); err != nil {
			t.Fatalf("prime Update eval: %v", err)
		}
		metal.Free(k, v)
	}
	return cache
}

// TestCompiledLayer_OwnerPreCapDecode_Good drives the compiled closure through
// the owner pre-capacity regime (offset+L within the band): the projections are
// quantised so the layer is trace-eligible, and the hit counter proves the
// closure served the step rather than declining to the uncompiled path.
func TestCompiledLayer_OwnerPreCapDecode_Good(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[1] // full attention owner
	if layer.LayerType != "full_attention" {
		t.Fatalf("layer 1 type = %s, want full_attention", layer.LayerType)
	}
	cache := cov4clPrimeFixed(t, 16, 32, 1)

	before := CompiledLayerDecodeHits()
	for step := 0; step < 4; step++ {
		x := seqArray(0.01*float32(step+1), 1, 1, 32)
		out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
		if !ok {
			t.Fatalf("step %d declined the compiled path", step)
		}
		if err := metal.Eval(out); err != nil {
			t.Fatalf("step %d eval: %v", step, err)
		}
		shape := out.Shape()
		if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 32 {
			t.Fatalf("step %d hidden shape = %v, want [1 1 32]", step, shape)
		}
		metal.Free(out, x)
	}
	if got := CompiledLayerDecodeHits() - before; got != 4 {
		t.Fatalf("compiled hits = %d, want 4 (closure must serve every step)", got)
	}
}

// TestCompiledLayer_OwnerPostCapDecode_Good drives the sliding owner past
// capacity so the regime selector takes the post-cap rotate-and-write branch
// (NativeFixedSlidingSingleTokenAttention), the other half of the owner closure.
func TestCompiledLayer_OwnerPostCapDecode_Good(t *testing.T) {
	if !metal.NativeFixedSlidingAttentionEnabled() {
		t.Cleanup(metal.SetRuntimeGate(metal.GateNativeFixedSlidingAttention, true))
	}
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[0] // sliding owner
	if layer.LayerType != "sliding_attention" {
		t.Fatalf("layer 0 type = %s, want sliding_attention", layer.LayerType)
	}
	// fill the cache to capacity so the next step is post-cap.
	cache := cov4clPrimeFixed(t, 4, 32, 4)
	if cache.Len() < cache.MaxSize() {
		t.Fatalf("cache len %d < maxSize %d; expected full for post-cap", cache.Len(), cache.MaxSize())
	}

	before := CompiledLayerDecodeHits()
	hit := false
	for step := 0; step < 3; step++ {
		x := seqArray(0.01*float32(step+1), 1, 1, 32)
		out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
		if ok {
			hit = true
			if err := metal.Eval(out); err != nil {
				t.Fatalf("step %d eval: %v", step, err)
			}
			metal.Free(out)
		}
		metal.Free(x)
	}
	if !hit || CompiledLayerDecodeHits() == before {
		t.Fatal("post-cap sliding step never served the compiled closure")
	}
}

// TestCompiledLayer_KEqVDecode_Good covers the k=v projection-sharing arm of the
// closure (v cloned from k, no v_proj input) on a quantised owner layer.
func TestCompiledLayer_KEqVDecode_Good(t *testing.T) {
	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2, kEqV: true,
	})
	layer := m.Layers[1]
	cache := cov4clPrimeFixed(t, 16, 32, 1)

	before := CompiledLayerDecodeHits()
	x := seqArray(0.02, 1, 1, 32)
	out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	if !ok {
		t.Fatal("k=v owner declined the compiled path")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval: %v", err)
	}
	metal.Free(out, x)
	if CompiledLayerDecodeHits() == before {
		t.Fatal("k=v compiled closure did not serve")
	}
}

// TestCompiledLayer_DeclineLadder_Bad walks the cheap decline branches of
// compiledDecodeForward — gate off, non-decode shape, an explicit mask, a nil
// input, and a non-Fixed cache — each returning ok=false without panicking. The
// quantised model is only needed for the cache-type decline; the rest decline
// before any weight work.
func TestCompiledLayer_DeclineLadder_Bad(t *testing.T) {
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[1]
	cfg := m.Cfg

	// gate off: declines immediately regardless of inputs.
	x := seqArray(0.01, 1, 1, 32)
	if _, _, ok := layer.compiledDecodeForward(x, metal.NewKVCache(), 1, 1, nil, nil, sharedKV{}, cfg); ok {
		t.Fatal("gate off must decline")
	}
	metal.Free(x)

	t.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))

	// batch > 1 declines (not decode-shaped).
	xB := seqArray(0.01, 2, 1, 32)
	if _, _, ok := layer.compiledDecodeForward(xB, metal.NewFixedKVCache(8), 2, 1, nil, nil, sharedKV{}, cfg); ok {
		t.Fatal("B=2 must decline")
	}
	metal.Free(xB)

	// L beyond the compiled max declines.
	xL := seqArray(0.01, 1, 9, 32)
	if _, _, ok := layer.compiledDecodeForward(xL, metal.NewFixedKVCache(16), 1, 9, nil, nil, sharedKV{}, cfg); ok {
		t.Fatal("L=9 must decline")
	}
	metal.Free(xL)

	// an explicit mask declines.
	xm := seqArray(0.01, 1, 1, 32)
	mask := seqArray(0, 1, 1, 1, 1)
	if _, _, ok := layer.compiledDecodeForward(xm, metal.NewFixedKVCache(8), 1, 1, mask, nil, sharedKV{}, cfg); ok {
		t.Fatal("explicit mask must decline")
	}
	metal.Free(xm, mask)

	// nil cfg declines via the invalid-input branch.
	xc := seqArray(0.01, 1, 1, 32)
	if _, _, ok := layer.compiledDecodeForward(xc, metal.NewFixedKVCache(8), 1, 1, nil, nil, sharedKV{}, nil); ok {
		t.Fatal("nil cfg must decline")
	}
	metal.Free(xc)

	// a non-Fixed cache declines with the "cache is %T" reason.
	xn := seqArray(0.01, 1, 1, 32)
	if _, _, ok := layer.compiledDecodeForward(xn, metal.NewKVCache(), 1, 1, nil, nil, sharedKV{}, cfg); ok {
		t.Fatal("non-Fixed cache must decline")
	}
	metal.Free(xn)
}

// TestCompiledLayer_OutputsValid_Good covers gemma4CompiledLayerOutputsValid:
// matching geometry passes, a nil entry / rank / dim / dtype mismatch fails.
func TestCompiledLayer_OutputsValid_Good(t *testing.T) {
	requireMetalRuntime(t)
	cacheK := seqArray(0.1, 1, 1, 4, 8)
	cacheV := seqArray(0.2, 1, 1, 4, 8)
	defer metal.Free(cacheK, cacheV)

	goodH := seqArray(0.3, 1, 1, 8)
	goodK := seqArray(0.1, 1, 1, 4, 8)
	goodV := seqArray(0.2, 1, 1, 4, 8)
	defer metal.Free(goodH, goodK, goodV)
	if !gemma4CompiledLayerOutputsValid([]*metal.Array{goodH, goodK, goodV}, cacheK, cacheV) {
		t.Fatal("matching geometry should be valid")
	}

	// nil entry fails.
	if gemma4CompiledLayerOutputsValid([]*metal.Array{goodH, nil, goodV}, cacheK, cacheV) {
		t.Fatal("nil output entry should be invalid")
	}

	// wrong rank fails.
	badRank := seqArray(0.1, 1, 4, 8)
	defer metal.Free(badRank)
	if gemma4CompiledLayerOutputsValid([]*metal.Array{goodH, badRank, goodV}, cacheK, cacheV) {
		t.Fatal("rank mismatch should be invalid")
	}

	// wrong dim fails.
	badDim := seqArray(0.1, 1, 2, 4, 8)
	defer metal.Free(badDim)
	if gemma4CompiledLayerOutputsValid([]*metal.Array{goodH, badDim, goodV}, cacheK, cacheV) {
		t.Fatal("dim mismatch should be invalid")
	}

	// wrong dtype fails.
	badType := metal.AsType(goodK, metal.DTypeFloat16)
	defer metal.Free(badType)
	if gemma4CompiledLayerOutputsValid([]*metal.Array{goodH, badType, goodV}, cacheK, cacheV) {
		t.Fatal("dtype mismatch should be invalid")
	}
}
