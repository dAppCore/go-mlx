// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// ── parseGptOssConfig (pure Go, no Metal) ──────────────────────────────────

func TestGptOss_ParseGptOssConfig_Good(t *testing.T) {
	cfg, err := parseGptOssConfig([]byte(`{
		"model_type": "gpt_oss",
		"hidden_size": 1024,
		"num_hidden_layers": 24,
		"num_attention_heads": 16,
		"num_key_value_heads": 4,
		"head_dim": 64,
		"vocab_size": 201088,
		"num_local_experts": 32,
		"num_experts_per_tok": 4,
		"rms_norm_eps": 1e-6,
		"rope_theta": 500000,
		"decoder_sparse_step": 2
	}`))
	if err != nil {
		t.Fatalf("parseGptOssConfig: %v", err)
	}
	if cfg.HiddenSize != 1024 || cfg.NumHiddenLayers != 24 || cfg.NumAttentionHeads != 16 {
		t.Fatalf("cfg = %+v, want hidden=1024 layers=24 heads=16", cfg)
	}
	if cfg.HeadDim != 64 {
		t.Fatalf("HeadDim = %d, want explicit 64 (not derived)", cfg.HeadDim)
	}
	if cfg.RopeTheta != 500000 {
		t.Fatalf("RopeTheta = %v, want explicit 500000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Fatalf("RMSNormEps = %v, want explicit 1e-6", cfg.RMSNormEps)
	}
	if cfg.Scale != 1.0 {
		t.Fatalf("Scale = %v, want 1.0 once HeadDim > 0", cfg.Scale)
	}
	if cfg.expertCount() != 32 {
		t.Fatalf("expertCount() = %d, want 32", cfg.expertCount())
	}
	if cfg.topK() != 4 {
		t.Fatalf("topK() = %d, want 4", cfg.topK())
	}
}

func TestGptOss_ParseGptOssConfig_Bad(t *testing.T) {
	if _, err := parseGptOssConfig([]byte("{not valid json")); err == nil {
		t.Fatal("parseGptOssConfig(broken JSON) error = nil, want parse error")
	}
}

// TestGptOss_ParseGptOssConfig_Ugly exercises the derived-default branches:
// head_dim from hidden/heads, scale-on-headdim, and the rope_theta / rms_norm_eps
// fallbacks (gptoss.go lines 128-139). Also asserts the expertCount/topK fall to
// their literals when the config omits both expert fields.
func TestGptOss_ParseGptOssConfig_Ugly(t *testing.T) {
	cfg, err := parseGptOssConfig([]byte(`{
		"model_type": "gpt_oss",
		"hidden_size": 512,
		"num_attention_heads": 8
	}`))
	if err != nil {
		t.Fatalf("parseGptOssConfig: %v", err)
	}
	if cfg.HeadDim != 64 {
		t.Fatalf("HeadDim = %d, want derived 512/8 = 64", cfg.HeadDim)
	}
	if cfg.Scale != 1.0 {
		t.Fatalf("Scale = %v, want 1.0 once HeadDim derived > 0", cfg.Scale)
	}
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("RopeTheta = %v, want default 1000000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-5 {
		t.Fatalf("RMSNormEps = %v, want default 1e-5", cfg.RMSNormEps)
	}
	if cfg.expertCount() != 8 {
		t.Fatalf("expertCount() = %d, want literal 8 when experts unset", cfg.expertCount())
	}
	if cfg.topK() != 2 {
		t.Fatalf("topK() = %d, want literal 2 when num_experts_per_tok unset", cfg.topK())
	}
}

// TestGptOss_ParseGptOssConfig_NestedQuantization confirms the nested
// quantization_config wrapper is picked up by FirstQuantization.
func TestGptOss_ParseGptOssConfig_NestedQuantization(t *testing.T) {
	cfg, err := parseGptOssConfig([]byte(`{
		"model_type": "gpt_oss",
		"hidden_size": 64,
		"num_attention_heads": 8,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`))
	if err != nil {
		t.Fatalf("parseGptOssConfig: %v", err)
	}
	if cfg.Quantization == nil {
		t.Fatal("Quantization = nil, want populated from quantization_config")
	}
	if cfg.Quantization.Bits != 4 || cfg.Quantization.GroupSize != 64 {
		t.Fatalf("Quantization = %+v, want bits=4 group=64", cfg.Quantization)
	}
}

// TestGptOss_ParseGptOssConfig_NestedQuantMismatch_Bad drives the second
// (wrapper) unmarshal error arm (gptoss.go lines 123-125): the top-level JSON is
// valid for the flat GptOssConfig (which has no quantization field, so the value
// is tolerated there) but quantization_config / quantization typed as a
// non-object fails the nested *metal.QuantizationConfig unmarshal — the
// "parse nested config" error. Mirrors the mixtral precedent.
func TestGptOss_ParseGptOssConfig_NestedQuantMismatch_Bad(t *testing.T) {
	for _, bad := range []string{
		`{"model_type":"gpt_oss","hidden_size":8,"num_attention_heads":2,"quantization_config":"notanobject"}`,
		`{"model_type":"gpt_oss","hidden_size":8,"num_attention_heads":2,"quantization":[1,2]}`,
	} {
		if _, err := parseGptOssConfig([]byte(bad)); err == nil {
			t.Errorf("expected nested-config parse error for %s", bad)
		}
	}
}

// ── gptOssToQwen3Config (pure Go, no Metal) ────────────────────────────────

func TestGptOss_GptOssToQwen3Config_Good(t *testing.T) {
	cfg := &GptOssConfig{
		HiddenSize:            1024,
		NumHiddenLayers:       24,
		NumAttentionHeads:     16,
		NumKeyValueHeads:      4,
		HeadDim:               64,
		VocabSize:             201088,
		RMSNormEps:            1e-6,
		MaxPositionEmbeddings: 131072,
		RopeTheta:             500000,
		Scale:                 1.0,
	}
	dc := gptOssToQwen3Config(cfg)
	if dc == nil {
		t.Fatal("gptOssToQwen3Config returned nil for a valid config")
	}
	if dc.HiddenSize != 1024 || dc.NumHiddenLayers != 24 || dc.NumAttentionHeads != 16 {
		t.Fatalf("DenseConfig = %+v, want hidden=1024 layers=24 heads=16", dc)
	}
	if dc.NumKeyValueHeads != 4 || dc.HeadDim != 64 || dc.VocabSize != 201088 {
		t.Fatalf("DenseConfig = %+v, want kv=4 headdim=64 vocab=201088", dc)
	}
	if dc.RopeTheta != 500000 || dc.Scale != 1.0 || dc.MaxPositionEmbeddings != 131072 {
		t.Fatalf("DenseConfig = %+v, want rope=500000 scale=1 maxpos=131072", dc)
	}
}

func TestGptOss_GptOssToQwen3Config_Ugly(t *testing.T) {
	if dc := gptOssToQwen3Config(nil); dc != nil {
		t.Fatalf("gptOssToQwen3Config(nil) = %+v, want nil", dc)
	}
}

func TestGptOss_LoadGptOssMissingWeights_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 201088,
		"num_local_experts": 32
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := LoadGptOss(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for gpt_oss without safetensors")
	}
	if !core.Contains(err.Error(), "gpt_oss") {
		t.Fatalf("error = %v, should contain gpt_oss", err)
	}
}

// ── LoadGptOss full load + the loaded-model fixture ────────────────────────

// gptOssMixedConfigJSON describes a synthetic 2-layer GPT-OSS model where
// decoder_sparse_step=2 makes layer 0 DENSE (SiLU MLP) and layer 1 MoE
// (router + experts). Loading it covers both decoder branches and the whole
// LoadGptOss assembly loop in one shot.
const gptOssMixedConfigJSON = `{
	"architectures": ["GptOssForCausalLM"],
	"model_type": "gpt_oss",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 2,
	"num_key_value_heads": 2,
	"head_dim": 4,
	"vocab_size": 5,
	"max_position_embeddings": 32,
	"rms_norm_eps": 1e-6,
	"rope_theta": 1000000,
	"decoder_sparse_step": 2,
	"num_local_experts": 2,
	"num_experts_per_tok": 2
}`

// writeMixedGptOssModel writes config.json, tokenizer.json and a safetensors
// file for the gptOssMixedConfigJSON geometry: layer 0 dense MLP weights,
// layer 1 router + 2 expert weight triples, plus shared embed/norm/lm_head.
func writeMixedGptOssModel(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gptOssMixedConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	const (
		h     = 8
		inter = 16
		v     = 5
		hd    = 4
		nh    = 2 // heads
		kvh   = 2 // kv heads
	)
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight": seqArr(0.01, v, h),
		"model.norm.weight":         seqArr(0.11, h),
		"lm_head.weight":            seqArr(0.12, v, h),
	}
	// Both layers carry attention + the two layernorms.
	for _, l := range []int{0, 1} {
		p := core.Sprintf("model.layers.%d", l)
		weights[p+".input_layernorm.weight"] = seqArr(0.02, h)
		weights[p+".post_attention_layernorm.weight"] = seqArr(0.03, h)
		weights[p+".self_attn.q_proj.weight"] = seqArr(0.04, nh*hd, h)
		weights[p+".self_attn.k_proj.weight"] = seqArr(0.05, kvh*hd, h)
		weights[p+".self_attn.v_proj.weight"] = seqArr(0.06, kvh*hd, h)
		weights[p+".self_attn.o_proj.weight"] = seqArr(0.07, h, nh*hd)
	}
	// Layer 0 → dense SiLU MLP.
	weights["model.layers.0.mlp.gate_proj.weight"] = seqArr(0.08, inter, h)
	weights["model.layers.0.mlp.up_proj.weight"] = seqArr(0.09, inter, h)
	weights["model.layers.0.mlp.down_proj.weight"] = seqArr(0.10, h, inter)
	// Layer 1 → MoE router + 2 experts.
	weights["model.layers.1.mlp.gate.weight"] = seqArr(0.20, 2, h)
	for e := range 2 {
		p := core.Sprintf("model.layers.1.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArr(0.30+float32(e)*0.03, inter, h)
		weights[p+".up_proj.weight"] = seqArr(0.31+float32(e)*0.03, inter, h)
		weights[p+".down_proj.weight"] = seqArr(0.32+float32(e)*0.03, h, inter)
	}
	defer freeGptOssArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
}

// loadMixedGptOss loads the synthetic mixed dense+MoE model and registers its
// cleanup; the returned model is the shared fixture for the method tests.
func loadMixedGptOss(t *testing.T) *GptOssModel {
	t.Helper()
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedGptOssModel(t, dir)
	model, err := LoadGptOss(dir)
	if err != nil {
		t.Fatalf("LoadGptOss(mixed dense+MoE) error = %v", err)
	}
	t.Cleanup(model.CloseModel)
	return model
}

func TestGptOss_LoadGptOss_Good(t *testing.T) {
	model := loadMixedGptOss(t)

	if model.ModelType() != "gpt_oss" {
		t.Fatalf("ModelType() = %q, want gpt_oss", model.ModelType())
	}
	if model.NumLayers() != 2 {
		t.Fatalf("NumLayers() = %d, want 2", model.NumLayers())
	}
	if model.Cfg.VocabSize != 5 || model.Cfg.HiddenSize != 8 {
		t.Fatalf("Cfg = %+v, want vocab=5 hidden=8", model.Cfg)
	}
	if model.EmbedTokens == nil || model.EmbedTokens.Weight == nil {
		t.Fatal("EmbedTokens.Weight = nil, want loaded embedding")
	}
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("Output.Weight = nil, want loaded lm_head")
	}
	// decoder_sparse_step=2 → layer 0 dense (SiLU MLP), layer 1 MoE.
	if model.Layers[0].isMoELayer() {
		t.Fatal("layer 0 isMoELayer() = true, want dense under sparse_step=2")
	}
	if model.Layers[0].Dense.MLP == nil {
		t.Fatal("layer 0 Dense.MLP = nil, want SiLU MLP loaded for the dense branch")
	}
	if !model.Layers[1].isMoELayer() {
		t.Fatal("layer 1 isMoELayer() = false, want MoE under sparse_step=2")
	}
	if n := len(model.Layers[1].MoE.Experts); n != 2 {
		t.Fatalf("layer 1 experts = %d, want 2", n)
	}
}

// TestGptOss_LoadGptOss_DerivesVocabFromEmbedding_Good omits vocab_size from the
// config so LoadGptOss must derive it from the embed tensor's first dim
// (gptoss.go lines 180-186).
func TestGptOss_LoadGptOss_DerivesVocabFromEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gpt_oss",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"decoder_sparse_step": 1,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`)
	writeMinimalTokenizer(t, dir)

	const (
		h     = 8
		inter = 16
		v     = 7 // derived from embed tensor first dim
		hd    = 4
	)
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight":                      seqArr(0.01, v, h),
		"model.norm.weight":                              seqArr(0.11, h),
		"lm_head.weight":                                 seqArr(0.12, v, h),
		"model.layers.0.input_layernorm.weight":          seqArr(0.02, h),
		"model.layers.0.post_attention_layernorm.weight": seqArr(0.03, h),
		"model.layers.0.self_attn.q_proj.weight":         seqArr(0.04, h, h),
		"model.layers.0.self_attn.k_proj.weight":         seqArr(0.05, h, h),
		"model.layers.0.self_attn.v_proj.weight":         seqArr(0.06, h, h),
		"model.layers.0.self_attn.o_proj.weight":         seqArr(0.07, h, h),
		"model.layers.0.mlp.gate.weight":                 seqArr(0.20, 2, h),
	}
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArr(0.30+float32(e)*0.03, inter, h)
		weights[p+".up_proj.weight"] = seqArr(0.31+float32(e)*0.03, inter, h)
		weights[p+".down_proj.weight"] = seqArr(0.32+float32(e)*0.03, h, inter)
	}
	defer freeGptOssArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGptOss(dir)
	if err != nil {
		t.Fatalf("LoadGptOss error = %v", err)
	}
	defer model.CloseModel()
	if model.Cfg.VocabSize != v {
		t.Fatalf("derived VocabSize = %d, want %d from embed tensor", model.Cfg.VocabSize, v)
	}
}

func TestGptOss_LoadGptOss_MissingConfig_Bad(t *testing.T) {
	_, err := LoadGptOss(t.TempDir())
	if err == nil {
		t.Fatal("LoadGptOss(no config.json) error = nil, want load-config error")
	}
	if !core.Contains(err.Error(), "gpt_oss") {
		t.Fatalf("error = %v, want gpt_oss prefix", err)
	}
}

func TestGptOss_LoadGptOss_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "{broken")
	_, err := LoadGptOss(dir)
	if err == nil {
		t.Fatal("LoadGptOss(broken config) error = nil, want parse error")
	}
}

func TestGptOss_LoadGptOss_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), gptOssMixedConfigJSON)
	_, err := LoadGptOss(dir)
	if err == nil {
		t.Fatal("LoadGptOss(no tokenizer) error = nil, want tokenizer error")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want tokenizer mention", err)
	}
}

// ── LoadGptOss quantized + tied-embedding load paths (synthetic, LOAD-ONLY) ─

// TestGptOss_LoadGptOss_Quantized_Good loads a synthetic 4-bit checkpoint:
// metal.Quantize packs each attention + dense-MLP projection, the MoE router,
// and the embedding/lm_head into the (weight, scales, biases) triplet the
// loader's quantized `linear` closure + gptOssLoadRouter resolve — driving the
// q != nil branches (gptoss.go 170-178, 188-194, 242-249, 299-304) that the
// bf16 mixed fixture skips.
//
// Experts stay PLAIN dense arrays on purpose: gptOssLoadExpert (gptoss.go
// 312-327) has no quant arm and would reinterpret packed bytes as bf16. For the
// same reason this test is load-only — it never runs Forward.
func TestGptOss_LoadGptOss_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	const (
		h     = int32(64)
		inter = int32(128)
		v     = int32(64)
		hd    = int32(32)
		nh    = int32(2) // nh*hd == h == 64 → o_proj last dim divides group_size
		gs    = 64       // MLX group sizes 32/64/128; gs must divide the quantized dim
		bits  = 4
	)
	dir := t.TempDir()
	// Two layers, decoder_sparse_step=2 → layer 0 dense, layer 1 MoE.
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"vocab_size": 64,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"decoder_sparse_step": 2,
		"num_local_experts": 2,
		"num_experts_per_tok": 2,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	weights := map[string]*metal.Array{}
	// addQuant packs a dense [out,in] weight into its quantized triplet under the
	// base name (.weight/.scales/.biases) the loader resolves.
	addQuant := func(name string, out, in int32) {
		dense := seqArr(0.02, int(out), int(in))
		wq, sc, bi, err := metal.Quantize(dense, gs, bits, "")
		if err != nil {
			t.Fatalf("Quantize(%s): %v", name, err)
		}
		metal.Free(dense)
		weights[name+".weight"] = wq
		weights[name+".scales"] = sc
		weights[name+".biases"] = bi
	}
	addQuant("model.embed_tokens", v, h)
	addQuant("lm_head", v, h)
	for _, l := range []int{0, 1} {
		p := core.Sprintf("model.layers.%d", l)
		weights[p+".input_layernorm.weight"] = seqArr(0.02, int(h))
		weights[p+".post_attention_layernorm.weight"] = seqArr(0.03, int(h))
		addQuant(p+".self_attn.q_proj", nh*hd, h)
		addQuant(p+".self_attn.k_proj", nh*hd, h)
		addQuant(p+".self_attn.v_proj", nh*hd, h)
		addQuant(p+".self_attn.o_proj", h, nh*hd)
	}
	weights["model.norm.weight"] = seqArr(0.11, int(h))
	// Layer 0 → quantized dense SiLU MLP.
	addQuant("model.layers.0.mlp.gate_proj", inter, h)
	addQuant("model.layers.0.mlp.up_proj", inter, h)
	addQuant("model.layers.0.mlp.down_proj", h, inter)
	// Layer 1 → quantized MoE router; experts stay plain dense (no quant arm).
	addQuant("model.layers.1.mlp.gate", 2, h)
	for e := range 2 {
		p := core.Sprintf("model.layers.1.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArr(0.30+float32(e)*0.03, int(inter), int(h))
		weights[p+".up_proj.weight"] = seqArr(0.31+float32(e)*0.03, int(inter), int(h))
		weights[p+".down_proj.weight"] = seqArr(0.32+float32(e)*0.03, int(h), int(inter))
	}
	defer freeGptOssArrayMap(weights)

	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGptOss(dir)
	if err != nil {
		t.Fatalf("LoadGptOss(quantized) error = %v", err)
	}
	defer model.CloseModel()

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != bits || info.QuantGroup != gs {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, bits, gs)
	}
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved (gptoss.go 188-194)")
	}
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("quantized lm_head Output not resolved (gptoss.go 242-249)")
	}
	// Distinct lm_head Output carries scales — not the tied embedding.
	if model.Output.Scales == nil {
		t.Error("quantized lm_head Output.Scales = nil, want resolved scales (gptoss.go 242-249)")
	}
	if model.Layers[1].MoE == nil || model.Layers[1].MoE.Router.Scales == nil {
		t.Error("quantized MoE router scales were not resolved (gptoss.go 299-304)")
	}
}

// TestGptOss_LoadGptOss_TiedEmbedding_Good omits lm_head entirely so LoadGptOss
// ties the output projection to the embedding via AsLinear (gptoss.go line 254)
// — the weights-untied default for this synthetic geometry.
func TestGptOss_LoadGptOss_TiedEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gpt_oss",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"decoder_sparse_step": 1,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`)
	writeMinimalTokenizer(t, dir)

	const (
		h     = 8
		inter = 16
		v     = 5
		hd    = 4
		nh    = 2
	)
	// No lm_head.weight → Output must tie to EmbedTokens.
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight":                      seqArr(0.01, v, h),
		"model.norm.weight":                              seqArr(0.11, h),
		"model.layers.0.input_layernorm.weight":          seqArr(0.02, h),
		"model.layers.0.post_attention_layernorm.weight": seqArr(0.03, h),
		"model.layers.0.self_attn.q_proj.weight":         seqArr(0.04, nh*hd, h),
		"model.layers.0.self_attn.k_proj.weight":         seqArr(0.05, nh*hd, h),
		"model.layers.0.self_attn.v_proj.weight":         seqArr(0.06, nh*hd, h),
		"model.layers.0.self_attn.o_proj.weight":         seqArr(0.07, h, nh*hd),
		"model.layers.0.mlp.gate.weight":                 seqArr(0.20, 2, h),
	}
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArr(0.30+float32(e)*0.03, inter, h)
		weights[p+".up_proj.weight"] = seqArr(0.31+float32(e)*0.03, inter, h)
		weights[p+".down_proj.weight"] = seqArr(0.32+float32(e)*0.03, h, inter)
	}
	defer freeGptOssArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGptOss(dir)
	if err != nil {
		t.Fatalf("LoadGptOss(tied embedding) error = %v", err)
	}
	defer model.CloseModel()
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("tied Output.Weight = nil, want the embedding weight via AsLinear")
	}
	if model.Output.Weight != model.EmbedTokens.Weight {
		t.Fatal("Output weight not tied to EmbedTokens weight (AsLinear, gptoss.go line 254)")
	}
}

// ── Forward / ForwardMasked over the loaded fixture ────────────────────────

// gptOssForwardTokens builds a [B=1, L] int32 token input within the synthetic
// vocab (size 5) and materialises it.
func gptOssForwardTokens(l int) *metal.Array {
	ids := make([]int32, l)
	for i := range ids {
		ids[i] = int32(i % 5)
	}
	tokens := metal.FromValues(ids, 1, l)
	metal.Materialize(tokens)
	return tokens
}

func TestGptOss_Forward_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	tokens := gptOssForwardTokens(3)
	defer metal.Free(tokens)

	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if logits == nil {
		t.Fatal("Forward returned nil logits")
	}
	defer metal.Free(logits)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}
	shape := logits.Shape()
	// [B=1, L=3, vocab=5]
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 5 {
		t.Fatalf("logits shape = %v, want [1 3 5]", shape)
	}
}

func TestGptOss_ForwardMasked_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	tokens := gptOssForwardTokens(1)
	defer metal.Free(tokens)

	caches := model.NewCache()
	// nil mask is the single-token decode path (the common case).
	logits := model.ForwardMasked(tokens, nil, caches)
	if logits == nil {
		t.Fatal("ForwardMasked returned nil logits")
	}
	defer metal.Free(logits)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}
	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 5 {
		t.Fatalf("logits shape = %v, want [1 1 5]", shape)
	}
}

// ── NewCache over the loaded fixture ───────────────────────────────────────

func TestGptOss_NewCache_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	caches := model.NewCache()
	if len(caches) != model.NumLayers() {
		t.Fatalf("NewCache() len = %d, want %d (one per layer)", len(caches), model.NumLayers())
	}
	for i, c := range caches {
		if c == nil {
			t.Fatalf("cache[%d] = nil, want a KV cache", i)
		}
	}
}

// ── expertCount / topK fallbacks + gptOssSwitchExperts edge (in-package) ───

// TestGptOss_ExpertCount_NumExpertsFallback covers the second expertCount
// branch: num_experts populated, num_local_experts absent (gptoss.go 67-69).
func TestGptOss_ExpertCount_NumExpertsFallback(t *testing.T) {
	cfg := &GptOssConfig{NumExperts: 16}
	if got := cfg.expertCount(); got != 16 {
		t.Fatalf("expertCount() = %d, want 16 from num_experts fallback", got)
	}
}

// TestGptOss_SwitchExperts_NilExpert_Ugly drives the nil-expert early return in
// gptOssSwitchExperts (gptoss.go lines 334-336): a nil entry yields (nil,false).
func TestGptOss_SwitchExperts_NilExpert_Ugly(t *testing.T) {
	experts := []*GptOssExpert{nil}
	se, ok := gptOssSwitchExperts(experts)
	if ok {
		t.Fatal("gptOssSwitchExperts(nil expert) ok = true, want false")
	}
	if se != nil {
		t.Fatal("gptOssSwitchExperts(nil expert) experts != nil, want nil")
	}
}

// TestGptOss_GptOssLoadRouter_NotFound_Ugly drives the no-router-weight fallback
// in gptOssLoadRouter (gptoss.go line 309): when no prefix/suffix combination
// resolves a gate weight, an empty *metal.MoERouter is returned rather than nil.
func TestGptOss_GptOssLoadRouter_NotFound_Ugly(t *testing.T) {
	router := gptOssLoadRouter(map[string]*metal.Array{}, 0, nil)
	if router == nil {
		t.Fatal("gptOssLoadRouter(no weights) = nil, want an empty &metal.MoERouter{}")
	}
	if router.Weight != nil {
		t.Fatalf("gptOssLoadRouter(no weights).Weight = %v, want nil", router.Weight)
	}
}

// TestGptOss_GptOssLoadExpert_NotFound_Ugly drives the no-expert-weight fallback
// in gptOssLoadExpert (gptoss.go line 326): when neither the mlp nor moe expert
// prefix resolves a gate weight, an empty &GptOssExpert{} is returned with all
// projections nil.
func TestGptOss_GptOssLoadExpert_NotFound_Ugly(t *testing.T) {
	expert := gptOssLoadExpert(func(string) *metal.Array { return nil }, 0, 0)
	if expert == nil {
		t.Fatal("gptOssLoadExpert(no weights) = nil, want an empty &GptOssExpert{}")
	}
	if expert.GateProj != nil || expert.UpProj != nil || expert.DownProj != nil {
		t.Fatalf("gptOssLoadExpert(no weights) = %+v, want all-nil projections", expert)
	}
}

// ── ApplyLoRA ──────────────────────────────────────────────────────────────

func TestGptOss_ApplyLoRA_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj", "v_proj"},
	})
	if adapter == nil {
		t.Fatal("ApplyLoRA returned nil adapter")
	}
	if adapter.Model == nil {
		t.Fatal("adapter.Model = nil, want the source model")
	}
	// q_proj + v_proj across both layers = 4 attention adapters.
	if len(adapter.Layers) != 4 {
		t.Fatalf("adapter.Layers = %d, want 4 (q_proj+v_proj over 2 layers)", len(adapter.Layers))
	}
	if _, ok := adapter.Layers["model.layers.0.self_attn.q_proj"]; !ok {
		t.Fatal("adapter missing model.layers.0.self_attn.q_proj key")
	}
	// LoRA must be wired onto the underlying projection.
	if model.Layers[0].Dense.Attention.QProj.LoRA == nil {
		t.Fatal("layer 0 QProj.LoRA = nil, want LoRA wired in")
	}
}

// TestGptOss_ApplyLoRA_AllAttentionTargets exercises every attention switch arm
// (q/k/v/o_proj) so each projection-selection branch is covered
// (gptoss.go lines 431-438).
func TestGptOss_ApplyLoRA_AllAttentionTargets(t *testing.T) {
	model := loadMixedGptOss(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	})
	// 4 attention projections over 2 layers = 8 adapters.
	if len(adapter.Layers) != 8 {
		t.Fatalf("adapter.Layers = %d, want 8 (q/k/v/o over 2 layers)", len(adapter.Layers))
	}
	attn := model.Layers[0].Dense.Attention
	for name, proj := range map[string]*metal.Linear{
		"q_proj": attn.QProj, "k_proj": attn.KProj, "v_proj": attn.VProj, "o_proj": attn.OProj,
	} {
		if proj.LoRA == nil {
			t.Fatalf("layer 0 %s.LoRA = nil, want LoRA wired in", name)
		}
	}
}

// TestGptOss_ApplyLoRA_DenseMLPTargets confirms gate/up/down LoRA targets land
// only on the dense layer (layer 0); the MoE layer has no Dense.MLP so it is
// skipped (gptoss.go lines 439-451).
func TestGptOss_ApplyLoRA_DenseMLPTargets(t *testing.T) {
	model := loadMixedGptOss(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"gate_proj", "up_proj", "down_proj"},
	})
	// Only layer 0 (dense) has a SiLU MLP → one adapter per gate/up/down = 3.
	if len(adapter.Layers) != 3 {
		t.Fatalf("adapter.Layers = %d, want 3 (gate/up/down on the single dense layer)", len(adapter.Layers))
	}
	for _, key := range []string{
		"model.layers.0.mlp.gate_proj",
		"model.layers.0.mlp.up_proj",
		"model.layers.0.mlp.down_proj",
	} {
		if _, ok := adapter.Layers[key]; !ok {
			t.Fatalf("adapter missing %s key", key)
		}
	}
	mlp := model.Layers[0].Dense.MLP
	if mlp.GateProj.LoRA == nil || mlp.UpProj.LoRA == nil || mlp.DownProj.LoRA == nil {
		t.Fatal("dense MLP gate/up/down LoRA not all wired in")
	}
}

func TestGptOss_ApplyLoRA_EmptyModel_Ugly(t *testing.T) {
	m := &GptOssModel{}
	adapter := m.ApplyLoRA(metal.LoRAConfig{Rank: 2, Alpha: 4, TargetLayers: []string{"q_proj"}})
	if adapter == nil {
		t.Fatal("ApplyLoRA(empty model) = nil, want a usable empty adapter")
	}
	if len(adapter.Layers) != 0 {
		t.Fatalf("adapter.Layers = %d, want 0 for a model with no layers", len(adapter.Layers))
	}
}

// seqArr builds a deterministic [shape] float32 tensor (the qwen3 seqArray
// recipe) for the synthetic on-disk weights.
func seqArr(start float32, shape ...int) *metal.Array {
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

func freeGptOssArrayMap(arrays map[string]*metal.Array) {
	for _, a := range arrays {
		metal.Free(a)
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
