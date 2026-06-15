// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// --- LoadKimi error paths ---

func TestModel_LoadKimi_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000
	}`)

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadKimi_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadKimi_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalKimiConfig(t, dir)
	writeMinimalKimiTokenizer(t, dir)

	_, err := LoadKimi(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "kimi") {
		t.Errorf("error should mention kimi, got: %v", err)
	}
}

// --- parseKimiConfig ---

func TestModel_ParseKimiConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseKimiConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("parseKimiConfig: %v", err)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-5 {
		t.Errorf("RMSNormEps default = %g, want 1e-5", cfg.RMSNormEps)
	}
	if cfg.VocabSize != 0 {
		t.Errorf("VocabSize at parse = %d, want 0 (dimension not fabricated — derived from the embed tensor at load)", cfg.VocabSize)
	}
	// head_dim inferred from hidden/heads when absent.
	if cfg.HeadDim != 128 {
		t.Errorf("HeadDim inferred = %d, want 128", cfg.HeadDim)
	}
}

func TestModel_ParseKimiConfig_QuantizationNested_Good(t *testing.T) {
	cfg, err := parseKimiConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"head_dim": 128,
		"quantization_config": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseKimiConfig: %v", err)
	}
	if cfg.Quantization == nil {
		t.Fatal("expected quantization config from quantization_config key")
	}
	if cfg.Quantization.GroupSize != 64 {
		t.Errorf("GroupSize = %d, want 64", cfg.Quantization.GroupSize)
	}
	if cfg.Quantization.Bits != 4 {
		t.Errorf("Bits = %d, want 4", cfg.Quantization.Bits)
	}
}

func TestModel_ParseKimiConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseKimiConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// TestModel_ParseKimiConfig_MalformedQuantization_Bad drives the nested-wrapper
// unmarshal failure (kimi.go 127-129). KimiConfig.Quantization is json:"-" so the
// first unmarshal ignores a `quantization` key and succeeds; wrapper.Quantization
// is json:"quantization" so the second unmarshal of a string-where-object-expected
// value fails — the only way to reach the "parse nested config" wrap.
func TestModel_ParseKimiConfig_MalformedQuantization_Bad(t *testing.T) {
	_, err := parseKimiConfig([]byte(`{
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"quantization": "not-an-object"
	}`))
	if err == nil {
		t.Fatal("expected error for a malformed quantization value (string, not object)")
	}
	if !core.Contains(err.Error(), "kimi") {
		t.Fatalf("error = %v, want kimi-prefixed parse error", err)
	}
}

// --- KimiConfig expert sizing ---

func TestModel_KimiConfig_ExpertCount_Good(t *testing.T) {
	// num_experts wins when present.
	if got := (&KimiConfig{NumExperts: 16}).expertCount(); got != 16 {
		t.Errorf("expertCount(num_experts=16) = %d, want 16", got)
	}
	// falls back to num_local_experts, then n_routed_experts.
	if got := (&KimiConfig{NumLocalExperts: 8}).expertCount(); got != 8 {
		t.Errorf("expertCount(num_local_experts=8) = %d, want 8", got)
	}
	if got := (&KimiConfig{NRoutedExperts: 4}).expertCount(); got != 4 {
		t.Errorf("expertCount(n_routed_experts=4) = %d, want 4", got)
	}
	// default when none set.
	if got := (&KimiConfig{}).expertCount(); got != 8 {
		t.Errorf("expertCount(default) = %d, want 8", got)
	}
}

func TestModel_KimiConfig_TopK_Good(t *testing.T) {
	if got := (&KimiConfig{NumExpertsPerTok: 6}).topK(); got != 6 {
		t.Errorf("topK(num_experts_per_tok=6) = %d, want 6", got)
	}
	if got := (&KimiConfig{MoETopK: 3}).topK(); got != 3 {
		t.Errorf("topK(moe_topk=3) = %d, want 3", got)
	}
	if got := (&KimiConfig{}).topK(); got != 2 {
		t.Errorf("topK(default) = %d, want 2", got)
	}
}

// --- MoETextRuntimeAvailable (relocated from package metal) ---

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	m := &KimiModel{
		Layers: []*KimiDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &KimiMoEBlock{
				Router:        router,
				Experts:       []*KimiExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !m.MoETextRuntimeAvailable() {
		t.Fatal("KimiModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := m.MoETextDecodeFamily(); got != "kimi" {
		t.Fatalf("MoETextDecodeFamily() = %q, want kimi", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&KimiModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty KimiModel.MoETextRuntimeAvailable() = true, want false")
	}
	incomplete := &KimiModel{Layers: []*KimiDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete KimiModel.MoETextRuntimeAvailable() = true, want false")
	}
}

// --- kimiToQwen3Config (pure Go, no Metal) ---

func TestModel_KimiToQwen3Config_Good(t *testing.T) {
	cfg := &KimiConfig{
		HiddenSize:            1024,
		NumHiddenLayers:       24,
		NumAttentionHeads:     16,
		NumKeyValueHeads:      4,
		HeadDim:               64,
		VocabSize:             163840,
		RMSNormEps:            1e-6,
		MaxPositionEmbeddings: 131072,
		RopeTheta:             500000,
		Scale:                 1.0,
	}
	dc := kimiToQwen3Config(cfg)
	if dc == nil {
		t.Fatal("kimiToQwen3Config returned nil for a valid config")
	}
	if dc.HiddenSize != 1024 || dc.NumHiddenLayers != 24 || dc.NumAttentionHeads != 16 {
		t.Fatalf("DenseConfig = %+v, want hidden=1024 layers=24 heads=16", dc)
	}
	if dc.NumKeyValueHeads != 4 || dc.HeadDim != 64 || dc.VocabSize != 163840 {
		t.Fatalf("DenseConfig = %+v, want kv=4 headdim=64 vocab=163840", dc)
	}
	if dc.RopeTheta != 500000 || dc.Scale != 1.0 || dc.MaxPositionEmbeddings != 131072 {
		t.Fatalf("DenseConfig = %+v, want rope=500000 scale=1 maxpos=131072", dc)
	}
	if dc.RMSNormEps != 1e-6 {
		t.Fatalf("DenseConfig.RMSNormEps = %v, want 1e-6", dc.RMSNormEps)
	}
}

// TestModel_KimiToQwen3Config_Ugly drives the nil-config guard (kimi.go 394-396):
// a nil KimiConfig yields a nil DenseConfig rather than panicking.
func TestModel_KimiToQwen3Config_Ugly(t *testing.T) {
	if dc := kimiToQwen3Config(nil); dc != nil {
		t.Fatalf("kimiToQwen3Config(nil) = %+v, want nil", dc)
	}
}

// --- kimiSwitchExperts edge (pure Go, no Metal) ---

// TestModel_KimiSwitchExperts_NilExpert_Ugly drives the nil-expert early return
// in kimiSwitchExperts (kimi.go 337-340): a nil entry yields (nil, false).
func TestModel_KimiSwitchExperts_NilExpert_Ugly(t *testing.T) {
	se, ok := kimiSwitchExperts([]*KimiExpert{nil})
	if ok {
		t.Fatal("kimiSwitchExperts(nil expert) ok = true, want false")
	}
	if se != nil {
		t.Fatal("kimiSwitchExperts(nil expert) experts != nil, want nil")
	}
}

// --- ModelType / NumLayers / MoETextDecodeFamily (hand-built, no Metal) ---

func TestModel_ModelType_Good(t *testing.T) {
	m := &KimiModel{modelType: "kimi"}
	if m.ModelType() != "kimi" {
		t.Fatalf("ModelType() = %q, want kimi", m.ModelType())
	}
}

func TestModel_NumLayers_Good(t *testing.T) {
	m := &KimiModel{Layers: []*KimiDecoderLayer{nil, nil, nil}}
	if m.NumLayers() != 3 {
		t.Fatalf("NumLayers() = %d, want 3", m.NumLayers())
	}
}

func TestModel_MoETextDecodeFamily_Good(t *testing.T) {
	if got := (&KimiModel{}).MoETextDecodeFamily(); got != "kimi" {
		t.Fatalf("MoETextDecodeFamily() = %q, want kimi", got)
	}
}

// TestModel_MoETextRuntimeAvailable_NilReceiver_Ugly drives the nil-receiver
// guard in MoETextRuntimeAvailable (kimi.go 91-93).
func TestModel_MoETextRuntimeAvailable_NilReceiver_Ugly(t *testing.T) {
	var m *KimiModel
	if m.MoETextRuntimeAvailable() {
		t.Fatal("nil-receiver MoETextRuntimeAvailable() = true, want false")
	}
}

// TestModel_MoETextRuntimeAvailable_NilLayer_Ugly drives the per-layer nil guard
// inside the reporter callback (kimi.go 95-97).
func TestModel_MoETextRuntimeAvailable_NilLayer_Ugly(t *testing.T) {
	m := &KimiModel{Layers: []*KimiDecoderLayer{nil}}
	if m.MoETextRuntimeAvailable() {
		t.Fatal("model with a nil layer MoETextRuntimeAvailable() = true, want false")
	}
}

// TestModel_FillModelInfo_Quantized confirms the quant fields are copied when the
// config carries a quantization block (methods.go 24-27) — no Metal needed.
func TestModel_FillModelInfo_Quantized(t *testing.T) {
	model := &KimiModel{Cfg: &KimiConfig{
		VocabSize:             100,
		HiddenSize:            64,
		MaxPositionEmbeddings: 4096,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("info quant = bits %d group %d, want bits 4 group 64", info.QuantBits, info.QuantGroup)
	}
	if info.VocabSize != 100 || info.HiddenSize != 64 || info.ContextLength != 4096 {
		t.Fatalf("info = %+v, want vocab=100 hidden=64 ctx=4096", info)
	}
}

func TestModel_CloseModel_NilModel_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("CloseModel on nil model panicked: %v", recovered)
		}
	}()
	var m *KimiModel
	m.CloseModel()
}

// ── LoadKimi full load + the loaded-model fixture ──────────────────────────

// kimiMixedConfigJSON describes a synthetic 2-layer Kimi model where
// decoder_sparse_step=2 makes layer 0 DENSE (SiLU MLP) and layer 1 MoE (router +
// experts) — the dense/MoE twin geometry. Loading it covers both decoder
// branches of kimiDecoderLayerForward and the whole LoadKimi assembly loop
// (kimi.go 209-243) in one shot.
const kimiMixedConfigJSON = `{
	"architectures": ["KimiForCausalLM"],
	"model_type": "kimi",
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

// writeMixedKimiModel writes config.json, tokenizer.json and a safetensors file
// for the kimiMixedConfigJSON geometry: layer 0 dense MLP weights, layer 1 router
// + 2 expert weight triples, plus shared embed/norm/lm_head. Mirrors the LoadKimi
// weight-name convention (kimiLoadRouter resolves the ".gate" suffix; kimiLoadExpert
// resolves "mlp.experts.N.{gate,up,down}_proj").
func writeMixedKimiModel(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), kimiMixedConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalKimiTokenizer(t, dir)

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
	defer freeKimiArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
}

// loadMixedKimi loads the synthetic mixed dense+MoE model and registers its
// cleanup; the returned model is the shared fixture for the method tests.
func loadMixedKimi(t *testing.T) *KimiModel {
	t.Helper()
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedKimiModel(t, dir)
	model, err := LoadKimi(dir)
	if err != nil {
		t.Fatalf("LoadKimi(mixed dense+MoE) error = %v", err)
	}
	t.Cleanup(model.CloseModel)
	return model
}

func TestModel_LoadKimi_Good(t *testing.T) {
	model := loadMixedKimi(t)

	if model.ModelType() != "kimi" {
		t.Fatalf("ModelType() = %q, want kimi", model.ModelType())
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

// TestModel_LoadKimi_DerivesVocabFromEmbedding_Good omits vocab_size from the
// config so LoadKimi must derive it from the embed tensor's first dim
// (kimi.go 184-190).
func TestModel_LoadKimi_DerivesVocabFromEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
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
	writeMinimalKimiTokenizer(t, dir)

	const (
		h     = 8
		inter = 16
		v     = 7 // derived from embed tensor first dim
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
	defer freeKimiArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadKimi(dir)
	if err != nil {
		t.Fatalf("LoadKimi error = %v", err)
	}
	defer model.CloseModel()
	if model.Cfg.VocabSize != v {
		t.Fatalf("derived VocabSize = %d, want %d from embed tensor", model.Cfg.VocabSize, v)
	}
}

func TestModel_LoadKimi_MissingConfig_Bad(t *testing.T) {
	_, err := LoadKimi(t.TempDir())
	if err == nil {
		t.Fatal("LoadKimi(no config.json) error = nil, want load-config error")
	}
	if !core.Contains(err.Error(), "kimi") {
		t.Fatalf("error = %v, want kimi prefix", err)
	}
}

// ── Forward / ForwardMasked over the loaded fixture ────────────────────────

// kimiForwardTokens builds a [B=1, L] int32 token input within the synthetic
// vocab (size 5) and materialises it.
func kimiForwardTokens(l int) *metal.Array {
	ids := make([]int32, l)
	for i := range ids {
		ids[i] = int32(i % 5)
	}
	tokens := metal.FromValues(ids, 1, l)
	metal.Materialize(tokens)
	return tokens
}

func TestModel_Forward_Good(t *testing.T) {
	model := loadMixedKimi(t)
	tokens := kimiForwardTokens(3)
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

func TestModel_ForwardMasked_Good(t *testing.T) {
	model := loadMixedKimi(t)
	tokens := kimiForwardTokens(1)
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

// ── NewCache / FillModelInfo / Tokenizer over the loaded fixture ───────────

func TestModel_NewCache_Good(t *testing.T) {
	model := loadMixedKimi(t)
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

func TestModel_FillModelInfo_Good(t *testing.T) {
	model := loadMixedKimi(t)
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 5 {
		t.Fatalf("info.VocabSize = %d, want 5", info.VocabSize)
	}
	if info.HiddenSize != 8 {
		t.Fatalf("info.HiddenSize = %d, want 8", info.HiddenSize)
	}
	if info.ContextLength != 32 {
		t.Fatalf("info.ContextLength = %d, want 32 (max_position_embeddings)", info.ContextLength)
	}
}

func TestModel_Tokenizer_Good(t *testing.T) {
	model := loadMixedKimi(t)
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want the loaded tokenizer")
	}
	if model.Tokenizer() != model.Tok {
		t.Fatal("Tokenizer() did not return the model's Tok field")
	}
}

// TestModel_MoETextRuntimeAvailable_Loaded_Good asserts the fully-loaded mixed
// model reports native MoE decode available (the loaded counterpart to the
// hand-built MoETextRuntimeAvailable_Good above).
func TestModel_MoETextRuntimeAvailable_Loaded_Good(t *testing.T) {
	model := loadMixedKimi(t)
	if !model.MoETextRuntimeAvailable() {
		t.Fatal("loaded model MoETextRuntimeAvailable() = false, want true (native MoE decode linked)")
	}
}

// ── ApplyLoRA ──────────────────────────────────────────────────────────────

func TestModel_ApplyLoRA_Good(t *testing.T) {
	model := loadMixedKimi(t)
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

// TestModel_ApplyLoRA_AllAttentionTargets exercises every attention switch arm
// (q/k/v/o_proj) so each projection-selection branch is covered (kimi.go 435-442).
func TestModel_ApplyLoRA_AllAttentionTargets(t *testing.T) {
	model := loadMixedKimi(t)
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

// TestModel_ApplyLoRA_DenseMLPTargets confirms gate/up/down LoRA targets land
// only on the dense layer (layer 0); the MoE layer has no Dense.MLP so it is
// skipped (kimi.go 443-454).
func TestModel_ApplyLoRA_DenseMLPTargets(t *testing.T) {
	model := loadMixedKimi(t)
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

func TestModel_ApplyLoRA_EmptyModel_Ugly(t *testing.T) {
	m := &KimiModel{}
	adapter := m.ApplyLoRA(metal.LoRAConfig{Rank: 2, Alpha: 4, TargetLayers: []string{"q_proj"}})
	if adapter == nil {
		t.Fatal("ApplyLoRA(empty model) = nil, want a usable empty adapter")
	}
	if len(adapter.Layers) != 0 {
		t.Fatalf("adapter.Layers = %d, want 0 for a model with no layers", len(adapter.Layers))
	}
	// Config still normalised even with no layers (Alpha/Scale resolved).
	if adapter.Config.Rank != 2 || adapter.Config.Alpha != 4 {
		t.Fatalf("adapter.Config = %+v, want rank=2 alpha=4 normalised", adapter.Config)
	}
}

// ── CloseModel ─────────────────────────────────────────────────────────────

func TestModel_CloseModel_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedKimiModel(t, dir)
	model, err := LoadKimi(dir)
	if err != nil {
		t.Fatalf("LoadKimi error = %v", err)
	}
	embedW := model.EmbedTokens.Weight
	outW := model.Output.Weight
	denseGate := model.Layers[0].Dense.MLP.GateProj.Weight
	qW := model.Layers[0].Dense.Attention.QProj.Weight
	routerW := model.Layers[1].MoE.Router.Weight
	expertGate := model.Layers[1].MoE.Experts[0].GateProj.Weight

	model.CloseModel()

	if embedW.Valid() {
		t.Error("embed weight should be freed after CloseModel")
	}
	if outW != embedW && outW.Valid() {
		t.Error("output weight should be freed after CloseModel")
	}
	if denseGate.Valid() {
		t.Error("dense MLP gate weight should be freed after CloseModel")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed after CloseModel")
	}
	if routerW.Valid() {
		t.Error("MoE router weight should be freed after CloseModel")
	}
	if expertGate.Valid() {
		t.Error("MoE expert gate weight should be freed after CloseModel")
	}
	if model.Layers != nil {
		t.Error("Layers should be nil after CloseModel")
	}
}

// ── LoadKimi quantized load path (synthetic 4-bit, LOAD-ONLY) ──────────────

// TestModel_LoadKimi_Quantized_Good loads a synthetic 4-bit checkpoint:
// metal.Quantize packs each attention + dense-MLP projection, the MoE router,
// and the embedding/lm_head into the (weight, scales, biases) triplet the
// loader's quantized `linear` closure + kimiLoadRouter resolve — driving the
// q != nil branches (kimi.go 170-180, 192-198, 247-253, 305-308) that the bf16
// mixed fixture skips.
//
// Experts stay PLAIN dense arrays on purpose: kimiLoadExpert (kimi.go 316-331)
// has no quant arm and would reinterpret packed bytes as bf16. For the same
// reason this test is load-only — it never runs Forward.
func TestModel_LoadKimi_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	const (
		h     = int32(64)
		inter = int32(128)
		v     = int32(64)
		hd    = int32(32)
		nh    = int32(2) // nh*hd == h == 64 → o_proj last dim divides group_size
		gs    = 64       // MLX supports group sizes 32, 64, 128 and it must divide the quantized dim
		bits  = 4
	)
	dir := t.TempDir()
	// Two layers, decoder_sparse_step=2 → layer 0 dense, layer 1 MoE.
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
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
	writeMinimalKimiTokenizer(t, dir)

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
	defer freeKimiArrayMap(weights)

	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadKimi(dir)
	if err != nil {
		t.Fatalf("LoadKimi(quantized) error = %v", err)
	}
	defer model.CloseModel()

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != bits || info.QuantGroup != gs {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, bits, gs)
	}
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved (kimi.go 192-198)")
	}
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("quantized lm_head Output not resolved (kimi.go 247-253)")
	}
	if model.Layers[1].MoE == nil || model.Layers[1].MoE.Router.Scales == nil {
		t.Error("quantized MoE router scales were not resolved (kimi.go 305-308)")
	}
}

// TestModel_LoadKimi_TiedEmbedding_Good omits lm_head entirely so LoadKimi ties
// the output projection to the embedding via AsLinear (kimi.go 257-259) — the
// weights-untied default for this synthetic geometry.
func TestModel_LoadKimi_TiedEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "kimi",
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
	writeMinimalKimiTokenizer(t, dir)

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
	defer freeKimiArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadKimi(dir)
	if err != nil {
		t.Fatalf("LoadKimi(tied embedding) error = %v", err)
	}
	defer model.CloseModel()
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("tied Output.Weight = nil, want the embedding weight via AsLinear")
	}
	if model.Output.Weight != model.EmbedTokens.Weight {
		t.Fatal("Output weight not tied to EmbedTokens weight (AsLinear, kimi.go 257-259)")
	}
}

// ── kimiLoadRouter / kimiLoadExpert not-found fallbacks (pure Go, no Metal) ──

// TestModel_KimiLoadRouter_NotFound_Ugly drives the no-router-weight fallback in
// kimiLoadRouter (kimi.go 313): when no prefix/suffix combination resolves a gate
// weight, an empty *metal.MoERouter is returned rather than nil.
func TestModel_KimiLoadRouter_NotFound_Ugly(t *testing.T) {
	router := kimiLoadRouter(map[string]*metal.Array{}, 0, nil)
	if router == nil {
		t.Fatal("kimiLoadRouter(no weights) = nil, want an empty &MoERouter{}")
	}
	if router.Weight != nil {
		t.Fatalf("kimiLoadRouter(no weights).Weight = %v, want nil", router.Weight)
	}
}

// TestModel_KimiLoadExpert_NotFound_Ugly drives the no-expert-weight fallback in
// kimiLoadExpert (kimi.go 330): when neither the mlp nor moe expert prefix
// resolves a gate weight, an empty &KimiExpert{} is returned.
func TestModel_KimiLoadExpert_NotFound_Ugly(t *testing.T) {
	expert := kimiLoadExpert(func(string) *metal.Array { return nil }, 0, 0)
	if expert == nil {
		t.Fatal("kimiLoadExpert(no weights) = nil, want an empty &KimiExpert{}")
	}
	if expert.GateProj != nil || expert.UpProj != nil || expert.DownProj != nil {
		t.Fatalf("kimiLoadExpert(no weights) = %+v, want all-nil projections", expert)
	}
}

// TestModel_CloseModel_NilLayer_Ugly drives the nil/Dense-less layer continue in
// closeKimi (close.go 25): a model carrying a nil layer entry must skip it
// without panicking. No Metal — the free helpers are nil-tolerant.
func TestModel_CloseModel_NilLayer_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("CloseModel with a nil layer panicked: %v", recovered)
		}
	}()
	m := &KimiModel{Layers: []*KimiDecoderLayer{nil, {Dense: nil}}}
	m.CloseModel()
	if m.Layers != nil {
		t.Fatal("Layers should be nil after CloseModel")
	}
}

// ── registry dispatch (init closure, methods.go 13-15) ─────────────────────

// TestModel_RegistryDispatch_Good loads the mixed fixture through
// metal.LoadAndInit, which probes model_type="kimi" and dispatches via the
// loader registry — exercising the closure registered in init (methods.go
// 13-15) that bridges the registry to LoadKimi.
func TestModel_RegistryDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedKimiModel(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(kimi) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "kimi" {
		t.Fatalf("dispatched ModelType() = %q, want kimi", model.ModelType())
	}
}

// --- helpers ---

// requireMetalRuntime skips a test unless the Metal runtime is both compiled in
// (-tags metal_runtime) and available on the host.
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// seqArr builds a deterministic [shape] float32 tensor (the qwen3/gptoss seqArray
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

func freeKimiArrayMap(arrays map[string]*metal.Array) {
	for _, a := range arrays {
		metal.Free(a)
	}
}

func moeReadyRuntimeParts(t *testing.T) (*metal.MoERouter, *metal.MoESwiGLUExperts, func()) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
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

func writeMinimalKimiConfig(t *testing.T, dir string) {
	t.Helper()
	config := `{
		"model_type": "kimi",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"vocab_size": 100,
		"rms_norm_eps": 1e-5,
		"num_local_experts": 2,
		"num_experts_per_tok": 2
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

func writeMinimalKimiTokenizer(t *testing.T, dir string) {
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
