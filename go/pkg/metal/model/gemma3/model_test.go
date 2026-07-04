// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// --- LoadGemma3 error paths ---

func TestModel_LoadGemma3_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma3",
		"hidden_size": 1152,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"vocab_size": 1000
	}`)

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadGemma3_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

// TestModel_LoadGemma3_MissingConfig_Bad drives the config-read failure arm
// (the coreio.Local.Read error before any parse): an empty directory has no
// config.json, so the loader must wrap "load config" rather than panic. The
// invalid/not-json cases above reach parseConfig — this one never gets there.
func TestModel_LoadGemma3_MissingConfig_Bad(t *testing.T) {
	_, err := LoadGemma3(t.TempDir())
	if err == nil {
		t.Fatal("expected error for missing config.json")
	}
	if !core.Contains(err.Error(), "config") {
		t.Errorf("error should mention config, got: %v", err)
	}
}

func TestModel_LoadGemma3_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalGemma3Config(t, dir, "gemma3")
	writeMinimalGemma3Tokenizer(t, dir)

	_, err := LoadGemma3(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Errorf("error should mention safetensors, got: %v", err)
	}
}

// --- parseConfig ---

func TestModel_ParseConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 128
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.RopeTheta != 1000000 {
		t.Errorf("RopeTheta default = %f, want 1000000", cfg.RopeTheta)
	}
	if cfg.RopeLocalBaseFreq != 10000 {
		t.Errorf("RopeLocalBaseFreq default = %f, want 10000", cfg.RopeLocalBaseFreq)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Errorf("RMSNormEps default = %f, want 1e-6", cfg.RMSNormEps)
	}
	if cfg.SlidingWindowPattern != 6 {
		t.Errorf("SlidingWindowPattern default = %d, want 6", cfg.SlidingWindowPattern)
	}
	if cfg.VocabSize != 0 {
		t.Errorf("VocabSize at parse = %d, want 0 (dimension not fabricated — derived from the embed tensor at load)", cfg.VocabSize)
	}
}

func TestModel_ParseConfig_QuantizationTopLevel_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"head_dim": 128,
		"quantization": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.Quantization == nil {
		t.Fatal("expected quantization config")
	}
	if cfg.Quantization.GroupSize != 64 {
		t.Errorf("GroupSize = %d, want 64", cfg.Quantization.GroupSize)
	}
	if cfg.Quantization.Bits != 4 {
		t.Errorf("Bits = %d, want 4", cfg.Quantization.Bits)
	}
}

func TestModel_ParseConfig_NestedTextConfig_Good(t *testing.T) {
	// Multimodal Gemma3 has text_config nested inside a wrapper.
	cfg, err := parseConfig([]byte(`{
		"model_type": "gemma3",
		"text_config": {
			"hidden_size": 2048,
			"num_hidden_layers": 16,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"vocab_size": 262144
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want 2048", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 16 {
		t.Errorf("NumHiddenLayers = %d, want 16", cfg.NumHiddenLayers)
	}
}

func TestModel_ParseConfig_PreservesModelType_Good(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"model_type": "gemma2",
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 128
	}`))
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.ModelType != "gemma2" {
		t.Fatalf("ModelType = %q, want gemma2", cfg.ModelType)
	}

	cfg, err = parseConfig([]byte(`{
		"model_type": "gemma2",
		"text_config": {
			"hidden_size": 2048,
			"num_hidden_layers": 16,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig nested: %v", err)
	}
	if cfg.ModelType != "gemma2" {
		t.Fatalf("nested ModelType = %q, want gemma2", cfg.ModelType)
	}
}

func TestModel_ParseConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// TestModel_ParseConfig_TopLevelTypeMismatch_Bad drives the top-level reparse
// failure arm (the second JSONUnmarshal inside parseConfig). The wrapper parse
// tolerates the wrongly-typed known field and leaves NumHiddenLayers zero, so
// the top-level fallback fires — and there the string-valued num_attention_heads
// fails to decode into the int32 field, exercising the "parse top-level config"
// error wrap. Valid JSON, well-formed wrapper, mistyped scalar.
func TestModel_ParseConfig_TopLevelTypeMismatch_Bad(t *testing.T) {
	_, err := parseConfig([]byte(`{"num_attention_heads": "not-a-number"}`))
	if err == nil {
		t.Fatal("expected error when the top-level reparse hits a type mismatch")
	}
	if !core.Contains(err.Error(), "top-level") {
		t.Errorf("error should come from the top-level fallback, got: %v", err)
	}
}

// --- isLayerSliding ---

func TestModel_IsLayerSliding_Good(t *testing.T) {
	// Pattern=6: every 6th layer is NOT sliding (global attention).
	// Layer 5 (index=5, i+1=6) → 6%6=0 → not sliding (global)
	// Layer 0 (index=0, i+1=1) → 1%6=1 → sliding
	tests := []struct {
		idx     int32
		pattern int32
		want    bool
	}{
		{0, 6, true},   // layer 1: 1%6=1 → sliding
		{4, 6, true},   // layer 5: 5%6=5 → sliding
		{5, 6, false},  // layer 6: 6%6=0 → global
		{11, 6, false}, // layer 12: 12%6=0 → global
		{0, 0, false},  // pattern=0 → no sliding
		{0, -1, false}, // pattern<0 → no sliding
	}
	for _, tt := range tests {
		got := isLayerSliding(tt.idx, tt.pattern)
		if got != tt.want {
			t.Errorf("isLayerSliding(%d, %d) = %v, want %v", tt.idx, tt.pattern, got, tt.want)
		}
	}
}

// --- Ugly paths ---

// TestModel_ParseConfig_NullBytes_Ugly tests parseConfig with null bytes in input.
// Should return a parse error, not panic.
func TestModel_ParseConfig_NullBytes_Ugly(t *testing.T) {
	_, err := parseConfig([]byte("\x00\x00\x00"))
	if err == nil {
		t.Fatal("expected error for null-byte input")
	}
}

// TestModel_ParseConfig_TruncatedJSON_Ugly tests parseConfig with truncated JSON.
// Should return a parse error, not panic.
func TestModel_ParseConfig_TruncatedJSON_Ugly(t *testing.T) {
	_, err := parseConfig([]byte(`{"hidden_size": 102`))
	if err == nil {
		t.Fatal("expected error for truncated JSON")
	}
}

// --- FillModelInfo ---

// TestModel_FillModelInfo_Unquantized_Good copies the sizing fields out of the
// config into a ModelInfo (ModelInfoReporter capability) and leaves the quant
// fields zeroed when the model is not quantized.
func TestModel_FillModelInfo_Unquantized_Good(t *testing.T) {
	m := &GemmaModel{Cfg: &TextConfig{
		VocabSize:             262144,
		HiddenSize:            1152,
		MaxPositionEmbeddings: 32768,
	}}
	var info metal.ModelInfo
	m.FillModelInfo(&info)

	if info.VocabSize != 262144 {
		t.Errorf("VocabSize = %d, want 262144", info.VocabSize)
	}
	if info.HiddenSize != 1152 {
		t.Errorf("HiddenSize = %d, want 1152", info.HiddenSize)
	}
	if info.ContextLength != 32768 {
		t.Errorf("ContextLength = %d, want 32768 (from MaxPositionEmbeddings)", info.ContextLength)
	}
	if info.QuantBits != 0 || info.QuantGroup != 0 {
		t.Errorf("unquantized model: QuantBits=%d QuantGroup=%d, want 0/0", info.QuantBits, info.QuantGroup)
	}
}

// TestModel_FillModelInfo_Quantized_Good drives the cfg.Quantization != nil
// branch — the quant bits/group from the config must surface in ModelInfo.
func TestModel_FillModelInfo_Quantized_Good(t *testing.T) {
	m := &GemmaModel{Cfg: &TextConfig{
		VocabSize:    262144,
		HiddenSize:   1152,
		Quantization: &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	var info metal.ModelInfo
	m.FillModelInfo(&info)

	if info.QuantBits != 4 {
		t.Errorf("QuantBits = %d, want 4", info.QuantBits)
	}
	if info.QuantGroup != 64 {
		t.Errorf("QuantGroup = %d, want 64", info.QuantGroup)
	}
}

// writeMinimalGemma3Config writes a minimal valid config.json for load tests.
func writeMinimalGemma3Config(t *testing.T, dir string, modelType string) {
	t.Helper()
	config := `{
		"model_type": "` + modelType + `",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 32,
		"vocab_size": 100,
		"rms_norm_eps": 1e-6
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
}

// writeMinimalGemma3Tokenizer writes a minimal valid tokenizer.json for load tests.
func writeMinimalGemma3Tokenizer(t *testing.T, dir string) {
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

// --- synthetic full-load harness ------------------------------------------
//
// These exercise the bulk of LoadGemma3 — the weight-resolution closures, the
// head_dim/vocab inference from tensor shapes, the per-layer build loop, and
// the tied/untied lm_head branches — from an in-package SaveSafetensors fixture
// (the qwen3_test.go / gemma4 assistant_test.go pattern, AX-11-clean: no live
// model, no safetensors on disk beyond the temp dir). The error arms above
// stop at the config/tokenizer/weights boundaries; these run past it.
//
// Geometry matches writeMinimalGemma3Config: hidden=64, 1 layer, inter=128,
// 2 query heads, 1 K/V head, head_dim=32, vocab=100.
const (
	loadHidden = 64
	loadInter  = 128
	loadHeads  = 2
	loadKV     = 1
	loadHeadD  = 32
	loadVocab  = 100
)

// gemma3SeqArray builds a deterministic [shape] f32 tensor — values are
// irrelevant to a load (the loader only resolves and reshapes), so a cheap
// ramp keeps the fixture self-contained.
func gemma3SeqArray(start float32, shape ...int) *metal.Array {
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

// freeGemma3Arrays releases a fixture weight map.
func freeGemma3Arrays(arrays map[string]*metal.Array) {
	for _, array := range arrays {
		metal.Free(array)
	}
}

// tinyGemma3Weights returns the full safetensors weight set for a 1-layer
// Gemma 3 checkpoint at the minimal-config geometry — every tensor LoadGemma3
// resolves, so a load drives the whole 167-281 body: the weight/linear/embed
// closures, the per-layer attention (q/k/v/o + q_norm/k_norm) and MLP build,
// model.norm, and (when lm_head.weight is present) the untied-output branch.
func tinyGemma3Weights() map[string]*metal.Array {
	const (
		h    = loadHidden
		i    = loadInter
		v    = loadVocab
		qOut = loadHeads * loadHeadD // 64
		kOut = loadKV * loadHeadD    // 32
	)
	return map[string]*metal.Array{
		"model.embed_tokens.weight":                        gemma3SeqArray(0.01, v, h),
		"model.layers.0.input_layernorm.weight":            gemma3SeqArray(0.02, h),
		"model.layers.0.post_attention_layernorm.weight":   gemma3SeqArray(0.03, h),
		"model.layers.0.pre_feedforward_layernorm.weight":  gemma3SeqArray(0.04, h),
		"model.layers.0.post_feedforward_layernorm.weight": gemma3SeqArray(0.05, h),
		"model.layers.0.self_attn.q_proj.weight":           gemma3SeqArray(0.06, qOut, h),
		"model.layers.0.self_attn.k_proj.weight":           gemma3SeqArray(0.07, kOut, h),
		"model.layers.0.self_attn.v_proj.weight":           gemma3SeqArray(0.08, kOut, h),
		"model.layers.0.self_attn.o_proj.weight":           gemma3SeqArray(0.09, h, qOut),
		"model.layers.0.self_attn.q_norm.weight":           gemma3SeqArray(0.10, loadHeadD),
		"model.layers.0.self_attn.k_norm.weight":           gemma3SeqArray(0.11, loadHeadD),
		"model.layers.0.mlp.gate_proj.weight":              gemma3SeqArray(0.12, i, h),
		"model.layers.0.mlp.up_proj.weight":                gemma3SeqArray(0.13, i, h),
		"model.layers.0.mlp.down_proj.weight":              gemma3SeqArray(0.14, h, i),
		"model.norm.weight":                                gemma3SeqArray(0.15, h),
		"lm_head.weight":                                   gemma3SeqArray(0.16, v, h),
	}
}

// TestModel_LoadGemma3_FullLoad_UntiedOutput_Good drives the dense loader
// end-to-end from a synthetic safetensors directory: config → tokenizer →
// weights → per-layer build → final norm + precomputeScaledWeights. lm_head.weight
// is present, so the untied-output branch (Output is its own Linear) is taken.
func TestModel_LoadGemma3_FullLoad_UntiedOutput_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalGemma3Config(t, dir, "gemma3")
	writeMinimalGemma3Tokenizer(t, dir)

	weights := tinyGemma3Weights()
	defer freeGemma3Arrays(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma3(dir)
	if err != nil {
		t.Fatalf("LoadGemma3(synthetic dense) error = %v", err)
	}
	defer closeGemma(model)

	if model.NumLayers() != 1 {
		t.Fatalf("NumLayers = %d, want 1", model.NumLayers())
	}
	if model.ModelType() != "gemma3" {
		t.Errorf("ModelType = %q, want gemma3", model.ModelType())
	}
	// lm_head.weight was present — Output must be its own Linear, not the tied
	// embedding fallback.
	if model.Output == nil || model.Output.Weight == nil || model.Output.Weight == model.EmbedTokens.Weight {
		t.Error("untied lm_head.weight was present — Output must not alias EmbedTokens")
	}
	// Norm scales must be materialised by precomputeScaledWeights.
	if model.NormScaled == nil || !model.NormScaled.Valid() {
		t.Error("NormScaled not materialised after load")
	}
	if l := model.Layers[0]; l.InputNormScaled == nil || !l.InputNormScaled.Valid() {
		t.Error("layer InputNormScaled not materialised after load")
	}
}

// TestModel_LoadGemma3_RegisteredLoader_Good drives the loader through the
// public metal.LoadAndInit entry point on the synthetic dir. LoadAndInit →
// loadModel → lookupModelLoader("gemma3") dispatches to the closure registered
// in this package's init(), so the closure body (return LoadGemma3(modelPath))
// runs — the registry wiring methods.go installs, exercised end-to-end without
// a live model. The config's model_type is "gemma3", one of the ids init binds.
func TestModel_LoadGemma3_RegisteredLoader_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalGemma3Config(t, dir, "gemma3")
	writeMinimalGemma3Tokenizer(t, dir)

	weights := tinyGemma3Weights()
	defer freeGemma3Arrays(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(gemma3 synthetic) error = %v", err)
	}
	defer model.Close()

	if model.ModelType() != "gemma3" {
		t.Errorf("dispatched model type = %q, want gemma3", model.ModelType())
	}
}

// TestModel_LoadGemma3_FullLoad_TiedOutput_Good drops lm_head.weight so the
// loader falls back to the tied embedding-as-linear output (the else arm at the
// lm_head branch), and closeGemma must then NOT double-free the shared weight.
func TestModel_LoadGemma3_FullLoad_TiedOutput_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalGemma3Config(t, dir, "gemma3")
	writeMinimalGemma3Tokenizer(t, dir)

	weights := tinyGemma3Weights()
	delete(weights, "lm_head.weight") // force the tied-output fallback
	defer freeGemma3Arrays(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma3(dir)
	if err != nil {
		t.Fatalf("LoadGemma3(tied output) error = %v", err)
	}
	defer closeGemma(model)

	if model.Output == nil || model.Output.Weight != model.EmbedTokens.Weight {
		t.Error("no lm_head.weight — Output must alias EmbedTokens (tied)")
	}
}

// TestModel_LoadGemma3_FullLoad_Quantized_Good drives the quantized arms of the
// loader: the quant-config log line, the `linear` closure's NewQuantizedLinear
// branch (scales present), the quantized embedding branch, and the quantized
// lm_head branch. Weights are packed with metal.Quantize (the qwen3_test.go
// fixture style) so the .scales/.biases companions are well-formed — no manual
// byte packing, still no live model.
func TestModel_LoadGemma3_FullLoad_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	const (
		h    = loadHidden // 64
		i    = loadInter  // 128
		v    = loadVocab  // 100
		qOut = loadHeads * loadHeadD
		kOut = loadKV * loadHeadD
		gs   = 64
		bits = 4
	)
	dir := t.TempDir()
	// group_size 64 needs the contracted (in) dim divisible by 64; hidden=64
	// and inter=128 both qualify. vocab=100 is the embed/lm_head OUT dim (rows),
	// not contracted, so it is unconstrained.
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma3",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 32,
		"vocab_size": 100,
		"rms_norm_eps": 1e-6,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalGemma3Tokenizer(t, dir)

	weights := map[string]*metal.Array{}
	addQuant := func(name string, out, in int32) {
		dense := gemma3SeqArray(0.02, int(out), int(in))
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
	addQuant("model.layers.0.self_attn.q_proj", qOut, h)
	addQuant("model.layers.0.self_attn.k_proj", kOut, h)
	addQuant("model.layers.0.self_attn.v_proj", kOut, h)
	addQuant("model.layers.0.self_attn.o_proj", h, qOut)
	addQuant("model.layers.0.mlp.gate_proj", i, h)
	addQuant("model.layers.0.mlp.up_proj", i, h)
	addQuant("model.layers.0.mlp.down_proj", h, i)
	addQuant("lm_head", v, h)
	weights["model.layers.0.input_layernorm.weight"] = gemma3SeqArray(0.02, h)
	weights["model.layers.0.post_attention_layernorm.weight"] = gemma3SeqArray(0.03, h)
	weights["model.layers.0.pre_feedforward_layernorm.weight"] = gemma3SeqArray(0.04, h)
	weights["model.layers.0.post_feedforward_layernorm.weight"] = gemma3SeqArray(0.05, h)
	weights["model.layers.0.self_attn.q_norm.weight"] = gemma3SeqArray(0.06, loadHeadD)
	weights["model.layers.0.self_attn.k_norm.weight"] = gemma3SeqArray(0.07, loadHeadD)
	weights["model.norm.weight"] = gemma3SeqArray(0.08, h)
	defer freeGemma3Arrays(weights)

	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma3(dir)
	if err != nil {
		t.Fatalf("LoadGemma3(quantized) error = %v", err)
	}
	defer closeGemma(model)

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != bits || info.QuantGroup != gs {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, bits, gs)
	}
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved")
	}
	// lm_head.weight was present and quantized → untied quantized Output.
	if model.Output == nil || model.Output.Scales == nil {
		t.Error("quantized lm_head scales were not resolved on Output")
	}
	if q := model.Layers[0].Attention.QProj; q == nil || q.Scales == nil {
		t.Error("quantized q_proj scales were not resolved")
	}
}

// TestModel_LoadGemma3_InfersHeadDimAndVocab_Good omits head_dim and vocab_size
// from the config so the loader must derive both from tensor shapes: head_dim
// from the q_proj row count / num_attention_heads, vocab_size from the
// embed-token row count. The config never declares either — both are read from
// the weights, never fabricated.
func TestModel_LoadGemma3_InfersHeadDimAndVocab_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	// Config WITHOUT head_dim or vocab_size — they must be inferred.
	cfg := `{
		"model_type": "gemma3",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"rms_norm_eps": 1e-6
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalGemma3Tokenizer(t, dir)

	weights := tinyGemma3Weights()
	defer freeGemma3Arrays(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma3(dir)
	if err != nil {
		t.Fatalf("LoadGemma3(infer head_dim/vocab) error = %v", err)
	}
	defer closeGemma(model)

	// q_proj is [num_heads*head_dim, hidden] = [64,64] → head_dim = 64/2 = 32.
	if model.Cfg.HeadDim != loadHeadD {
		t.Errorf("inferred HeadDim = %d, want %d (q_proj rows / num_attention_heads)", model.Cfg.HeadDim, loadHeadD)
	}
	// embed_tokens is [vocab, hidden] = [100,64] → vocab = 100.
	if model.Cfg.VocabSize != loadVocab {
		t.Errorf("inferred VocabSize = %d, want %d (embed_tokens row count)", model.Cfg.VocabSize, loadVocab)
	}
}
