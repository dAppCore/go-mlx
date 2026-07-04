// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"math"
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

func TestQwen3_ParseConfigMissingHeads_Bad(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("ParseDenseConfig panicked for missing heads: %v", recovered)
		}
	}()

	cfg, err := metal.ParseDenseConfig([]byte(`{"model_type":"qwen2","vocab_size":16,"hidden_size":4,"num_hidden_layers":1,"max_position_embeddings":32}`))

	if err != nil {
		t.Fatalf("ParseDenseConfig: %v", err)
	}
	if cfg.HeadDim != 0 {
		t.Fatalf("head_dim = %d, want 0 when attention heads are absent", cfg.HeadDim)
	}
}

func TestModel_LoadQwen3_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing config")
	}
}

func TestModel_LoadQwen3_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "{broken")

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadQwen3_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000
	}`)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadQwen3_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "safetensors") {
		t.Errorf("error should mention safetensors, got: %v", err)
	}
}

// tinyDenseWeights returns the full safetensors weight set for a 1-layer dense
// Qwen3 checkpoint at a tiny geometry (hidden=8, 2 heads, vocab=5) — every
// tensor LoadQwen3 resolves, so a load drives the whole 94-198 body: the linear
// closure, embed, the per-layer attention + MLP build, and the final norm. The
// lm_head.weight key is present, so the untied-output branch is taken.
func tinyDenseWeights(hidden, intermediate, vocab int32) map[string]*metal.Array {
	h, i, v := int(hidden), int(intermediate), int(vocab)
	return map[string]*metal.Array{
		"model.embed_tokens.weight":                      seqArray(0.01, v, h),
		"model.layers.0.input_layernorm.weight":          seqArray(0.02, h),
		"model.layers.0.post_attention_layernorm.weight": seqArray(0.03, h),
		"model.layers.0.self_attn.q_proj.weight":         seqArray(0.04, h, h),
		"model.layers.0.self_attn.k_proj.weight":         seqArray(0.05, h, h),
		"model.layers.0.self_attn.v_proj.weight":         seqArray(0.06, h, h),
		"model.layers.0.self_attn.o_proj.weight":         seqArray(0.07, h, h),
		"model.layers.0.self_attn.q_norm.weight":         seqArray(0.08, h),
		"model.layers.0.self_attn.k_norm.weight":         seqArray(0.09, h),
		"model.layers.0.mlp.gate_proj.weight":            seqArray(0.10, i, h),
		"model.layers.0.mlp.up_proj.weight":              seqArray(0.11, i, h),
		"model.layers.0.mlp.down_proj.weight":            seqArray(0.12, h, i),
		"model.norm.weight":                              seqArray(0.13, h),
		"lm_head.weight":                                 seqArray(0.14, v, h),
	}
}

// TestModel_LoadQwen3_FullLoad_Good drives the dense loader end-to-end from a
// synthetic safetensors directory (the in-package SaveSafetensors fixture style,
// AX-11-clean — no live model): config → tokenizer → weights → layer build →
// untied lm_head. It is the coverage path for the bulk of LoadQwen3.
func TestModel_LoadQwen3_FullLoad_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	weights := tinyDenseWeights(64, 128, 100)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3(dir)
	if err != nil {
		t.Fatalf("LoadQwen3(synthetic dense) error = %v", err)
	}
	defer closeQwen3(model)

	if model.ModelType() == "" {
		t.Error("ModelType is empty after load")
	}
	if model.NumLayers() != 1 {
		t.Fatalf("NumLayers = %d, want 1", model.NumLayers())
	}
	// lm_head.weight was present, so Output must be its own Linear, not the
	// tied embedding fallback.
	if model.Output == nil || model.Output.Weight == model.EmbedTokens.Weight {
		t.Error("untied lm_head.weight was present — Output must not alias EmbedTokens")
	}
}

// TestModel_LoadQwen3_TiedOutput_Good drops lm_head.weight, so the loader falls
// back to the tied embedding-as-linear output (the `else` branch at qwen3.go),
// and closeQwen3 must then NOT double-free the shared weight.
func TestModel_LoadQwen3_TiedOutput_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeMinimalConfig(t, dir, "qwen3")
	writeMinimalTokenizer(t, dir)

	weights := tinyDenseWeights(64, 128, 100)
	delete(weights, "lm_head.weight") // force the tied-output fallback
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3(dir)
	if err != nil {
		t.Fatalf("LoadQwen3(tied output) error = %v", err)
	}
	if model.Output.Weight != model.EmbedTokens.Weight {
		t.Fatal("with no lm_head.weight, Output must alias the embedding weight")
	}

	// closeQwen3 must free the embedding once and skip the aliased output — no
	// double free, no panic.
	closeQwen3(model)
	if model.EmbedTokens.Weight.Valid() {
		t.Error("tied embedding/output weight should be freed exactly once")
	}
}

// TestModel_LoadQwen3_Quantized_Good loads a synthetic 4-bit quantized dense
// checkpoint: metal.Quantize packs each projection into the (weight, scales,
// biases) triplet the loader's quantized `linear` closure resolves, plus a
// quantized embedding — driving the q != nil branches that the bf16 fixture
// skips. No live model.
func TestModel_LoadQwen3_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	const (
		hidden = int32(64)
		inter  = int32(128)
		vocab  = int32(64)
		gs     = 64
		bits   = 4
	)
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 32,
		"vocab_size": 64,
		"rms_norm_eps": 1e-6,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	weights := map[string]*metal.Array{}
	// quant packs a dense [out,in] weight into its quantized triplet under the
	// base name (no .weight suffix replacement needed — SaveSafetensors keys are
	// the map keys).
	addQuant := func(name string, out, in int32) {
		dense := seqArray(0.02, int(out), int(in))
		wq, sc, bi, err := metal.Quantize(dense, gs, bits, "")
		if err != nil {
			t.Fatalf("Quantize(%s): %v", name, err)
		}
		metal.Free(dense)
		weights[name+".weight"] = wq
		weights[name+".scales"] = sc
		weights[name+".biases"] = bi
	}
	addQuant("model.embed_tokens", vocab, hidden)
	addQuant("model.layers.0.self_attn.q_proj", hidden, hidden)
	addQuant("model.layers.0.self_attn.k_proj", hidden, hidden)
	addQuant("model.layers.0.self_attn.v_proj", hidden, hidden)
	addQuant("model.layers.0.self_attn.o_proj", hidden, hidden)
	addQuant("model.layers.0.mlp.gate_proj", inter, hidden)
	addQuant("model.layers.0.mlp.up_proj", inter, hidden)
	addQuant("model.layers.0.mlp.down_proj", hidden, inter)
	addQuant("lm_head", vocab, hidden)
	weights["model.layers.0.input_layernorm.weight"] = seqArray(0.02, int(hidden))
	weights["model.layers.0.post_attention_layernorm.weight"] = seqArray(0.03, int(hidden))
	weights["model.layers.0.self_attn.q_norm.weight"] = seqArray(0.04, 32)
	weights["model.layers.0.self_attn.k_norm.weight"] = seqArray(0.05, 32)
	weights["model.norm.weight"] = seqArray(0.06, int(hidden))
	defer freeArrayMap(weights)

	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3(dir)
	if err != nil {
		t.Fatalf("LoadQwen3(quantized) error = %v", err)
	}
	defer closeQwen3(model)

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != bits || info.QuantGroup != gs {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, bits, gs)
	}
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved")
	}
}

// TestModel_LoadQwen3_MoEConfigGuard_Bad confirms the dense loader rejects a
// MoE config (num_experts present, not hybrid) with the "not implemented"
// guard at qwen3.go rather than attempting a dense load.
func TestModel_LoadQwen3_MoEConfigGuard_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"vocab_size": 100,
		"max_position_embeddings": 4096,
		"num_experts": 4,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 32
	}`)

	_, err := LoadQwen3(dir)
	if err == nil {
		t.Fatal("expected MoE-not-implemented guard error from LoadQwen3")
	}
	if !core.Contains(err.Error(), "sparse expert") {
		t.Fatalf("error = %v, want sparse-expert not-implemented guard", err)
	}
}

func writeMinimalConfig(t *testing.T, dir string, modelType string) {
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

// --- synthetic dense-model harness ---------------------------------------
//
// qwen3TinyModel hand-builds a runnable Qwen3Model at the qwen3 bench geometry
// (hidden=256, 4 query heads, 2 K/V heads, SwiGLU inter=512) without a model
// load — the AX-11-clean fixture style of qwen3_bench_test.go and close_test.go,
// extended with the Cfg (VocabSize, Scale, RopeTheta) the model-level Forward
// needs so a synthetic [B,L] token array flows embed → layers → norm → output.

const (
	tinyVocab  = 16
	tinyHidden = qwen3BenchHidden  // 256
	tinyHeads  = qwen3BenchHeads   // 4
	tinyKV     = qwen3BenchKVHeads // 2
	tinyHeadD  = qwen3BenchHeadDim // 64
	tinyInter  = qwen3BenchInter   // 512
	tinyLayers = 2
)

// qwen3TinyModel returns a 2-layer Qwen3Model with materialised synthetic
// weights and a configured Cfg. Free it with closeQwen3 (or CloseModel).
func qwen3TinyModel(t testing.TB) *Qwen3Model {
	t.Helper()
	requireMetalRuntime(t)

	embedW := seqArray(0.01, tinyVocab, tinyHidden)
	normW := seqArray(0.02, tinyHidden)
	outW := seqArray(0.03, tinyVocab, tinyHidden)
	metal.Materialize(embedW, normW, outW)

	layers := make([]*metal.DenseDecoderLayer, tinyLayers)
	for i := range layers {
		layers[i] = qwen3BenchLayer()
	}

	cfg := qwen3BenchCfg()
	cfg.VocabSize = tinyVocab
	cfg.NumHiddenLayers = tinyLayers
	cfg.MaxPositionEmbeddings = 64

	return &Qwen3Model{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Layers:      layers,
		Norm:        &metal.RMSNormModule{Weight: normW},
		Output:      metal.NewLinear(outW, nil),
		Cfg:         cfg,
		modelType:   "qwen3",
	}
}

// TestQwen3_Qwen3Model_Forward_Good drives the real trunk path: a synthetic
// [B=1, L=3] token array flows embed → 2 dense decoder layers → final norm →
// output projection, producing logits [1, 3, vocab]. No model load.
func TestQwen3_Qwen3Model_Forward_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	tokens := metal.FromValues([]int32{1, 2, 3}, 1, 3)
	defer metal.Free(tokens)
	caches := m.NewCache()
	defer metal.FreeCaches(caches)

	out := m.Forward(tokens, caches)
	defer metal.Free(out)
	metal.Materialize(out)

	got := out.Shape()
	if len(got) != 3 || got[0] != 1 || got[1] != 3 || got[2] != tinyVocab {
		t.Fatalf("Forward out shape = %v, want [1 3 %d]", got, tinyVocab)
	}
}

// TestQwen3_Qwen3Model_ForwardMasked_Good drives the masked path with an
// explicit additive causal mask [B,1,L,L] — the branch Forward (mask=nil)
// skips.
func TestQwen3_Qwen3Model_ForwardMasked_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	const L = 3
	tokens := metal.FromValues([]int32{1, 2, 3}, 1, L)
	defer metal.Free(tokens)

	// Lower-triangular additive mask: 0 on/below the diagonal, -inf above.
	maskVals := make([]float32, L*L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			if j > i {
				maskVals[i*L+j] = float32(math.Inf(-1))
			}
		}
	}
	mask := metal.FromValues(maskVals, 1, 1, L, L)
	defer metal.Free(mask)

	caches := m.NewCache()
	defer metal.FreeCaches(caches)

	out := m.ForwardMasked(tokens, mask, caches)
	defer metal.Free(out)
	metal.Materialize(out)

	got := out.Shape()
	if len(got) != 3 || got[0] != 1 || got[1] != L || got[2] != tinyVocab {
		t.Fatalf("ForwardMasked out shape = %v, want [1 %d %d]", got, L, tinyVocab)
	}
}

// TestQwen3_Qwen3Model_Forward_DecodeStep_Ugly drives the single-token decode
// regime: a warm prefill populates the KV cache, then a follow-on L==1 step
// appends to it — the steady-state generation path with cache continuity.
func TestQwen3_Qwen3Model_Forward_DecodeStep_Ugly(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	caches := m.NewCache()
	defer metal.FreeCaches(caches)

	prefill := metal.FromValues([]int32{1, 2, 3, 4}, 1, 4)
	defer metal.Free(prefill)
	pre := m.Forward(prefill, caches)
	metal.Materialize(pre)
	metal.Free(pre)

	step := metal.FromValues([]int32{5}, 1, 1)
	defer metal.Free(step)
	out := m.Forward(step, caches)
	defer metal.Free(out)
	metal.Materialize(out)

	got := out.Shape()
	if len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != tinyVocab {
		t.Fatalf("decode-step out shape = %v, want [1 1 %d]", got, tinyVocab)
	}
}

// TestQwen3_Qwen3Model_NewCache_Good builds per-layer KV caches sized to the
// layer count, each a *metal.KVCache (Qwen 3 is global attention only).
func TestQwen3_Qwen3Model_NewCache_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	caches := m.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) != tinyLayers {
		t.Fatalf("NewCache len = %d, want %d", len(caches), tinyLayers)
	}
	for i, c := range caches {
		if _, ok := c.(*metal.KVCache); !ok {
			t.Fatalf("cache[%d] = %T, want *metal.KVCache", i, c)
		}
	}
}

// TestQwen3_Qwen3Model_SplitAccessors_Good exercises the split-inference
// accessors (SplitEmbedding / SplitDecoderLayers / SplitNorm / SplitOutput /
// SplitConfig) — they return the model's own composed modules unchanged.
func TestQwen3_Qwen3Model_SplitAccessors_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	if m.SplitEmbedding() != m.EmbedTokens {
		t.Error("SplitEmbedding did not return EmbedTokens")
	}
	if len(m.SplitDecoderLayers()) != tinyLayers {
		t.Errorf("SplitDecoderLayers len = %d, want %d", len(m.SplitDecoderLayers()), tinyLayers)
	}
	if m.SplitNorm() != m.Norm {
		t.Error("SplitNorm did not return Norm")
	}
	if m.SplitOutput() != m.Output {
		t.Error("SplitOutput did not return Output")
	}
	if m.SplitConfig() != m.Cfg {
		t.Error("SplitConfig did not return Cfg")
	}
}

// TestQwen3_Qwen3Model_NumQueryHeads_Good reports the configured query-head
// count; TestQwen3_Qwen3Model_NumQueryHeads_Bad covers the nil-Cfg guard.
func TestQwen3_Qwen3Model_NumQueryHeads_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)
	if got := m.NumQueryHeads(); got != tinyHeads {
		t.Fatalf("NumQueryHeads = %d, want %d", got, tinyHeads)
	}
}

func TestQwen3_Qwen3Model_NumQueryHeads_Bad(t *testing.T) {
	m := &Qwen3Model{} // Cfg == nil
	if got := m.NumQueryHeads(); got != 0 {
		t.Fatalf("NumQueryHeads with nil Cfg = %d, want 0", got)
	}
}

// TestQwen3_Qwen3Model_ResolveLoRALinear_Good resolves each LoRA-targetable
// projection by path on a real layer; _Bad covers out-of-range index; _Ugly
// covers an unknown projection path.
func TestQwen3_Qwen3Model_ResolveLoRALinear_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	layer := m.Layers[0]
	cases := map[string]*metal.Linear{
		"self_attn.q_proj": layer.Attention.QProj,
		"self_attn.k_proj": layer.Attention.KProj,
		"self_attn.v_proj": layer.Attention.VProj,
		"self_attn.o_proj": layer.Attention.OProj,
		"mlp.gate_proj":    layer.MLP.GateProj,
		"mlp.up_proj":      layer.MLP.UpProj,
		"mlp.down_proj":    layer.MLP.DownProj,
	}
	for path, want := range cases {
		if got := m.ResolveLoRALinear(0, path); got != want {
			t.Errorf("ResolveLoRALinear(0, %q) = %p, want %p", path, got, want)
		}
	}
}

func TestQwen3_Qwen3Model_ResolveLoRALinear_Bad(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)
	if got := m.ResolveLoRALinear(99, "self_attn.q_proj"); got != nil {
		t.Fatalf("ResolveLoRALinear(out-of-range) = %p, want nil", got)
	}
}

func TestQwen3_Qwen3Model_ResolveLoRALinear_Ugly(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)
	if got := m.ResolveLoRALinear(0, "self_attn.nonexistent_proj"); got != nil {
		t.Fatalf("ResolveLoRALinear(unknown path) = %p, want nil", got)
	}
}

// TestQwen3_Qwen3Model_Tokenizer_Good returns the model's configured tokenizer.
func TestQwen3_Qwen3Model_Tokenizer_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)
	tok := &metal.Tokenizer{}
	m.Tok = tok
	if m.Tokenizer() != tok {
		t.Fatal("Tokenizer did not return the configured tokenizer")
	}
}

// TestQwen3_Qwen3Model_FillModelInfo_Good copies config metadata into a
// ModelInfo; _Ugly covers the quantized branch (QuantBits/QuantGroup).
func TestQwen3_Qwen3Model_FillModelInfo_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.VocabSize != tinyVocab || info.HiddenSize != tinyHidden || info.ContextLength != 64 {
		t.Fatalf("FillModelInfo = %+v, want vocab=%d hidden=%d ctx=64", info, tinyVocab, tinyHidden)
	}
	if info.QuantBits != 0 || info.QuantGroup != 0 {
		t.Fatalf("FillModelInfo quant = %d/%d, want 0/0 for unquantized", info.QuantBits, info.QuantGroup)
	}
}

func TestQwen3_Qwen3Model_FillModelInfo_Ugly(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)
	m.Cfg.Quantization = &metal.QuantizationConfig{Bits: 4, GroupSize: 64}

	info := &metal.ModelInfo{}
	m.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("FillModelInfo quant = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
	}
}

// TestQwen3_Qwen3Model_ModelType_Good returns the architecture identifier.
func TestQwen3_Qwen3Model_ModelType_Good(t *testing.T) {
	m := &Qwen3Model{modelType: "llama"}
	if got := m.ModelType(); got != "llama" {
		t.Fatalf("ModelType = %q, want llama", got)
	}
}

// TestQwen3_Qwen3Model_CloseModel_Good drives CloseModel (the exported wrapper
// over closeQwen3) and confirms it frees the held weights.
func TestQwen3_Qwen3Model_CloseModel_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	embedW := m.EmbedTokens.Weight
	outW := m.Output.Weight

	m.CloseModel()

	if embedW.Valid() {
		t.Error("CloseModel did not free embed weight")
	}
	if outW.Valid() {
		t.Error("CloseModel did not free output weight")
	}
}

// TestQwen3_Qwen3Model_ApplyLoRA_Good wraps attention + MLP projections with
// LoRA adapters on a real 2-layer model and confirms the adapter map is keyed
// by the full projection path.
func TestQwen3_Qwen3Model_ApplyLoRA_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	adapter := m.ApplyLoRA(metal.LoRAConfig{
		Rank:         4,
		Alpha:        8,
		TargetLayers: []string{"q_proj", "down_proj"},
	})

	// 2 targets × 2 layers = 4 wrapped projections.
	if len(adapter.Layers) != 4 {
		t.Fatalf("ApplyLoRA wrapped %d projections, want 4", len(adapter.Layers))
	}
	if _, ok := adapter.Layers["model.layers.0.self_attn.q_proj"]; !ok {
		t.Error("missing LoRA adapter for layer 0 q_proj")
	}
	if _, ok := adapter.Layers["model.layers.1.mlp.down_proj"]; !ok {
		t.Error("missing LoRA adapter for layer 1 down_proj")
	}
	if m.Layers[0].Attention.QProj.LoRA == nil {
		t.Error("q_proj.LoRA not attached after ApplyLoRA")
	}
}

// TestQwen3_Qwen3Model_ApplyLoRA_AllTargets_Good wraps every attention and MLP
// projection on the 2-layer synthetic model — driving each switch arm of
// ApplyLoRA (k/v/o_proj and gate/up_proj) the q_proj+down_proj case leaves
// untouched.
func TestQwen3_Qwen3Model_ApplyLoRA_AllTargets_Good(t *testing.T) {
	m := qwen3TinyModel(t)
	defer closeQwen3(m)

	adapter := m.ApplyLoRA(metal.LoRAConfig{
		Rank:         4,
		Alpha:        8,
		TargetLayers: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
	})

	// 7 targets × 2 layers = 14 wrapped projections.
	if len(adapter.Layers) != 14 {
		t.Fatalf("ApplyLoRA wrapped %d projections, want 14 (all targets, 2 layers)", len(adapter.Layers))
	}
	for _, key := range []string{
		"model.layers.0.self_attn.k_proj",
		"model.layers.0.self_attn.v_proj",
		"model.layers.0.self_attn.o_proj",
		"model.layers.1.mlp.gate_proj",
		"model.layers.1.mlp.up_proj",
	} {
		if _, ok := adapter.Layers[key]; !ok {
			t.Errorf("missing LoRA adapter for %s", key)
		}
	}
}
