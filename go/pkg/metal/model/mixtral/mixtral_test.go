// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	"math"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// requireMetalRuntime skips the test unless the Metal runtime is both built in
// (-tags metal_runtime) and available on this host — the gptoss/qwen3 gate.
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// --- LoadMixtral error paths ---

func TestModel_LoadMixtral_MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "mixtral",
		"hidden_size": 1024,
		"num_hidden_layers": 1,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000
	}`)

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Errorf("error should mention tokenizer, got: %v", err)
	}
}

func TestModel_LoadMixtral_InvalidConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "not json")

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
}

func TestModel_LoadMixtral_NoSafetensors_Bad(t *testing.T) {
	dir := t.TempDir()
	writeMinimalMixtralConfig(t, dir)
	writeMinimalMixtralTokenizer(t, dir)

	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for missing safetensors files")
	}
	if !core.Contains(err.Error(), "mixtral") {
		t.Errorf("error should mention mixtral, got: %v", err)
	}
}

// --- parseMixtralConfig ---

func TestModel_ParseMixtralConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseMixtralConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("parseMixtralConfig: %v", err)
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
	if cfg.NumLocalExperts != 0 {
		t.Errorf("NumLocalExperts at parse = %d, want 0 (dimension not fabricated — derived from the routed-expert tensors at load)", cfg.NumLocalExperts)
	}
	if cfg.NumExpertsPerTok != 2 {
		t.Errorf("NumExpertsPerTok default = %d, want 2", cfg.NumExpertsPerTok)
	}
	// head_dim inferred from hidden/heads when absent.
	if cfg.HeadDim != 128 {
		t.Errorf("HeadDim inferred = %d, want 128", cfg.HeadDim)
	}
}

func TestModel_ParseMixtralConfig_QuantizationNested_Good(t *testing.T) {
	cfg, err := parseMixtralConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 8,
		"head_dim": 128,
		"quantization_config": {"group_size": 64, "bits": 4}
	}`))
	if err != nil {
		t.Fatalf("parseMixtralConfig: %v", err)
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

func TestModel_ParseMixtralConfig_InvalidJSON_Bad(t *testing.T) {
	_, err := parseMixtralConfig([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// TestModel_ParseMixtralConfig_NestedQuantMismatch_Bad: top-level JSON valid for
// the flat config (which has no quantization field) but with quantization_config
// typed as a non-object — the second unmarshal into the *metal.QuantizationConfig
// wrapper fails, the "parse nested config" error arm.
func TestModel_ParseMixtralConfig_NestedQuantMismatch_Bad(t *testing.T) {
	for _, bad := range []string{
		`{"hidden_size":8,"num_attention_heads":2,"quantization_config":"notanobject"}`,
		`{"hidden_size":8,"num_attention_heads":2,"quantization":[1,2]}`,
	} {
		if _, err := parseMixtralConfig([]byte(bad)); err == nil {
			t.Errorf("expected nested-config parse error for %s", bad)
		}
	}
}

// --- mixtralMoELayerMask ---

func TestModel_MixtralMoELayerMask_Good(t *testing.T) {
	// SparseStep<=0 → every layer is MoE.
	dense := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 3, SparseStep: 0})
	for i, m := range dense {
		if !m {
			t.Errorf("SparseStep=0 layer %d = %v, want true", i, m)
		}
	}
	// SparseStep=2 → only every 2nd layer (i%2==1) is MoE.
	stepped := mixtralMoELayerMask(&MixtralConfig{NumHiddenLayers: 4, SparseStep: 2})
	want := []bool{false, true, false, true}
	for i := range want {
		if stepped[i] != want[i] {
			t.Errorf("SparseStep=2 layer %d = %v, want %v", i, stepped[i], want[i])
		}
	}
}

// --- MoETextRuntimeAvailable (relocated from package metal) ---

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	m := &MixtralModel{
		Layers: []*MixtralDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &MixtralMoEBlock{
				Router:        router,
				Experts:       []*MixtralExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !m.MoETextRuntimeAvailable() {
		t.Fatal("MixtralModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := m.MoETextDecodeFamily(); got != "mixtral" {
		t.Fatalf("MoETextDecodeFamily() = %q, want mixtral", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	// nil receiver → false (the m == nil guard).
	if (*MixtralModel)(nil).MoETextRuntimeAvailable() {
		t.Fatal("nil MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
	// Empty (non-nil) model with no layers → false (zero-length layer walk).
	if (&MixtralModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
	// A nil layer entry → the parts closure returns empty parts (OK=false) → false.
	nilLayer := &MixtralModel{Layers: []*MixtralDecoderLayer{nil}}
	if nilLayer.MoETextRuntimeAvailable() {
		t.Fatal("nil-layer MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
	// A layer whose dense parts are unpopulated → false.
	incomplete := &MixtralModel{Layers: []*MixtralDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
}

// --- LoadMixtral success path + loaded-model methods ---
//
// A synthetic on-disk model (config.json + tokenizer.json + real safetensors,
// the gptoss precedent) drives the whole load chain that the error-path tests
// above never reach: the dense/MoE layer interleave (decoder_sparse_step=2),
// the router/expert/switch loaders, mixtralInferNumExperts reading the expert
// count from the tensors, and the lm_head branch. The loaded model then backs
// the Forward/ForwardMasked/NewCache/FillModelInfo/CloseModel method tests so
// they run against real weights rather than a hand-stubbed struct. Dims are
// tiny so the Metal compute is cheap; the test skips without -tags metal_runtime.

const mixtralLoadConfigJSON = `{
	"model_type": "mixtral",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 2,
	"num_key_value_heads": 2,
	"head_dim": 4,
	"vocab_size": 5,
	"max_position_embeddings": 32,
	"rms_norm_eps": 1e-5,
	"rope_theta": 1000000,
	"decoder_sparse_step": 2,
	"num_experts_per_tok": 2
}`

const (
	mixtralLoadHidden  = 8
	mixtralLoadInter   = 16
	mixtralLoadVocab   = 5
	mixtralLoadHeadDim = 4
	mixtralLoadHeads   = 2
	mixtralLoadKVHeads = 2
	mixtralLoadExperts = 2
)

// writeMixtralModel writes config.json, tokenizer.json and a safetensors file
// for the mixtralLoadConfigJSON geometry: layer 0 dense (SiLU MLP), layer 1 MoE
// (router + 2 experts, w1/w2/w3). num_local_experts is deliberately omitted from
// the config so LoadMixtral derives it from the routed-expert tensors here.
func writeMixtralModel(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), mixtralLoadConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalMixtralTokenizer(t, dir)

	const (
		h     = mixtralLoadHidden
		inter = mixtralLoadInter
		v     = mixtralLoadVocab
		hd    = mixtralLoadHeadDim
		nh    = mixtralLoadHeads
		kvh   = mixtralLoadKVHeads
	)
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight": mixtralSeqArr(0.01, v, h),
		"model.norm.weight":         mixtralSeqArr(0.11, h),
		"lm_head.weight":            mixtralSeqArr(0.12, v, h),
	}
	// Both layers carry attention + the two layernorms.
	for _, l := range []int{0, 1} {
		p := core.Sprintf("model.layers.%d", l)
		weights[p+".input_layernorm.weight"] = mixtralSeqArr(0.02, h)
		weights[p+".post_attention_layernorm.weight"] = mixtralSeqArr(0.03, h)
		weights[p+".self_attn.q_proj.weight"] = mixtralSeqArr(0.04, nh*hd, h)
		weights[p+".self_attn.k_proj.weight"] = mixtralSeqArr(0.05, kvh*hd, h)
		weights[p+".self_attn.v_proj.weight"] = mixtralSeqArr(0.06, kvh*hd, h)
		weights[p+".self_attn.o_proj.weight"] = mixtralSeqArr(0.07, h, nh*hd)
	}
	// Layer 0 → dense SiLU MLP (decoder_sparse_step=2 makes i%2==0 dense).
	weights["model.layers.0.mlp.gate_proj.weight"] = mixtralSeqArr(0.08, inter, h)
	weights["model.layers.0.mlp.up_proj.weight"] = mixtralSeqArr(0.09, inter, h)
	weights["model.layers.0.mlp.down_proj.weight"] = mixtralSeqArr(0.10, h, inter)
	// Layer 1 → MoE block_sparse_moe router + 2 experts (Mixtral w1/w3/w2 naming).
	weights["model.layers.1.block_sparse_moe.gate.weight"] = mixtralSeqArr(0.20, mixtralLoadExperts, h)
	for e := 0; e < mixtralLoadExperts; e++ {
		p := core.Sprintf("model.layers.1.block_sparse_moe.experts.%d", e)
		weights[p+".w1.weight"] = mixtralSeqArr(0.30+float32(e)*0.03, inter, h) // gate
		weights[p+".w3.weight"] = mixtralSeqArr(0.31+float32(e)*0.03, inter, h) // up
		weights[p+".w2.weight"] = mixtralSeqArr(0.32+float32(e)*0.03, h, inter) // down
	}
	defer mixtralFreeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
}

// loadMixtralModel loads the synthetic mixed dense+MoE model and registers its
// cleanup; the returned model is the shared fixture for the method tests.
func loadMixtralModel(t *testing.T) *MixtralModel {
	t.Helper()
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralModel(t, dir)
	model, err := LoadMixtral(dir)
	if err != nil {
		t.Fatalf("LoadMixtral(mixed dense+MoE) error = %v", err)
	}
	t.Cleanup(model.CloseModel)
	return model
}

func TestModel_LoadMixtral_Good(t *testing.T) {
	model := loadMixtralModel(t)

	if model.ModelType() != "mixtral" {
		t.Fatalf("ModelType() = %q, want mixtral", model.ModelType())
	}
	if model.NumLayers() != 2 {
		t.Fatalf("NumLayers() = %d, want 2", model.NumLayers())
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want loaded tokenizer")
	}
	if model.Cfg.VocabSize != mixtralLoadVocab || model.Cfg.HiddenSize != mixtralLoadHidden {
		t.Fatalf("Cfg = vocab %d hidden %d, want vocab=%d hidden=%d", model.Cfg.VocabSize, model.Cfg.HiddenSize, mixtralLoadVocab, mixtralLoadHidden)
	}
	if model.EmbedTokens == nil || model.EmbedTokens.Weight == nil {
		t.Fatal("EmbedTokens.Weight = nil, want loaded embedding")
	}
	if model.Output == nil || model.Output.Weight == nil {
		t.Fatal("Output.Weight = nil, want loaded lm_head")
	}
	// num_local_experts omitted from config → derived from the routed-expert
	// tensors by mixtralInferNumExperts.
	if model.Cfg.NumLocalExperts != mixtralLoadExperts {
		t.Fatalf("NumLocalExperts = %d, want %d (derived from expert tensors)", model.Cfg.NumLocalExperts, mixtralLoadExperts)
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
	if n := len(model.Layers[1].MoE.Experts); n != mixtralLoadExperts {
		t.Fatalf("layer 1 experts = %d, want %d", n, mixtralLoadExperts)
	}
	if model.Layers[1].MoE.Router == nil || model.Layers[1].MoE.Router.Weight == nil {
		t.Fatal("layer 1 MoE.Router.Weight = nil, want loaded router")
	}
	if model.Layers[1].MoE.SwitchExperts == nil {
		t.Fatal("layer 1 MoE.SwitchExperts = nil, want batched SwiGLU experts")
	}
}

// mixtralQuantConfigJSON omits vocab_size (so the loader derives it from the
// quantized embed tensor) and declares a top-level "quantization" block — the
// q != nil arms of LoadMixtral the bf16 fixture never reaches.
const mixtralQuantConfigJSON = `{
	"model_type": "mixtral",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 2,
	"num_key_value_heads": 2,
	"head_dim": 4,
	"max_position_embeddings": 32,
	"rms_norm_eps": 1e-5,
	"decoder_sparse_step": 2,
	"num_experts_per_tok": 2,
	"quantization": {"bits": 4, "group_size": 64}
}`

// writeMixtralQuantizedModel writes a 4-bit quantized mixed dense+MoE checkpoint:
// metal.Quantize packs each projection (and the embedding + lm_head + MoE router)
// into the (weight, scales, biases) triplet the loader's quantized `linear` /
// `mixtralLoadRouter` closures resolve. Geometry matches mixtralQuantConfigJSON.
// vocab_size is omitted from the config so the loader derives it from the
// quantized embed tensor. Group size 64 needs the inner dim ≥ 64, so the matvec
// dims are widened to 64 here (the layernorms stay tiny, unquantized).
func writeMixtralQuantizedModel(t *testing.T, dir string) {
	t.Helper()
	const (
		h      = 64
		inter  = 128
		v      = 64
		hd     = 32
		nh     = 2
		kvh    = 2
		gs     = 64
		bits   = 4
		expert = mixtralLoadExperts
	)
	cfg := `{
		"model_type": "mixtral",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-5,
		"decoder_sparse_step": 2,
		"num_experts_per_tok": 2,
		"quantization": {"bits": 4, "group_size": 64}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalMixtralTokenizer(t, dir)

	weights := map[string]*metal.Array{
		"model.norm.weight": mixtralSeqArr(0.11, h),
	}
	addQuant := func(name string, out, in int32) {
		dense := mixtralSeqArr(0.02, int(out), int(in))
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
		weights[p+".input_layernorm.weight"] = mixtralSeqArr(0.02, h)
		weights[p+".post_attention_layernorm.weight"] = mixtralSeqArr(0.03, h)
		addQuant(p+".self_attn.q_proj", nh*hd, h)
		addQuant(p+".self_attn.k_proj", kvh*hd, h)
		addQuant(p+".self_attn.v_proj", kvh*hd, h)
		addQuant(p+".self_attn.o_proj", h, nh*hd)
	}
	// Layer 0 → dense quantized SiLU MLP.
	addQuant("model.layers.0.mlp.gate_proj", inter, h)
	addQuant("model.layers.0.mlp.up_proj", inter, h)
	addQuant("model.layers.0.mlp.down_proj", h, inter)
	// Layer 1 → MoE: quantized router (drives the q != nil arm of
	// mixtralLoadRouter) + experts. Experts load via NewLinear (mixtralLoadExpert
	// does not quantize) so their tensors stay dense.
	addQuant("model.layers.1.block_sparse_moe.gate", expert, h)
	for e := 0; e < expert; e++ {
		p := core.Sprintf("model.layers.1.block_sparse_moe.experts.%d", e)
		weights[p+".w1.weight"] = mixtralSeqArr(0.30+float32(e)*0.03, inter, h)
		weights[p+".w3.weight"] = mixtralSeqArr(0.31+float32(e)*0.03, inter, h)
		weights[p+".w2.weight"] = mixtralSeqArr(0.32+float32(e)*0.03, h, inter)
	}
	defer mixtralFreeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
}

// TestModel_LoadMixtral_Quantized_Good loads the synthetic 4-bit quantized mixed
// dense+MoE checkpoint. It drives the q != nil arms the bf16 fixture skips: the
// quantized `linear` closure (NewQuantizedLinear), the quantized embedding
// scales/biases, vocab_size derived from the quantized embed tensor, the
// quantized lm_head branch, and the quantized router arm of mixtralLoadRouter.
// No live model.
func TestModel_LoadMixtral_Quantized_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralQuantizedModel(t, dir)

	model, err := LoadMixtral(dir)
	if err != nil {
		t.Fatalf("LoadMixtral(quantized) error = %v", err)
	}
	t.Cleanup(model.CloseModel)

	// vocab_size omitted from config → derived from the quantized embed tensor.
	if model.Cfg.VocabSize != 64 {
		t.Fatalf("VocabSize = %d, want 64 (derived from quantized embed tensor)", model.Cfg.VocabSize)
	}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("quant info = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
	}
	// Quantized embedding → scales resolved (drives the embed.Scales arm).
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved")
	}
	// Quantized lm_head → distinct Output Linear with scales (not the tied embed).
	if model.Output == nil || model.Output.Scales == nil {
		t.Error("quantized lm_head scales were not resolved")
	}
	// Quantized router on the MoE layer → router carries scales.
	if r := model.Layers[1].MoE.Router; r == nil || r.Scales == nil {
		t.Error("quantized router scales were not resolved")
	}
}

// TestModel_LoadMixtral_TiedEmbedding_Good loads a checkpoint with NO lm_head
// tensor: LoadMixtral must fall back to the tied embedding (Output =
// EmbedTokens.AsLinear()), so Output.Weight aliases the embed weight. CloseModel
// must then NOT double-free the shared weight (covered by the close test).
func TestModel_LoadMixtral_TiedEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralModelNoLMHead(t, dir)

	model, err := LoadMixtral(dir)
	if err != nil {
		t.Fatalf("LoadMixtral(tied) error = %v", err)
	}
	t.Cleanup(model.CloseModel)

	if model.Output == nil {
		t.Fatal("Output = nil, want tied embedding linear")
	}
	if model.Output.Weight != model.EmbedTokens.Weight {
		t.Fatal("Output.Weight does not alias EmbedTokens.Weight — tied fallback not taken")
	}
}

// writeMixtralModelNoLMHead writes the bf16 fixture geometry but omits the
// lm_head tensor, forcing the tied-embedding fallback in LoadMixtral.
func writeMixtralModelNoLMHead(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), mixtralLoadConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalMixtralTokenizer(t, dir)
	const (
		h     = mixtralLoadHidden
		inter = mixtralLoadInter
		v     = mixtralLoadVocab
		hd    = mixtralLoadHeadDim
		nh    = mixtralLoadHeads
		kvh   = mixtralLoadKVHeads
	)
	weights := map[string]*metal.Array{
		"model.embed_tokens.weight": mixtralSeqArr(0.01, v, h),
		"model.norm.weight":         mixtralSeqArr(0.11, h),
		// no lm_head.weight → tied embedding fallback
	}
	for _, l := range []int{0, 1} {
		p := core.Sprintf("model.layers.%d", l)
		weights[p+".input_layernorm.weight"] = mixtralSeqArr(0.02, h)
		weights[p+".post_attention_layernorm.weight"] = mixtralSeqArr(0.03, h)
		weights[p+".self_attn.q_proj.weight"] = mixtralSeqArr(0.04, nh*hd, h)
		weights[p+".self_attn.k_proj.weight"] = mixtralSeqArr(0.05, kvh*hd, h)
		weights[p+".self_attn.v_proj.weight"] = mixtralSeqArr(0.06, kvh*hd, h)
		weights[p+".self_attn.o_proj.weight"] = mixtralSeqArr(0.07, h, nh*hd)
	}
	weights["model.layers.0.mlp.gate_proj.weight"] = mixtralSeqArr(0.08, inter, h)
	weights["model.layers.0.mlp.up_proj.weight"] = mixtralSeqArr(0.09, inter, h)
	weights["model.layers.0.mlp.down_proj.weight"] = mixtralSeqArr(0.10, h, inter)
	weights["model.layers.1.block_sparse_moe.gate.weight"] = mixtralSeqArr(0.20, mixtralLoadExperts, h)
	for e := 0; e < mixtralLoadExperts; e++ {
		p := core.Sprintf("model.layers.1.block_sparse_moe.experts.%d", e)
		weights[p+".w1.weight"] = mixtralSeqArr(0.30+float32(e)*0.03, inter, h)
		weights[p+".w3.weight"] = mixtralSeqArr(0.31+float32(e)*0.03, inter, h)
		weights[p+".w2.weight"] = mixtralSeqArr(0.32+float32(e)*0.03, h, inter)
	}
	defer mixtralFreeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
}

// TestModel_LoadMixtral_MissingConfig_Bad: an empty model dir (no config.json)
// fails at the config read, the first LoadMixtral error arm.
func TestModel_LoadMixtral_MissingConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	_, err := LoadMixtral(dir)
	if err == nil {
		t.Fatal("expected error for missing config.json")
	}
	if !core.Contains(err.Error(), "config") {
		t.Errorf("error should mention config, got: %v", err)
	}
}

// TestModel_Forward_Good drives the full model Forward over a short prompt: the
// embedding, both decoder layers (dense layer 0 + MoE layer 1), the final norm
// and the lm_head projection, force-Eval'd. The logits must be [B, L, vocab].
func TestModel_Forward_Good(t *testing.T) {
	model := loadMixtralModel(t)
	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	defer metal.Free(tokens)

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	logits := model.Forward(tokens, caches)
	if logits == nil {
		t.Fatal("Forward returned nil")
	}
	defer metal.Free(logits)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}
	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != mixtralLoadVocab {
		t.Fatalf("Forward logits shape = %v, want [1 3 %d]", shape, mixtralLoadVocab)
	}
}

// TestModel_ForwardMasked_Good drives ForwardMasked with an explicit causal mask
// and asserts the same logit geometry — the masked entry point Forward delegates
// to.
func TestModel_ForwardMasked_Good(t *testing.T) {
	model := loadMixtralModel(t)
	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	defer metal.Free(tokens)

	// Lower-triangular additive causal mask: row i attends to keys 0..i.
	const L = 3
	negInf := float32(math.Inf(-1))
	maskData := make([]float32, L*L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			if j > i {
				maskData[i*L+j] = negInf
			}
		}
	}
	mask := metal.FromValues(maskData, 1, 1, L, L)
	defer metal.Free(mask)

	caches := model.NewCache()
	defer metal.FreeCaches(caches)

	logits := model.ForwardMasked(tokens, mask, caches)
	if logits == nil {
		t.Fatal("ForwardMasked returned nil")
	}
	defer metal.Free(logits)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval(logits): %v", err)
	}
	shape := logits.Shape()
	if len(shape) != 3 || shape[2] != mixtralLoadVocab {
		t.Fatalf("ForwardMasked logits shape = %v, want [.. .. %d]", shape, mixtralLoadVocab)
	}
}

// TestModel_DecoderLayerForward_DiagnosticFallback_Ugly drives the diagnostic
// fallback in mixtralDecoderLayerForward: a MoE layer (isMoELayer() true) where
// MoESwiGLUForward returns ok=false. Forcing NumExpertsPerTok=0 makes the
// router top-k non-positive → ok=false → the layer falls back to the residual
// add (h + normed2) instead of panicking or dropping the layer. The output keeps
// the input geometry. This is the inspectable-until-every-sparse-path-is-enabled
// safety net, exercised directly off the loaded MoE layer.
func TestModel_DecoderLayerForward_DiagnosticFallback_Ugly(t *testing.T) {
	model := loadMixtralModel(t)
	if !model.Layers[1].isMoELayer() {
		t.Fatal("layer 1 is not MoE — fixture changed; fallback test needs a MoE layer")
	}

	const B, L = int32(1), int32(2)
	x := mixtralSeqArr(0.05, int(B), int(L), mixtralLoadHidden)
	defer metal.Free(x)
	cache := metal.NewKVCache()
	defer metal.FreeCaches([]metal.Cache{cache})

	// Copy the config with top-k forced to 0 → MoESwiGLUForward ok=false.
	cfg := *model.Cfg
	cfg.NumExpertsPerTok = 0

	out := mixtralDecoderLayerForward(model.Layers[1], x, cache, B, L, nil, &cfg)
	if out == nil {
		t.Fatal("mixtralDecoderLayerForward (fallback) returned nil")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(fallback out): %v", err)
	}
	shape := out.Shape()
	if len(shape) != 3 || shape[0] != B || shape[1] != L || shape[2] != mixtralLoadHidden {
		t.Fatalf("fallback out shape = %v, want [%d %d %d]", shape, B, L, mixtralLoadHidden)
	}
}

// TestModel_DecoderLayerForward_MoESuccess_Good drives the native selected-expert
// MoE decode body of mixtralDecoderLayerForward (the ok=true arm of
// MoESwiGLUForward): the router top-k kernel requires a single-token decode step
// (router scores [1,1,experts]), so this runs the loaded MoE layer with B=1, L=1
// and the real top-k. The output keeps the [1,1,hidden] geometry. (The full
// Forward_Good test uses L=3 prefill, where the native decode kernel declines and
// the layer takes the diagnostic fallback instead — only the single-token decode
// reaches this success arm without a live model.)
func TestModel_DecoderLayerForward_MoESuccess_Good(t *testing.T) {
	model := loadMixtralModel(t)
	if !model.Layers[1].isMoELayer() {
		t.Fatal("layer 1 is not MoE — fixture changed; MoE-success test needs a MoE layer")
	}

	const B, L = int32(1), int32(1)
	x := mixtralSeqArr(0.05, int(B), int(L), mixtralLoadHidden)
	defer metal.Free(x)
	cache := metal.NewKVCache()
	defer metal.FreeCaches([]metal.Cache{cache})

	out := mixtralDecoderLayerForward(model.Layers[1], x, cache, B, L, nil, model.Cfg)
	if out == nil {
		t.Fatal("mixtralDecoderLayerForward (MoE success) returned nil")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(MoE-success out): %v", err)
	}
	shape := out.Shape()
	if len(shape) != 3 || shape[0] != B || shape[1] != L || shape[2] != mixtralLoadHidden {
		t.Fatalf("MoE-success out shape = %v, want [%d %d %d]", shape, B, L, mixtralLoadHidden)
	}
}

// TestModel_NewCache_Good asserts NewCache builds one KVCache per layer on a
// real loaded model.
func TestModel_NewCache_Good(t *testing.T) {
	model := loadMixtralModel(t)
	caches := model.NewCache()
	defer metal.FreeCaches(caches)
	if len(caches) != model.NumLayers() {
		t.Fatalf("NewCache() length = %d, want layer count %d", len(caches), model.NumLayers())
	}
	for i, c := range caches {
		if _, ok := c.(*metal.KVCache); !ok {
			t.Fatalf("cache[%d] = %T, want *metal.KVCache", i, c)
		}
	}
}

// TestModel_ApplyLoRA_Good authors a LoRA adapter over the attention projections
// of the loaded model. Both layers carry attention, so q_proj+v_proj across two
// layers yields four adapted projections, each wired back onto its Linear.
func TestModel_ApplyLoRA_Good(t *testing.T) {
	model := loadMixtralModel(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	})
	if adapter == nil {
		t.Fatal("ApplyLoRA returned nil")
	}
	// 2 layers × {q,k,v,o}_proj = 8 adapted attention projections.
	if len(adapter.Layers) != 8 {
		t.Fatalf("adapter.Layers = %d, want 8 (2 layers × q/k/v/o_proj)", len(adapter.Layers))
	}
	if model.Layers[0].Dense.Attention.QProj.LoRA == nil {
		t.Fatal("layer 0 QProj.LoRA = nil, want adapter wired onto the projection")
	}
	if model.Layers[1].Dense.Attention.OProj.LoRA == nil {
		t.Fatal("layer 1 OProj.LoRA = nil, want adapter wired onto every attention projection")
	}
	for _, want := range []string{
		"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
		"model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
	} {
		if _, ok := adapter.Layers[want]; !ok {
			t.Fatalf("adapter keyed %v, missing %s", keys(adapter.Layers), want)
		}
	}
}

// TestModel_ApplyLoRA_DenseMLPTarget_Ugly targets all three MLP projections —
// only the dense layer 0 has an MLP, so the MoE layer 1 contributes none. The
// adapter must still build, with exactly the dense layer's gate/up/down wired
// (this drives the gate_proj / up_proj / down_proj switch arms of ApplyLoRA).
func TestModel_ApplyLoRA_DenseMLPTarget_Ugly(t *testing.T) {
	model := loadMixtralModel(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"gate_proj", "up_proj", "down_proj"},
	})
	// Only layer 0 is dense → exactly three MLP adapters; layer 1 (MoE) skipped.
	if len(adapter.Layers) != 3 {
		t.Fatalf("adapter.Layers = %d, want 3 (only the dense layer has an MLP: gate/up/down)", len(adapter.Layers))
	}
	for _, want := range []string{
		"model.layers.0.mlp.gate_proj",
		"model.layers.0.mlp.up_proj",
		"model.layers.0.mlp.down_proj",
	} {
		if _, ok := adapter.Layers[want]; !ok {
			t.Fatalf("adapter keyed %v, missing %s", keys(adapter.Layers), want)
		}
	}
	// Each adapted projection is wired back onto its Linear.
	if model.Layers[0].Dense.MLP.UpProj.LoRA == nil || model.Layers[0].Dense.MLP.DownProj.LoRA == nil {
		t.Fatal("up_proj/down_proj LoRA not wired onto the dense MLP projections")
	}
}

// TestModel_MoETextRuntimeAvailable_Loaded_Good asserts the real loaded model
// reports its native MoE decode runtime as available across all layers.
func TestModel_MoETextRuntimeAvailable_Loaded_Good(t *testing.T) {
	model := loadMixtralModel(t)
	if !model.MoETextRuntimeAvailable() {
		t.Fatal("loaded MixtralModel.MoETextRuntimeAvailable() = false, want true")
	}
}

// --- mixtralToQwen3Config ---

func TestModel_MixtralToQwen3Config_Good(t *testing.T) {
	cfg := &MixtralConfig{
		HiddenSize:            8,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		NumKeyValueHeads:      2,
		HeadDim:               4,
		VocabSize:             5,
		RMSNormEps:            1e-5,
		MaxPositionEmbeddings: 32,
		RopeTheta:             1000000,
		Scale:                 1.0,
	}
	dc := mixtralToQwen3Config(cfg)
	if dc == nil {
		t.Fatal("mixtralToQwen3Config returned nil for a valid config")
	}
	if dc.HiddenSize != 8 || dc.NumAttentionHeads != 2 || dc.HeadDim != 4 {
		t.Fatalf("DenseConfig = hidden %d heads %d headdim %d, want 8/2/4", dc.HiddenSize, dc.NumAttentionHeads, dc.HeadDim)
	}
	if dc.RopeTheta != 1000000 || dc.Scale != 1.0 {
		t.Fatalf("DenseConfig rope %f scale %f, want 1000000/1.0", dc.RopeTheta, dc.Scale)
	}
}

func TestModel_MixtralToQwen3Config_Nil_Bad(t *testing.T) {
	if dc := mixtralToQwen3Config(nil); dc != nil {
		t.Fatalf("mixtralToQwen3Config(nil) = %+v, want nil", dc)
	}
}

// --- mixtralInferNumExperts ---

// TestModel_MixtralInferNumExperts_NoTensors_Bad: a MoE layer mask with no
// expert tensors present yields 0 — the loader never fabricates a count.
func TestModel_MixtralInferNumExperts_NoTensors_Bad(t *testing.T) {
	got := mixtralInferNumExperts(map[string]*metal.Array{}, []bool{true})
	if got != 0 {
		t.Fatalf("mixtralInferNumExperts(empty) = %d, want 0", got)
	}
	// All-dense mask → the loop never finds a MoE layer, also 0.
	if got := mixtralInferNumExperts(map[string]*metal.Array{}, []bool{false, false}); got != 0 {
		t.Fatalf("mixtralInferNumExperts(all-dense) = %d, want 0", got)
	}
}

// --- mixtralLoadRouter ---

// TestModel_MixtralLoadRouter_NoRouter_Ugly: weights with none of the router
// tensor names (.gate/.router/.gate_proj) yield an empty router (Weight nil)
// rather than a nil pointer — the fall-through return at the end of the loop.
func TestModel_MixtralLoadRouter_NoRouter_Ugly(t *testing.T) {
	router := mixtralLoadRouter(map[string]*metal.Array{}, 0, nil)
	if router == nil {
		t.Fatal("mixtralLoadRouter(empty) = nil, want empty non-nil router")
	}
	if router.Weight != nil {
		t.Fatalf("mixtralLoadRouter(empty).Weight = %v, want nil", router.Weight)
	}
}

// TestModel_MixtralLoadRouter_Quantized_Ugly: with a router .gate tensor plus
// scales present and a non-nil quant config, the router carries the group/bits
// from the config — the q != nil arm of mixtralLoadRouter.
func TestModel_MixtralLoadRouter_Quantized_Ugly(t *testing.T) {
	requireMetalRuntime(t)
	w := mixtralSeqArr(0.2, 2, 4)
	sc := mixtralSeqArr(0.01, 2, 1)
	defer metal.Free(w, sc)
	weights := map[string]*metal.Array{
		"model.layers.0.block_sparse_moe.gate.weight": w,
		"model.layers.0.block_sparse_moe.gate.scales": sc,
	}
	router := mixtralLoadRouter(weights, 0, &metal.QuantizationConfig{Bits: 4, GroupSize: 64})
	if router.Weight == nil {
		t.Fatal("router.Weight = nil, want resolved gate weight")
	}
	if router.Scales == nil {
		t.Fatal("router.Scales = nil, want resolved gate scales")
	}
	if router.GroupSize != 64 || router.Bits != 4 {
		t.Fatalf("router quant = group %d/bits %d, want 64/4", router.GroupSize, router.Bits)
	}
}

// --- mixtralSwitchExperts ---

// TestModel_MixtralSwitchExperts_NilExpert_Ugly: a nil entry in the experts
// slice short-circuits to ok=false rather than dereferencing it.
func TestModel_MixtralSwitchExperts_NilExpert_Ugly(t *testing.T) {
	if _, ok := mixtralSwitchExperts([]*MixtralExpert{nil}); ok {
		t.Fatal("mixtralSwitchExperts([nil]) ok = true, want false")
	}
}

// --- helpers ---

func keys(m map[string]*metal.LoRALinear) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func mixtralSeqArr(start float32, shape ...int) *metal.Array {
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

func mixtralFreeArrayMap(arrays map[string]*metal.Array) {
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

func writeMinimalMixtralConfig(t *testing.T, dir string) {
	t.Helper()
	config := `{
		"model_type": "mixtral",
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

func writeMinimalMixtralTokenizer(t *testing.T, dir string) {
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
