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
	if (&MixtralModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty MixtralModel.MoETextRuntimeAvailable() = true, want false")
	}
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

// TestModel_FillModelInfo_Good copies sizing out of the loaded config.
func TestModel_FillModelInfo_Good(t *testing.T) {
	model := loadMixtralModel(t)
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != mixtralLoadVocab || info.HiddenSize != mixtralLoadHidden {
		t.Fatalf("FillModelInfo = vocab %d hidden %d, want %d/%d", info.VocabSize, info.HiddenSize, mixtralLoadVocab, mixtralLoadHidden)
	}
	if info.ContextLength != 32 {
		t.Fatalf("ContextLength = %d, want 32", info.ContextLength)
	}
}

// TestModel_FillModelInfo_Quantized_Good exercises the quantization branch:
// a config carrying a QuantizationConfig must report its bits/group into
// ModelInfo (the loaded-model fixture is unquantized, so this drives the branch
// directly off a hand-built config).
func TestModel_FillModelInfo_Quantized_Good(t *testing.T) {
	model := &MixtralModel{Cfg: &MixtralConfig{
		VocabSize:             5,
		HiddenSize:            8,
		MaxPositionEmbeddings: 32,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("FillModelInfo quant = %d-bit/group %d, want 4/64", info.QuantBits, info.QuantGroup)
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

// TestModel_ApplyLoRA_DenseMLPTarget_Ugly targets gate_proj — only the dense
// layer 0 has an MLP, so the MoE layer 1 contributes no gate_proj adapter. The
// adapter must still build, with exactly the dense layer's projection wired.
func TestModel_ApplyLoRA_DenseMLPTarget_Ugly(t *testing.T) {
	model := loadMixtralModel(t)
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"gate_proj"},
	})
	// Only layer 0 is dense → exactly one gate_proj adapter; layer 1 (MoE) skipped.
	if len(adapter.Layers) != 1 {
		t.Fatalf("adapter.Layers = %d, want 1 (only the dense layer has an MLP)", len(adapter.Layers))
	}
	if _, ok := adapter.Layers["model.layers.0.mlp.gate_proj"]; !ok {
		t.Fatalf("adapter keyed %v, want model.layers.0.mlp.gate_proj", keys(adapter.Layers))
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

// TestModel_CloseModel_Good releases a loaded model and asserts the layer slice
// is cleared. A second CloseModel and a nil-receiver CloseModel must not panic
// (idempotent teardown). This test owns its own model so it can close early
// rather than leaning on the shared t.Cleanup.
func TestModel_CloseModel_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralModel(t, dir)
	model, err := LoadMixtral(dir)
	if err != nil {
		t.Fatalf("LoadMixtral: %v", err)
	}
	if len(model.Layers) == 0 {
		t.Fatal("loaded model has no layers")
	}
	model.CloseModel()
	if model.Layers != nil {
		t.Fatalf("after CloseModel Layers = %v, want nil", model.Layers)
	}
	model.CloseModel()           // idempotent: second close is a no-op
	(*MixtralModel)(nil).CloseModel() // nil receiver must not panic
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
