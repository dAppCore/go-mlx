// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Coverage companions for qwen3_moe.go — they drive the registered-loader
// dispatch arm, the quantized load closures, the config-guard branches, the
// num_experts==0 expert-count fallback, the runnable MoE decode branch of
// qwen3MoEDecoderLayerForward, and the shared-expert close path. All synthetic
// (SaveSafetensors + temp-dir configs + hand-built runnable parts), no live
// model — the AX-11-clean fixture style of moe_model_test.go.

package qwen3

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

// TestQwen3MoE_LoaderDispatch_Bad routes a qwen3_moe config with no tokenizer
// through metal.LoadAndInit: the dispatch resolves the qwen3_moe family to the
// closure registered in qwen3_moe.go init(); LoadQwen3MoE fails inside it and the
// closure wraps the failure with "load qwen3_moe native model" — the dispatch
// arm's error path that the direct LoadQwen3MoE fixtures bypass.
func TestQwen3MoE_LoaderDispatch_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"vocab_size": 5,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	// no tokenizer.json → LoadQwen3MoE fails inside the registered closure.
	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(qwen3_moe with no tokenizer) error = nil, want a wrapped loader failure")
	}
}

// TestQwen3MoE_LoadConfigMissing_Bad covers the config-read failure arm of
// LoadQwen3MoE (no config.json in the directory).
func TestQwen3MoE_LoadConfigMissing_Bad(t *testing.T) {
	if _, err := LoadQwen3MoE(t.TempDir()); err == nil {
		t.Fatal("LoadQwen3MoE(no config) error = nil, want a config-read failure")
	}
}

// TestQwen3MoE_LoadConfigInvalid_Bad covers the parse-config failure arm
// (malformed JSON never reaches the MoE-metadata checks).
func TestQwen3MoE_LoadConfigInvalid_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), "{broken")
	if _, err := LoadQwen3MoE(dir); err == nil {
		t.Fatal("LoadQwen3MoE(malformed config) error = nil, want a parse failure")
	}
}

// TestQwen3MoE_LoadHybridGuard_Bad covers the hybrid-config guard arm: a
// qwen3_6_moe (hybrid linear-attention) config must be refused by the qwen3_moe
// native loader rather than loaded as a plain sparse model.
func TestQwen3MoE_LoadHybridGuard_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_6_moe",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"num_experts": 8,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 32,
		"layer_types": ["linear_attention", "full_attention"]
	}`)
	_, err := LoadQwen3MoE(dir)
	if err == nil {
		t.Fatal("LoadQwen3MoE(qwen3_6_moe hybrid) error = nil, want the hybrid guard")
	}
	if !core.Contains(err.Error(), "qwen3_6_moe") {
		t.Fatalf("error = %v, want the qwen3_6_moe hybrid guard", err)
	}
}

// TestQwen3MoE_LoadNotMoE_Bad covers the missing-MoE-metadata guard arm: a
// plain-qwen3 config (model_type not _moe, no expert counts) is not a sparse
// model, so LoadQwen3MoE must refuse it before any weight load. IsMoE() keys on
// expert counts OR a _moe model_type, so the config uses neither.
func TestQwen3MoE_LoadNotMoE_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"vocab_size": 5
	}`)
	writeMinimalTokenizer(t, dir)
	_, err := LoadQwen3MoE(dir)
	if err == nil {
		t.Fatal("LoadQwen3MoE(no MoE metadata) error = nil, want the MoE-metadata guard")
	}
	if !core.Contains(err.Error(), "MoE metadata") {
		t.Fatalf("error = %v, want the MoE-metadata guard", err)
	}
}

// TestQwen3MoE_LoadMissingTokenizer_Bad covers the tokenizer-load failure arm:
// a valid MoE config with no tokenizer.json.
func TestQwen3MoE_LoadMissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"vocab_size": 5,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	_, err := LoadQwen3MoE(dir)
	if err == nil {
		t.Fatal("LoadQwen3MoE(no tokenizer) error = nil, want a tokenizer-load failure")
	}
	if !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want a tokenizer-load failure", err)
	}
}

// addQuantMoE packs a dense [out,in] weight into its quantized (weight, scales,
// biases) triplet under the base name + suffixes, mirroring the dense-quant
// fixture in qwen3_test.go.
func addQuantMoE(t *testing.T, weights map[string]*metal.Array, name string, out, in int32, gs, bits int) {
	t.Helper()
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

// TestQwen3MoE_LoadQuantized_Good loads a synthetic 4-bit quantized MoE
// checkpoint: every projection (attention, the gate router, both experts) plus
// the embedding and lm_head are packed into quantized triplets. This drives the
// q != nil branches of LoadQwen3MoE — the quantized linear closure, the quantized
// embedding, the quantized router (qwen3MoELoadRouter), and the quantized
// lm_head — that the bf16 fixtures skip. No live model.
func TestQwen3MoE_LoadQuantized_Good(t *testing.T) {
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
		"model_type": "qwen3_moe",
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 32,
		"vocab_size": 64,
		"max_position_embeddings": 4096,
		"rms_norm_eps": 1e-6,
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 128,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	weights := map[string]*metal.Array{}
	addQuantMoE(t, weights, "model.embed_tokens", vocab, hidden, gs, bits)
	addQuantMoE(t, weights, "model.layers.0.self_attn.q_proj", hidden, hidden, gs, bits)
	addQuantMoE(t, weights, "model.layers.0.self_attn.k_proj", hidden, hidden, gs, bits)
	addQuantMoE(t, weights, "model.layers.0.self_attn.v_proj", hidden, hidden, gs, bits)
	addQuantMoE(t, weights, "model.layers.0.self_attn.o_proj", hidden, hidden, gs, bits)
	addQuantMoE(t, weights, "model.layers.0.mlp.gate", 2, hidden, gs, bits) // quantized router
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		addQuantMoE(t, weights, p+".gate_proj", inter, hidden, gs, bits)
		addQuantMoE(t, weights, p+".up_proj", inter, hidden, gs, bits)
		addQuantMoE(t, weights, p+".down_proj", hidden, inter, gs, bits)
	}
	addQuantMoE(t, weights, "lm_head", vocab, hidden, gs, bits)
	weights["model.layers.0.input_layernorm.weight"] = seqArray(0.02, int(hidden))
	weights["model.layers.0.post_attention_layernorm.weight"] = seqArray(0.03, int(hidden))
	weights["model.norm.weight"] = seqArray(0.04, int(hidden))
	defer freeArrayMap(weights)

	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3MoE(dir)
	if err != nil {
		t.Fatalf("LoadQwen3MoE(quantized) error = %v", err)
	}
	defer closeQwen3MoE(model)

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != bits || info.QuantGroup != gs {
		t.Fatalf("quant info = %d/%d, want %d/%d", info.QuantBits, info.QuantGroup, bits, gs)
	}
	if model.EmbedTokens.Scales == nil {
		t.Error("quantized embedding scales were not resolved")
	}
	layer := model.Layers[0]
	if layer.MoE == nil || layer.MoE.Router == nil || layer.MoE.Router.Scales == nil {
		t.Error("quantized router scales were not resolved on the sparse layer")
	}
}

// TestQwen3MoE_LoadVocabFromEmbed_NoExpertsConfig_Good drops both vocab_size and
// num_experts from a bf16 MoE config: the loader backfills vocab from the embed
// weight rows (the cfg.VocabSize == 0 branch) and counts experts from the weight
// triplets via qwen3MoECountExperts (the numExperts == 0 fallback). The tied
// lm_head (no lm_head.weight) also takes the embedding-as-linear branch.
func TestQwen3MoE_LoadVocabFromEmbed_NoExpertsConfig_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	// no vocab_size, no num_experts — both must be derived from the weights.
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"decoder_sparse_step": 1,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	delete(weights, "lm_head.weight") // force the tied-output fallback
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadQwen3MoE(dir)
	if err != nil {
		t.Fatalf("LoadQwen3MoE(no vocab/experts) error = %v", err)
	}
	defer closeQwen3MoE(model)

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 5 {
		t.Fatalf("VocabSize = %d, want 5 backfilled from the embed weight rows", info.VocabSize)
	}
	layer := model.Layers[0]
	if layer.MoE == nil || len(layer.MoE.Experts) != 2 {
		t.Fatalf("experts = %d, want 2 counted from the weight triplets", len(layer.MoE.Experts))
	}
	// tied output: Output must alias the embedding weight.
	if model.Output.Weight != model.EmbedTokens.Weight {
		t.Error("with no lm_head.weight, MoE Output must alias the embedding weight")
	}
}

// qwen3MoERunnableBlock hand-builds a runnable MoE block at hidden=4, inter=8,
// 2 experts: a router weight [experts, hidden] plus batched switch experts, in
// the moeReadyRuntimeParts style — enough for MoESwiGLUForward to return ok=true.
func qwen3MoERunnableBlock(t *testing.T) (*Qwen3MoEBlock, func()) {
	t.Helper()
	const hidden, inter, experts = 4, 8, 2
	routerWeight := seqArray(0.05, experts, hidden)
	metal.Materialize(routerWeight)

	gate := make([]*metal.Linear, experts)
	up := make([]*metal.Linear, experts)
	down := make([]*metal.Linear, experts)
	for e := 0; e < experts; e++ {
		gate[e] = metal.NewLinear(seqArray(0.06+float32(e)*0.01, inter, hidden), nil)
		up[e] = metal.NewLinear(seqArray(0.07+float32(e)*0.01, inter, hidden), nil)
		down[e] = metal.NewLinear(seqArray(0.08+float32(e)*0.01, hidden, inter), nil)
	}
	switchExperts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		t.Fatal("NewMoESwiGLUExpertsFromLinears() ok = false, want true")
	}
	block := &Qwen3MoEBlock{
		Router:        &metal.MoERouter{Weight: routerWeight},
		SwitchExperts: switchExperts,
		Experts:       []*Qwen3MoEExpert{{GateProj: gate[0]}, {GateProj: gate[1]}},
	}
	cleanup := func() {
		metal.Free(routerWeight)
		metal.FreeMoESwiGLUExperts(switchExperts)
	}
	return block, cleanup
}

// TestQwen3MoE_DecoderLayerForward_MoEBranch_Good drives the runnable sparse
// branch of qwen3MoEDecoderLayerForward: attention residual → post-norm →
// MoESwiGLUForward returns ok=true → its output is added to the residual (the
// h + mlpOut path the diagnostic-fallback test never reaches). Output shape must
// hold at [B,L,hidden].
func TestQwen3MoE_DecoderLayerForward_MoEBranch_Good(t *testing.T) {
	requireMetalRuntime(t)

	block, cleanup := qwen3MoERunnableBlock(t)
	defer cleanup()
	layer := &Qwen3MoEDecoderLayer{Dense: qwen3MoETinyAttnLayer(), MoE: block}
	cfg := qwen3MoETinyCfg()

	x := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, 1, 2, 4)
	defer metal.Free(x)
	metal.Materialize(x)
	c := metal.NewKVCache()
	defer c.Reset()

	out := qwen3MoEDecoderLayerForward(layer, x, c, 1, 2, nil, cfg)
	defer metal.Free(out)
	metal.Materialize(out)

	got := out.Shape()
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 4 {
		t.Fatalf("MoE-branch out shape = %v, want [1 2 4]", got)
	}
}

// TestQwen3MoE_Generate_Quantized_Ugly drives an end-to-end generate over a
// synthetic quantized MoE checkpoint via LoadAndInit + Generate — the dispatch
// closure's success path plus the runtime decode loop. Two tokens must emerge
// with no error.
func TestQwen3MoE_Generate_Quantized_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000,
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 2,
		"moe_intermediate_size": 16
	}`)
	writeMinimalTokenizer(t, dir)

	weights := tinyMoEDecoderWeights(8, 16, 2, 5)
	for e := range 2 {
		p := core.Sprintf("model.layers.0.mlp.experts.%d", e)
		weights[p+".gate_proj.weight"] = seqArray(0.30+float32(e)*0.03, 16, 8)
		weights[p+".up_proj.weight"] = seqArray(0.31+float32(e)*0.03, 16, 8)
		weights[p+".down_proj.weight"] = seqArray(0.32+float32(e)*0.03, 8, 16)
	}
	weights["model.layers.0.mlp.gate.weight"] = seqArray(0.20, 2, 8)
	defer freeArrayMap(weights)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := metal.LoadAndInit(dir, metal.LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(qwen3_moe dispatch) error = %v", err)
	}
	defer model.Close()

	var genCount int
	for range model.Generate(context.Background(), "hello", metal.GenerateConfig{MaxTokens: 2}) {
		genCount++
	}
	if genCount != 2 {
		t.Fatalf("generated %d token(s), want 2", genCount)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

// TestQwen3MoE_CloseSharedExpert_Good closes a MoE model whose sparse layer has a
// populated shared expert, driving closeQwen3MoE's shared-expert free path (and
// the per-expert free loop) that the tied-output close test does not reach. It
// also includes a nil layer entry to drive the nil-layer continue guard.
func TestQwen3MoE_CloseSharedExpert_Good(t *testing.T) {
	requireMetalRuntime(t)

	embed := &metal.Embedding{Weight: seqArray(0.01, 5, 8)}
	norm := &metal.RMSNormModule{Weight: seqArray(0.02, 8)}
	out := metal.NewLinear(seqArray(0.03, 5, 8), nil)
	metal.Materialize(embed.Weight, norm.Weight, out.Weight)

	sharedGate := metal.NewLinear(seqArray(0.10, 8, 8), nil)
	sharedUp := metal.NewLinear(seqArray(0.11, 8, 8), nil)
	sharedDown := metal.NewLinear(seqArray(0.12, 8, 8), nil)
	expGate := metal.NewLinear(seqArray(0.13, 16, 8), nil)
	expUp := metal.NewLinear(seqArray(0.14, 16, 8), nil)
	expDown := metal.NewLinear(seqArray(0.15, 8, 16), nil)
	metal.Materialize(sharedGate.Weight, sharedUp.Weight, sharedDown.Weight, expGate.Weight, expUp.Weight, expDown.Weight)

	sparse := &Qwen3MoEDecoderLayer{
		Dense: &metal.DenseDecoderLayer{},
		MoE: &Qwen3MoEBlock{
			Router: &metal.MoERouter{},
			SharedExpert: &Qwen3MoESharedExpert{
				GateProj: sharedGate, UpProj: sharedUp, DownProj: sharedDown,
			},
			Experts: []*Qwen3MoEExpert{{GateProj: expGate, UpProj: expUp, DownProj: expDown}},
		},
	}

	m := &Qwen3MoEModel{
		EmbedTokens: embed,
		Norm:        norm,
		Output:      out,
		Layers:      []*Qwen3MoEDecoderLayer{nil, sparse}, // nil layer drives the continue guard
	}
	closeQwen3MoE(m)

	if sharedGate.Weight.Valid() {
		t.Error("shared-expert gate weight should be freed")
	}
	if expGate.Weight.Valid() {
		t.Error("expert gate weight should be freed")
	}
	if embed.Weight.Valid() {
		t.Error("embedding weight should be freed")
	}
}
