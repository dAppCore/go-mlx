// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

func TestModel_LoadModel_Qwen3MoEFullLoad_Good(t *testing.T) {
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

	model, err := LoadQwen3MoE(dir)
	if err != nil {
		t.Fatalf("LoadQwen3MoE(qwen3_moe) error = %v", err)
	}
	if model.ModelType() != "qwen3_moe" {
		t.Fatalf("ModelType() = %q, want qwen3_moe", model.ModelType())
	}

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 5 || info.HiddenSize != 8 || info.NumLayers != 1 {
		t.Fatalf("Info() = %+v, want vocab=5 hidden=8 layers=1", info)
	}
}

func TestModel_LoadModel_Qwen3MoEModelTypeDispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "qwen3_moe",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"vocab_size": 1000,
		"max_position_embeddings": 32768,
		"num_experts": 128,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 384,
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeMinimalTokenizer(t, dir)

	_, err := LoadQwen3MoE(dir)
	if err == nil {
		t.Fatal("expected weight-loading error for qwen3_moe without safetensors")
	}
	if !core.Contains(err.Error(), "qwen3_moe") {
		t.Fatalf("error = %v, should contain qwen3_moe", err)
	}
}

// Kimi full-load coverage travels with the model in package metal/model/kimi.
// Mixtral full-load coverage travels with the model in package
// metal/model/mixtral.

func TestModel_Generate_Qwen3MoEDiagnostic_Good(t *testing.T) {
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
		"decoder_sparse_step": 0,
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
		t.Fatalf("LoadAndInit(qwen3_moe) error = %v", err)
	}
	defer model.Close()

	var genCount int
	for range model.Generate(context.Background(), "hello", metal.GenerateConfig{MaxTokens: 2}) {
		genCount++
	}
	if genCount != 2 {
		t.Fatalf("generated %d token(s), want 2 with native sparse-expert decode", genCount)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil after native sparse-expert decode", err)
	}
}

func TestModel_MoETextRuntimeAvailable_Good(t *testing.T) {
	router, experts, cleanup := moeReadyRuntimeParts(t)
	defer cleanup()

	model := &Qwen3MoEModel{
		Layers: []*Qwen3MoEDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{},
			MoE: &Qwen3MoEBlock{
				Router:        router,
				Experts:       []*Qwen3MoEExpert{{}},
				SwitchExperts: experts,
			},
		}},
	}
	if !model.MoETextRuntimeAvailable() {
		t.Fatal("Qwen3MoEModel.MoETextRuntimeAvailable() = false, want true")
	}
	if got := model.MoETextDecodeFamily(); got != "qwen3_moe" {
		t.Fatalf("MoETextDecodeFamily() = %q, want qwen3_moe", got)
	}
}

func TestModel_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&Qwen3MoEModel{}).MoETextRuntimeAvailable() {
		t.Fatal("empty Qwen3MoEModel.MoETextRuntimeAvailable() = true, want false")
	}
	incomplete := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}
	if incomplete.MoETextRuntimeAvailable() {
		t.Fatal("incomplete Qwen3MoEModel.MoETextRuntimeAvailable() = true, want false")
	}
}

// TestModel_Qwen3MoEModel_NumLayers_Good reports the layer count.
func TestModel_Qwen3MoEModel_NumLayers_Good(t *testing.T) {
	m := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{nil, nil, nil}}
	if got := m.NumLayers(); got != 3 {
		t.Fatalf("NumLayers = %d, want 3", got)
	}
}

// TestModel_Qwen3MoEModel_ApplyLoRA_Good wraps the attention projections (and
// the dense-layer MLP projections) of a small hand-built MoE model. Layer 0 is
// dense (MLP set), layer 1 is sparse (MoE block) — so gate_proj resolves only
// on the dense layer, while q_proj resolves on both.
func TestModel_Qwen3MoEModel_ApplyLoRA_Good(t *testing.T) {
	requireMetalRuntime(t)

	denseLayer := &Qwen3MoEDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			Attention: &metal.GQAAttention{
				QProj: metal.NewLinear(seqArray(0.01, 4, 4), nil),
				KProj: metal.NewLinear(seqArray(0.02, 4, 4), nil),
				VProj: metal.NewLinear(seqArray(0.03, 4, 4), nil),
				OProj: metal.NewLinear(seqArray(0.04, 4, 4), nil),
			},
			MLP: &metal.SiLUMLP{
				GateProj: metal.NewLinear(seqArray(0.05, 4, 4), nil),
				UpProj:   metal.NewLinear(seqArray(0.06, 4, 4), nil),
				DownProj: metal.NewLinear(seqArray(0.07, 4, 4), nil),
			},
		},
	}
	sparseLayer := &Qwen3MoEDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			Attention: &metal.GQAAttention{
				QProj: metal.NewLinear(seqArray(0.11, 4, 4), nil),
				KProj: metal.NewLinear(seqArray(0.12, 4, 4), nil),
				VProj: metal.NewLinear(seqArray(0.13, 4, 4), nil),
				OProj: metal.NewLinear(seqArray(0.14, 4, 4), nil),
			},
		},
		MoE: &Qwen3MoEBlock{Router: &metal.MoERouter{}, Experts: []*Qwen3MoEExpert{{}}},
	}
	m := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{denseLayer, sparseLayer}}
	defer closeQwen3MoE(m)

	adapter := m.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj", "gate_proj"},
	})

	// q_proj on both layers (2) + gate_proj only on the dense layer 0 (1) = 3.
	if len(adapter.Layers) != 3 {
		t.Fatalf("ApplyLoRA wrapped %d projections, want 3", len(adapter.Layers))
	}
	if _, ok := adapter.Layers["model.layers.0.self_attn.q_proj"]; !ok {
		t.Error("missing LoRA for dense layer 0 q_proj")
	}
	if _, ok := adapter.Layers["model.layers.1.self_attn.q_proj"]; !ok {
		t.Error("missing LoRA for sparse layer 1 q_proj")
	}
	if _, ok := adapter.Layers["model.layers.0.mlp.gate_proj"]; !ok {
		t.Error("missing LoRA for dense layer 0 gate_proj")
	}
	if _, ok := adapter.Layers["model.layers.1.mlp.gate_proj"]; ok {
		t.Error("sparse layer 1 has no dense MLP — gate_proj must not resolve")
	}
}

// TestModel_Qwen3MoEModel_ApplyLoRA_AllTargets_Good wraps every attention
// target (q/k/v/o_proj) plus every dense-MLP target (gate/up/down_proj) on a
// single dense MoE layer — driving each switch arm of the MoE ApplyLoRA that
// the q_proj+gate_proj case leaves untouched.
func TestModel_Qwen3MoEModel_ApplyLoRA_AllTargets_Good(t *testing.T) {
	requireMetalRuntime(t)

	layer := &Qwen3MoEDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			Attention: &metal.GQAAttention{
				QProj: metal.NewLinear(seqArray(0.01, 4, 4), nil),
				KProj: metal.NewLinear(seqArray(0.02, 4, 4), nil),
				VProj: metal.NewLinear(seqArray(0.03, 4, 4), nil),
				OProj: metal.NewLinear(seqArray(0.04, 4, 4), nil),
			},
			MLP: &metal.SiLUMLP{
				GateProj: metal.NewLinear(seqArray(0.05, 4, 4), nil),
				UpProj:   metal.NewLinear(seqArray(0.06, 4, 4), nil),
				DownProj: metal.NewLinear(seqArray(0.07, 4, 4), nil),
			},
		},
	}
	m := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{layer}}
	defer closeQwen3MoE(m)

	adapter := m.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
	})
	if len(adapter.Layers) != 7 {
		t.Fatalf("ApplyLoRA wrapped %d projections, want 7 (all targets)", len(adapter.Layers))
	}
	for _, key := range []string{
		"model.layers.0.self_attn.k_proj",
		"model.layers.0.self_attn.v_proj",
		"model.layers.0.self_attn.o_proj",
		"model.layers.0.mlp.up_proj",
		"model.layers.0.mlp.down_proj",
	} {
		if _, ok := adapter.Layers[key]; !ok {
			t.Errorf("missing LoRA adapter for %s", key)
		}
	}
}

// TestModel_MoETextRuntimeAvailable_NilLayer_Bad covers the nil-layer arm of
// MoETextRuntimeAvailable's per-layer probe (a Layers slice with a nil entry):
// the missing-parts path reports unavailable rather than panicking.
func TestModel_MoETextRuntimeAvailable_NilLayer_Bad(t *testing.T) {
	m := &Qwen3MoEModel{Layers: []*Qwen3MoEDecoderLayer{nil}}
	if m.MoETextRuntimeAvailable() {
		t.Fatal("MoETextRuntimeAvailable with a nil layer = true, want false")
	}
}

// TestModel_Qwen3MoELoadRouter_Good resolves a router weight under the first of
// the alternative gate/router names; _Bad returns an empty (Weight==nil) router
// when none of the candidate names are present.
func TestModel_Qwen3MoELoadRouter_Good(t *testing.T) {
	weights := map[string]*metal.Array{
		"model.layers.0.mlp.gate.weight": seqArray(0.2, 2, 8),
	}
	defer freeArrayMap(weights)
	router := qwen3MoELoadRouter(weights, 0, nil)
	if router == nil || router.Weight == nil {
		t.Fatal("qwen3MoELoadRouter did not resolve the gate.weight router")
	}
}

func TestModel_Qwen3MoELoadRouter_Bad(t *testing.T) {
	router := qwen3MoELoadRouter(map[string]*metal.Array{}, 0, nil)
	if router == nil {
		t.Fatal("qwen3MoELoadRouter should return a non-nil (empty) router")
	}
	if router.Weight != nil {
		t.Error("router.Weight should be nil when no candidate gate/router weight exists")
	}
}

// TestModel_Qwen3MoELoadSharedExpert_Good builds the shared expert from its
// gate/up/down weights; _Bad returns nil when the gate weight is absent (no
// shared expert in this checkpoint).
func TestModel_Qwen3MoELoadSharedExpert_Good(t *testing.T) {
	weights := map[string]*metal.Array{
		"model.layers.0.mlp.shared_expert.gate_proj.weight": seqArray(0.1, 4, 4),
		"model.layers.0.mlp.shared_expert.up_proj.weight":   seqArray(0.2, 4, 4),
		"model.layers.0.mlp.shared_expert.down_proj.weight": seqArray(0.3, 4, 4),
	}
	defer freeArrayMap(weights)
	w := func(name string) *metal.Array { return metal.ResolveWeight(weights, name) }
	shared := qwen3MoELoadSharedExpert(w, 0)
	if shared == nil || shared.GateProj == nil {
		t.Fatal("qwen3MoELoadSharedExpert did not build the shared expert")
	}
}

func TestModel_Qwen3MoELoadSharedExpert_Bad(t *testing.T) {
	w := func(string) *metal.Array { return nil }
	if shared := qwen3MoELoadSharedExpert(w, 0); shared != nil {
		t.Fatalf("qwen3MoELoadSharedExpert(no gate) = %+v, want nil", shared)
	}
}

// TestModel_Qwen3MoESwitchExperts_Bad covers the nil-expert short circuit of
// qwen3MoESwitchExperts: a slice with a nil expert cannot form switch tensors.
func TestModel_Qwen3MoESwitchExperts_Bad(t *testing.T) {
	if _, ok := qwen3MoESwitchExperts([]*Qwen3MoEExpert{nil}); ok {
		t.Fatal("qwen3MoESwitchExperts(nil expert) ok = true, want false")
	}
}

// TestModel_Qwen3MoELayerMask_Good covers both mask regimes: decoder_sparse_step
// == 0 makes every layer after the first sparse; a positive step makes every
// step-th layer sparse.
func TestModel_Qwen3MoELayerMask_Good(t *testing.T) {
	zeroStep := &metal.DenseConfig{}
	zeroStep.NumHiddenLayers = 3
	zeroStep.DecoderSparseStep = 0
	if got := qwen3MoELayerMask(zeroStep); got[0] || !got[1] || !got[2] {
		t.Fatalf("zero-step mask = %v, want [false true true]", got)
	}

	step2 := &metal.DenseConfig{}
	step2.NumHiddenLayers = 4
	step2.DecoderSparseStep = 2
	// sparse when (i % 2) == 1 → layers 1 and 3.
	if got := qwen3MoELayerMask(step2); got[0] || !got[1] || got[2] || !got[3] {
		t.Fatalf("step-2 mask = %v, want [false true false true]", got)
	}
}

// TestModel_CloseQwen3MoE_TiedOutput_Ugly builds a MoE model whose Output
// aliases the embedding weight (tied), then closes it — the shared weight must
// be freed exactly once (closeQwen3MoE skips the aliased Output). Also covers
// the nil-model guard.
func TestModel_CloseQwen3MoE_TiedOutput_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	embed := &metal.Embedding{Weight: metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)}
	norm := &metal.RMSNormModule{Weight: metal.FromValues([]float32{1, 1}, 2)}
	metal.Materialize(embed.Weight, norm.Weight)
	m := &Qwen3MoEModel{
		EmbedTokens: embed,
		Norm:        norm,
		Output:      embed.AsLinear(), // tied: Output.Weight == embed.Weight
	}
	closeQwen3MoE(m)
	if embed.Weight.Valid() {
		t.Error("tied embedding/output weight should be freed exactly once")
	}

	closeQwen3MoE(nil) // nil guard must not panic
}

// TestModel_Qwen3MoECountExperts_Good counts experts by the .mlp.experts.N
// weight triplets (gate/up/down → /3); _Bad falls back to 32 when no expert
// weights are present.
func TestModel_Qwen3MoECountExperts_Good(t *testing.T) {
	weights := map[string]*metal.Array{
		"model.layers.0.mlp.experts.0.gate_proj.weight": seqArray(0.1, 2, 2),
		"model.layers.0.mlp.experts.0.up_proj.weight":   seqArray(0.1, 2, 2),
		"model.layers.0.mlp.experts.0.down_proj.weight": seqArray(0.1, 2, 2),
		"model.layers.0.mlp.experts.1.gate_proj.weight": seqArray(0.1, 2, 2),
		"model.layers.0.mlp.experts.1.up_proj.weight":   seqArray(0.1, 2, 2),
		"model.layers.0.mlp.experts.1.down_proj.weight": seqArray(0.1, 2, 2),
	}
	defer freeArrayMap(weights)
	if got := qwen3MoECountExperts(weights, 0); got != 2 {
		t.Fatalf("qwen3MoECountExperts = %d, want 2", got)
	}
}

func TestModel_Qwen3MoECountExperts_Bad(t *testing.T) {
	if got := qwen3MoECountExperts(map[string]*metal.Array{}, 0); got != 32 {
		t.Fatalf("qwen3MoECountExperts(empty) = %d, want 32 fallback", got)
	}
}

// qwen3MoETinyAttnLayer builds the Dense (attention + norms) half of a MoE
// decoder layer at a tiny 4-wide geometry — enough to run the shared attention
// before the MoE/dense MLP branch under test.
func qwen3MoETinyAttnLayer() *metal.DenseDecoderLayer {
	return &metal.DenseDecoderLayer{
		InputNorm:    &metal.RMSNormModule{Weight: seqArray(0.02, 4)},
		PostAttnNorm: &metal.RMSNormModule{Weight: seqArray(0.03, 4)},
		Attention: &metal.GQAAttention{
			QProj: metal.NewLinear(seqArray(0.04, 4, 4), nil),
			KProj: metal.NewLinear(seqArray(0.05, 4, 4), nil),
			VProj: metal.NewLinear(seqArray(0.06, 4, 4), nil),
			OProj: metal.NewLinear(seqArray(0.07, 4, 4), nil),
		},
	}
}

func qwen3MoETinyCfg() *metal.DenseConfig {
	cfg := &metal.DenseConfig{RopeTheta: 1000000.0, Scale: 0.5}
	cfg.HiddenSize = 4
	cfg.NumAttentionHeads = 2
	cfg.NumKeyValueHeads = 2
	cfg.HeadDim = 2
	cfg.RMSNormEps = 1e-6
	cfg.NumExpertsPerTok = 2
	return cfg
}

// TestModel_Qwen3MoEDecoderLayerForward_DenseBranch_Good drives the dense-MLP
// branch of qwen3MoEDecoderLayerForward directly (MoE == nil, Dense.MLP set):
// attention residual → post-norm → SwiGLU MLP residual, output [B,L,hidden].
func TestModel_Qwen3MoEDecoderLayerForward_DenseBranch_Good(t *testing.T) {
	requireMetalRuntime(t)

	dense := qwen3MoETinyAttnLayer()
	dense.MLP = &metal.SiLUMLP{
		GateProj: metal.NewLinear(seqArray(0.08, 8, 4), nil),
		UpProj:   metal.NewLinear(seqArray(0.09, 8, 4), nil),
		DownProj: metal.NewLinear(seqArray(0.10, 4, 8), nil),
	}
	layer := &Qwen3MoEDecoderLayer{Dense: dense}
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
		t.Fatalf("dense-branch out shape = %v, want [1 2 4]", got)
	}
}

// TestModel_Qwen3MoEDecoderLayerForward_Fallback_Ugly drives the diagnostic
// fallback: the layer is sparse (MoE.Router non-nil so it is not the dense
// branch) but the router weight is unset and SwitchExperts is nil, so
// MoESwiGLUForward returns ok=false and the layer passes the post-norm
// activation through unchanged (h + normed2). Output shape must hold.
func TestModel_Qwen3MoEDecoderLayerForward_Fallback_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	layer := &Qwen3MoEDecoderLayer{
		Dense: qwen3MoETinyAttnLayer(),
		MoE:   &Qwen3MoEBlock{Router: &metal.MoERouter{}}, // Weight nil, SwitchExperts nil
	}
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
		t.Fatalf("fallback out shape = %v, want [1 2 4]", got)
	}
}

func tinyMoEDecoderWeights(hidden, intermediate, experts, vocab int32) map[string]*metal.Array {
	h := int(hidden)
	i := int(intermediate)
	v := int(vocab)
	return map[string]*metal.Array{
		"model.embed_tokens.weight":                      seqArray(0.01, v, h),
		"model.layers.0.input_layernorm.weight":          seqArray(0.02, h),
		"model.layers.0.post_attention_layernorm.weight": seqArray(0.03, h),
		"model.layers.0.self_attn.q_proj.weight":         seqArray(0.04, h, h),
		"model.layers.0.self_attn.k_proj.weight":         seqArray(0.05, h, h),
		"model.layers.0.self_attn.v_proj.weight":         seqArray(0.06, h, h),
		"model.layers.0.self_attn.o_proj.weight":         seqArray(0.07, h, h),
		"model.layers.0.mlp.gate_proj.weight":            seqArray(0.08, i, h),
		"model.layers.0.mlp.up_proj.weight":              seqArray(0.09, i, h),
		"model.layers.0.mlp.down_proj.weight":            seqArray(0.10, h, i),
		"model.norm.weight":                              seqArray(0.11, h),
		"lm_head.weight":                                 seqArray(0.12, v, h),
	}
}

// GPT-OSS full-load coverage travels with package metal/model/gptoss.

func seqArray(start float32, shape ...int) *metal.Array {
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

func moeReadyRuntimeParts(t *testing.T) (*metal.MoERouter, *metal.MoESwiGLUExperts, func()) {
	t.Helper()
	requireMetalRuntime(t)

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

func freeArrayMap(arrays map[string]*metal.Array) {
	for _, array := range arrays {
		metal.Free(array)
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
