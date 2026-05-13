// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
	"encoding/binary"
	"math"
	"testing"
)

const miniMaxM2FixtureConfig = `{
	"architectures": ["MiniMaxM2ForCausalLM"],
	"model_type": "minimax_m2",
	"vocab_size": 200064,
	"hidden_size": 3072,
	"intermediate_size": 1536,
	"num_hidden_layers": 62,
	"num_attention_heads": 48,
	"num_key_value_heads": 8,
	"head_dim": 128,
	"max_position_embeddings": 196608,
	"num_local_experts": 256,
	"num_experts_per_tok": 8,
	"scoring_func": "sigmoid",
	"use_routing_bias": true,
	"use_mtp": true,
	"num_mtp_modules": 3,
	"mtp_transformer_layers": 1,
	"use_qk_norm": true,
	"rotary_dim": 64,
	"rope_theta": 5000000
}`

func TestMiniMaxM2_ParseConfig_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}

	if cfg.ModelType != "minimax_m2" || cfg.HiddenSize != 3072 || cfg.IntermediateSize != 1536 || cfg.NumHiddenLayers != 62 {
		t.Fatalf("shape config = %+v", cfg)
	}
	if cfg.NumLocalExperts != 256 || cfg.NumExpertsPerToken != 8 || cfg.ScoringFunc != "sigmoid" || !cfg.UseRoutingBias {
		t.Fatalf("MoE config = %+v", cfg)
	}
	if !cfg.UseMTP || cfg.NumMTPModules != 3 || cfg.MTPTransformerLayers != 1 || !cfg.UseQKNorm {
		t.Fatalf("extra config = %+v", cfg)
	}
}

func TestMiniMaxM2_TensorPlanBuildsRouterAttentionAndExpertSpecs_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	if plan.Quantization == nil || plan.Quantization.Format != "mxtq" || plan.Quantization.RoleBits[string(jang.TensorRoleRoutedExpert)] != 2 {
		t.Fatalf("plan quantization = %+v, want MXTQ routed expert profile", plan.Quantization)
	}

	specs, err := plan.LayerTensorSpecs(0, 17)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}

	router := findMiniMaxM2Spec(specs, TensorRoleRouterGate)
	if router.Name != "model.layers.0.block_sparse_moe.gate.weight" || router.Packed != nil {
		t.Fatalf("router spec = %+v, want dense router gate", router)
	}
	attention := findMiniMaxM2Spec(specs, TensorRoleAttentionQ)
	if attention.Packed == nil || attention.Packed.Bits != 8 || attention.Packed.Role != jang.TensorRoleAttention {
		t.Fatalf("attention spec = %+v, want 8-bit packed attention descriptor", attention)
	}
	if len(attention.Shape) != 2 || attention.Shape[0] != 6144 || attention.Shape[1] != 3072 {
		t.Fatalf("attention shape = %+v, want q_size x hidden_size", attention.Shape)
	}
	key := findMiniMaxM2Spec(specs, TensorRoleAttentionK)
	if len(key.Shape) != 2 || key.Shape[0] != 1024 || key.Shape[1] != 3072 {
		t.Fatalf("key shape = %+v, want kv_size x hidden_size", key.Shape)
	}
	expert := findMiniMaxM2Spec(specs, TensorRoleExpertGate)
	if expert.Name != "model.layers.0.block_sparse_moe.experts.17.gate_proj.weight" {
		t.Fatalf("expert name = %q", expert.Name)
	}
	if expert.Packed == nil || expert.Packed.Bits != 2 || expert.Packed.Role != jang.TensorRoleRoutedExpert {
		t.Fatalf("expert spec = %+v, want 2-bit routed expert descriptor", expert)
	}
	if len(expert.Aliases) == 0 || expert.Aliases[0] != "model.layers.0.mlp.experts.17.gate_proj.weight" {
		t.Fatalf("expert aliases = %+v, want mlp checkpoint alias", expert.Aliases)
	}
}

func TestMiniMaxM2_LayerForwardSkeletonValidatesAttentionAndRouter_Good(t *testing.T) {
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   4,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
		UseRoutingBias:     true,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2SkeletonRawTensors(t, plan, false))

	skeleton, err := BuildLayerForwardSkeleton(plan, []string{weights}, 0)
	if err != nil {
		t.Fatalf("BuildLayerForwardSkeleton() error = %v", err)
	}

	if skeleton.Layer != 0 || len(skeleton.Attention) != 4 {
		t.Fatalf("skeleton layer/attention = %d/%d, want 0/4", skeleton.Layer, len(skeleton.Attention))
	}
	q := findMiniMaxM2ResolvedTensor(skeleton.Attention, TensorRoleAttentionQ)
	if q.Name != "model.layers.0.self_attn.q_proj.weight" || q.PackedBytes != 16 || !sameUint64Slice(q.LogicalShape, []uint64{4, 4}) {
		t.Fatalf("q tensor = %+v, want resolved packed q projection", q)
	}
	k := findMiniMaxM2ResolvedTensor(skeleton.Attention, TensorRoleAttentionK)
	if k.PackedBytes != 8 || !sameUint64Slice(k.LogicalShape, []uint64{2, 4}) {
		t.Fatalf("k tensor = %+v, want packed kv projection", k)
	}
	if skeleton.RouterGate.Name != "model.layers.0.block_sparse_moe.gate.weight" || !sameUint64Slice(skeleton.RouterGate.Shape, []uint64{3, 4}) {
		t.Fatalf("router gate = %+v, want dense [3 4] gate", skeleton.RouterGate)
	}
	if skeleton.RouterBias == nil || !sameUint64Slice(skeleton.RouterBias.Shape, []uint64{3}) {
		t.Fatalf("router bias = %+v, want dense [3] correction bias", skeleton.RouterBias)
	}
}

func TestMiniMaxM2_LayerForwardSkeletonRejectsWrongAttentionShape_Bad(t *testing.T) {
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   4,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 4, BitsDefault: 2, AttentionBits: 8, RoutedExpertBits: 2})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2SkeletonRawTensors(t, plan, true))

	_, err = BuildLayerForwardSkeleton(plan, []string{weights}, 0)
	if err == nil || !core.Contains(err.Error(), "q_proj") || !core.Contains(err.Error(), "packed") {
		t.Fatalf("error = %v, want q_proj packed shape diagnostic", err)
	}
}

func TestMiniMaxM2_ValidateTensorNames_BadMissingExpert(t *testing.T) {
	cfg, err := ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	err = plan.ValidateTensorNames(map[string]bool{
		"model.layers.0.block_sparse_moe.gate.weight":                true,
		"model.layers.0.block_sparse_moe.e_score_correction_bias":    true,
		"model.layers.0.self_attn.q_proj.weight":                     true,
		"model.layers.0.self_attn.k_proj.weight":                     true,
		"model.layers.0.self_attn.v_proj.weight":                     true,
		"model.layers.0.self_attn.o_proj.weight":                     true,
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": true,
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight": true,
	})
	if err == nil || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want missing expert up_proj", err)
	}
}

func TestMiniMaxM2_RouteTokens_Good(t *testing.T) {
	cfg := Config{NumLocalExperts: 4, NumExpertsPerToken: 2, ScoringFunc: "sigmoid", UseRoutingBias: true}

	decisions, err := RouteTokens(cfg, [][]float32{{0, 2, 1, -1}}, []float32{0, 0, 0, 4})
	if err != nil {
		t.Fatalf("RouteTokens() error = %v", err)
	}

	if len(decisions) != 1 || len(decisions[0].ExpertIDs) != 2 {
		t.Fatalf("decisions = %+v, want one top-2 decision", decisions)
	}
	if decisions[0].ExpertIDs[0] != 3 || decisions[0].ExpertIDs[1] != 1 {
		t.Fatalf("expert order = %+v, want bias-boosted expert 3 then expert 1", decisions[0].ExpertIDs)
	}
	if !roughlyEqual32(decisions[0].Weights[0]+decisions[0].Weights[1], 1, 0.0001) {
		t.Fatalf("weights = %+v, want renormalized top-k weights", decisions[0].Weights)
	}
}

func TestMiniMaxM2_DispatchExpertsAndProbes_Good(t *testing.T) {
	hidden := [][]float32{{1, 2}}
	decisions := []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{1, 0},
		Weights:    []float32{0.25, 0.75},
	}}
	experts := map[int]ExpertFunc{
		0: func(values []float32) []float32 { return []float32{values[0] * 10, values[1] * 10} },
		1: func(values []float32) []float32 { return []float32{values[0] * 2, values[1] * 2} },
	}

	out, err := DispatchExperts(hidden, decisions, experts)
	if err != nil {
		t.Fatalf("DispatchExperts() error = %v", err)
	}
	if len(out) != 1 || !roughlyEqual32(out[0][0], 8, 0.0001) || !roughlyEqual32(out[0][1], 16, 0.0001) {
		t.Fatalf("out = %+v, want weighted expert sum [8 16]", out)
	}

	events := RouterProbeEvents(3, []int32{42}, decisions)
	if len(events) != 1 || events[0].Kind != probe.KindRouterDecision || events[0].RouterDecision.Layer != 3 {
		t.Fatalf("events = %+v, want router decision probe", events)
	}
	if events[0].RouterDecision.TokenID != 42 || events[0].Meta["architecture"] != "minimax_m2" {
		t.Fatalf("event = %+v, want token id and architecture metadata", events[0])
	}
}

func TestMiniMaxM2_LoadSelectedPackedExpertsFromSafetensors_Good(t *testing.T) {
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}

	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2PackedSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.up_proj.weight", []uint8{1, 1, 2, 0}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.down_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.gate_proj.weight", []uint8{2, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.up_proj.weight", []uint8{0, 1, 1, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.down_proj.weight", []uint8{1, 1, 2, 0}),
	})

	experts, err := LoadPackedExpertsForDecisions(plan, []string{weights}, 0, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: []int{2, 1}, Weights: []float32{0.6, 0.4}},
		{TokenIndex: 1, ExpertIDs: []int{1}, Weights: []float32{1}},
	})
	if err != nil {
		t.Fatalf("LoadPackedExpertsForDecisions() error = %v", err)
	}

	if len(experts) != 2 || experts[1].GateProj.Descriptor.Name == "" || experts[2].DownProj.Descriptor.Name == "" {
		t.Fatalf("experts = %+v, want selected expert 1 and 2 payloads", experts)
	}
	if _, ok := experts[0]; ok {
		t.Fatalf("unexpected unselected expert 0 payload: %+v", experts[0])
	}
	if len(experts[1].GateProj.Packed) != 1 || experts[1].GateProj.Descriptor.PackedBytes != 1 {
		t.Fatalf("expert 1 gate packed = %+v desc=%+v, want one packed byte", experts[1].GateProj.Packed, experts[1].GateProj.Descriptor)
	}
	if len(experts[2].UpProj.Scales) != 1 || experts[2].UpProj.Scales[0] != 1 || experts[2].UpProj.Biases[0] != 0 {
		t.Fatalf("expert 2 up sidecars = scales:%+v biases:%+v", experts[2].UpProj.Scales, experts[2].UpProj.Biases)
	}
}

func TestMiniMaxM2_LoadLazyExpertsForHiddenLoadsOnlyRoutedExperts_Good(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))

	load, err := LoadLazyExpertsForHidden(plan, []string{weights}, 0, [][]float32{{1, 0}}, []int32{42}, nil)
	if err != nil {
		t.Fatalf("LoadLazyExpertsForHidden() error = %v", err)
	}

	if len(load.Decisions) != 1 || len(load.SelectedExpertIDs) != 1 || load.SelectedExpertIDs[0] != 2 {
		t.Fatalf("routing = decisions:%+v selected:%+v, want only expert 2", load.Decisions, load.SelectedExpertIDs)
	}
	if len(load.Experts) != 1 || load.Experts[2].GateProj.Descriptor.Name == "" {
		t.Fatalf("experts = %+v, want only routed expert 2 loaded", load.Experts)
	}
	if len(load.ProbeEvents) != 1 || load.ProbeEvents[0].RouterDecision.TokenID != 42 {
		t.Fatalf("ProbeEvents = %+v, want routed token probe", load.ProbeEvents)
	}
	if load.LoadedPackedBytes != 3 {
		t.Fatalf("LoadedPackedBytes = %d, want three one-byte packed projections", load.LoadedPackedBytes)
	}
}

func TestMiniMaxM2_DequantizedLazyExpertsReturnDenseWeights_Good(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	load, err := LoadLazyExpertsForHidden(plan, []string{weights}, 0, [][]float32{{1, 0}}, nil, nil)
	if err != nil {
		t.Fatalf("LoadLazyExpertsForHidden() error = %v", err)
	}

	dense, err := load.DequantizedExperts()
	if err != nil {
		t.Fatalf("DequantizedExperts() error = %v", err)
	}

	expert := dense[2]
	if !miniMaxM2Float32SlicesRoughlyEqual(expert.GateProj.Weight, []float32{1, 1.5, 2, 2.5}, 0.0001) {
		t.Fatalf("gate dense weight = %+v, want affine-dequantized projection", expert.GateProj.Weight)
	}
	if !sameUint64Slice(expert.GateProj.Descriptor.Shape, []uint64{2, 2}) {
		t.Fatalf("gate dense shape = %+v, want descriptor shape [2 2]", expert.GateProj.Descriptor.Shape)
	}
}

func TestMiniMaxM2_LoadPackedExpertsFromSafetensorsMissingSidecar_Bad(t *testing.T) {
	cfg := Config{ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2, NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 2, NumLocalExperts: 1, NumExpertsPerToken: 1}
	plan, err := BuildTensorPlan(cfg, &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 4, BitsDefault: 2, RoutedExpertBits: 2})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	gate := miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", []uint8{1, 0, 0, 1})
	up := miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.up_proj.weight", []uint8{1, 1, 2, 0})
	down := miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.down_proj.weight", []uint8{1, 0, 0, 1})
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		gate,
		miniMaxM2F32RawTensor(gate.Name+".biases", []float32{0}),
		up,
		miniMaxM2F32RawTensor(up.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(up.Name+".biases", []float32{0}),
		down,
		miniMaxM2F32RawTensor(down.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(down.Name+".biases", []float32{0}),
	})

	_, err = LoadPackedExperts(plan, []string{weights}, 0, []int{0})
	if err == nil || !core.Contains(err.Error(), "scales") {
		t.Fatalf("error = %v, want missing scales diagnostic", err)
	}
}

func TestMiniMaxM2_LoadRouterFromSafetensorsAndProjectScores_Good(t *testing.T) {
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
		UseRoutingBias:     true,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{Profile: "JANGTQ", WeightFormat: "mxtq", Method: "affine+mxtq", GroupSize: 4, BitsDefault: 2, RoutedExpertBits: 2})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			-1, 0,
			0, 1,
			1, 1,
		}, 3, 2),
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.5, -0.25}, 3),
	})

	router, err := LoadRouter(plan, []string{weights}, 0)
	if err != nil {
		t.Fatalf("LoadRouter() error = %v", err)
	}
	scores, err := ProjectRouterScores([][]float32{{1, 2}, {2, 1}}, router)
	if err != nil {
		t.Fatalf("ProjectRouterScores() error = %v", err)
	}

	if router.NumExperts != 3 || router.HiddenSize != 2 || len(router.Bias) != 3 {
		t.Fatalf("router = %+v, want 3 experts, hidden 2, bias", router)
	}
	want := [][]float32{{-1, 2, 3}, {-2, 1, 3}}
	for i := range want {
		if !miniMaxM2Float32SlicesRoughlyEqual(scores[i], want[i], 1e-5) {
			t.Fatalf("scores[%d] = %+v, want %+v", i, scores[i], want[i])
		}
	}
}

func findMiniMaxM2Spec(specs []TensorSpec, role TensorRole) TensorSpec {
	for _, spec := range specs {
		if spec.Role == role {
			return spec
		}
	}
	return TensorSpec{}
}

func findMiniMaxM2ResolvedTensor(tensors []ResolvedTensor, role TensorRole) ResolvedTensor {
	for _, tensor := range tensors {
		if tensor.Role == role {
			return tensor
		}
	}
	return ResolvedTensor{}
}

func roughlyEqual32(a, b, epsilon float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}

func miniMaxM2Float32SlicesRoughlyEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !roughlyEqual32(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

func miniMaxM2SkeletonRawTensors(t *testing.T, plan TensorPlan, badAttentionShape bool) []miniMaxM2RawSafetensor {
	t.Helper()
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	var tensors []miniMaxM2RawSafetensor
	for _, role := range []TensorRole{
		TensorRoleAttentionQ,
		TensorRoleAttentionK,
		TensorRoleAttentionV,
		TensorRoleAttentionO,
	} {
		spec := findMiniMaxM2Spec(specs, role)
		if spec.Packed == nil {
			t.Fatalf("attention spec %s has no packed descriptor", role)
		}
		packedBytes := spec.Packed.PackedBytes
		if badAttentionShape && role == TensorRoleAttentionQ {
			packedBytes--
		}
		tensors = append(tensors, miniMaxM2RawSafetensor{
			Name:  spec.Name,
			DType: "U8",
			Shape: []int{packedBytes},
			Raw:   make([]byte, packedBytes),
		})
	}
	tensors = append(tensors,
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
			1, 1, 0, 0,
		}, 3, 4),
	)
	if plan.Config.UseRoutingBias {
		tensors = append(tensors, miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.25, -0.25}, 3))
	}
	return tensors
}

func miniMaxM2SmallJANGTQPlan(t *testing.T) TensorPlan {
	t.Helper()
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 1,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	return plan
}

func miniMaxM2LazyExpertFixtureTensors(t *testing.T, expertID int, values []uint8) []miniMaxM2RawSafetensor {
	t.Helper()
	prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", expertID)
	gate := miniMaxM2PackedRawTensor(t, prefix+".gate_proj.weight", values)
	up := miniMaxM2PackedRawTensor(t, prefix+".up_proj.weight", values)
	down := miniMaxM2PackedRawTensor(t, prefix+".down_proj.weight", values)
	return []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			0, 0,
			-1, 0,
			3, 0,
		}, 3, 2),
		gate,
		miniMaxM2F32RawTensor(gate.Name+".scales", []float32{0.5}),
		miniMaxM2F32RawTensor(gate.Name+".biases", []float32{1}),
		up,
		miniMaxM2F32RawTensor(up.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(up.Name+".biases", []float32{0}),
		down,
		miniMaxM2F32RawTensor(down.Name+".scales", []float32{1}),
		miniMaxM2F32RawTensor(down.Name+".biases", []float32{0}),
	}
}

type miniMaxM2RawSafetensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

func miniMaxM2PackedRawTensor(t *testing.T, name string, values []uint8) miniMaxM2RawSafetensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:        name,
		Shape:       []uint64{2, 2},
		Elements:    4,
		Bits:        2,
		GroupSize:   4,
		PackedBytes: 1,
		ScaleCount:  1,
		BiasCount:   1,
	}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues() error = %v", err)
	}
	return miniMaxM2RawSafetensor{Name: name, DType: "U8", Shape: []int{len(packed)}, Raw: packed}
}

func writeMiniMaxM2PackedSafetensors(t *testing.T, path string, tensors []miniMaxM2RawSafetensor) {
	t.Helper()
	withSidecars := make([]miniMaxM2RawSafetensor, 0, len(tensors)*3)
	for _, tensor := range tensors {
		withSidecars = append(withSidecars, tensor)
		withSidecars = append(withSidecars,
			miniMaxM2F32RawTensor(tensor.Name+".scales", []float32{1}),
			miniMaxM2F32RawTensor(tensor.Name+".biases", []float32{0}),
		)
	}
	writeMiniMaxM2RawSafetensors(t, path, withSidecars)
}

func miniMaxM2F32RawTensor(name string, values []float32, shape ...int) miniMaxM2RawSafetensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	if len(shape) == 0 {
		shape = []int{len(values)}
	}
	return miniMaxM2RawSafetensor{Name: name, DType: "F32", Shape: append([]int(nil), shape...), Raw: raw}
}

func writeMiniMaxM2RawSafetensors(t *testing.T, path string, tensors []miniMaxM2RawSafetensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.Raw...)
		header[tensor.Name] = entry{
			DType:       tensor.DType,
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsMetalUsesFusedProjection_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	hidden := [][]float32{{1, 2}}
	decisions := []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{0.75, 0.25},
	}}
	experts := map[int]PackedExpertWeights{
		0: miniMaxM2PackedExpertFixture(t,
			[]uint8{1, 0, 0, 1},
			[]uint8{1, 1, 2, 0},
			[]uint8{1, 0, 0, 1},
		),
		1: miniMaxM2PackedExpertFixture(t,
			[]uint8{2, 0, 0, 1},
			[]uint8{0, 1, 1, 1},
			[]uint8{1, 1, 2, 0},
		),
	}

	got, err := DispatchPackedExpertsMetal(hidden, decisions, experts)
	if err != nil {
		t.Fatalf("DispatchPackedExpertsMetal() error = %v", err)
	}

	want := miniMaxM2PackedDispatchReference(t, hidden, decisions, experts)
	if len(got) != 1 || !float32SlicesRoughlyEqual(got[0], want[0], 1e-4) {
		t.Fatalf("got = %+v, want %+v", got, want)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsMetalRejectsMissingExpert_Bad(t *testing.T) {
	_, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{7},
		Weights:    []float32{1},
	}}, nil)
	if err == nil || !core.Contains(err.Error(), "missing expert 7") {
		t.Fatalf("error = %v, want missing expert diagnostic", err)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsMetalRejectsMalformedDecisions_Bad(t *testing.T) {
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, []RouterDecision{{
		TokenIndex: 2,
		ExpertIDs:  []int{0},
		Weights:    []float32{1},
	}}, nil); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("out-of-range error = %v", err)
	}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{1},
	}}, nil); err == nil || !core.Contains(err.Error(), "length mismatch") {
		t.Fatalf("length mismatch error = %v", err)
	}
	if _, err := ForwardLazyExpertLoadMetal([][]float32{{1, 2}}, LazyExpertLoad{
		Decisions: []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{3}, Weights: []float32{1}}},
	}); err == nil || !core.Contains(err.Error(), "missing expert") {
		t.Fatalf("lazy load error = %v, want missing expert", err)
	}
	if _, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Hidden:       [][]float32{{1, 2}},
		RouterScores: [][]float32{{1}, {2}},
	}); err == nil || !core.Contains(err.Error(), "hidden rows") {
		t.Fatalf("packed layer shape error = %v", err)
	}
	if got := swiGLU(0.5, 2); math.IsNaN(float64(got)) || got == 0 {
		t.Fatalf("swiGLU() = %v, want finite non-zero", got)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsFromSafetensorsMetal_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    2,
		NumExpertsPerToken: 2,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2PackedSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.up_proj.weight", []uint8{1, 1, 2, 0}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.0.down_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", []uint8{2, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.up_proj.weight", []uint8{0, 1, 1, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.down_proj.weight", []uint8{1, 1, 2, 0}),
	})
	hidden := [][]float32{{1, 2}}
	decisions := []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{0.75, 0.25},
	}}

	got, err := DispatchPackedExpertsFromSafetensorsMetal(plan, []string{weights}, 0, hidden, decisions)
	if err != nil {
		t.Fatalf("DispatchPackedExpertsFromSafetensorsMetal() error = %v", err)
	}
	experts, err := LoadPackedExpertsForDecisions(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadPackedExpertsForDecisions() error = %v", err)
	}
	want := miniMaxM2PackedDispatchReference(t, hidden, decisions, experts)
	if len(got) != 1 || !float32SlicesRoughlyEqual(got[0], want[0], 1e-4) {
		t.Fatalf("got = %+v, want %+v", got, want)
	}
}

func TestMiniMaxM2_ForwardLazyExpertLoadMetal_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	hidden := [][]float32{{1, 0}}
	load, err := LoadLazyExpertsForHidden(plan, []string{weights}, 0, hidden, []int32{42}, nil)
	if err != nil {
		t.Fatalf("LoadLazyExpertsForHidden() error = %v", err)
	}

	got, err := ForwardLazyExpertLoadMetal(hidden, load)
	if err != nil {
		t.Fatalf("ForwardLazyExpertLoadMetal() error = %v", err)
	}

	want := miniMaxM2PackedDispatchReference(t, hidden, load.Decisions, load.Experts)
	if len(got.Output) != 1 || !float32SlicesRoughlyEqual(got.Output[0], want[0], 1e-4) {
		t.Fatalf("output = %+v, want %+v", got.Output, want)
	}
	if got.LoadedPackedBytes != 3 || len(got.SelectedExpertIDs) != 1 || got.SelectedExpertIDs[0] != 2 {
		t.Fatalf("result metadata = bytes:%d experts:%+v, want 3/[2]", got.LoadedPackedBytes, got.SelectedExpertIDs)
	}
	if len(got.ProbeEvents) != 1 || got.ProbeEvents[0].RouterDecision.TokenID != 42 {
		t.Fatalf("probe events = %+v, want load probe events forwarded", got.ProbeEvents)
	}
}

func TestMiniMaxM2_ForwardPackedLayerMetalRoutesLoadsAndProbes_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
		ScoringFunc:        "sigmoid",
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2PackedSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.up_proj.weight", []uint8{1, 1, 2, 0}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.down_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.gate_proj.weight", []uint8{2, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.up_proj.weight", []uint8{0, 1, 1, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.down_proj.weight", []uint8{1, 1, 2, 0}),
	})
	hidden := [][]float32{{1, 2}, {2, 1}}
	routerScores := [][]float32{
		{-5, 3, 1},
		{-4, 2, 0},
	}
	recorder := probe.NewRecorder()

	got, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Plan:         plan,
		WeightFiles:  []string{weights},
		Layer:        0,
		Hidden:       hidden,
		RouterScores: routerScores,
		TokenIDs:     []int32{101, 102},
		ProbeSink:    recorder,
	})
	if err != nil {
		t.Fatalf("ForwardPackedLayerMetal() error = %v", err)
	}

	decisions, err := RouteTokens(cfg, routerScores, nil)
	if err != nil {
		t.Fatalf("RouteTokens() error = %v", err)
	}
	experts, err := LoadPackedExpertsForDecisions(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadPackedExpertsForDecisions() error = %v", err)
	}
	want := miniMaxM2PackedDispatchReference(t, hidden, decisions, experts)
	if len(got.Output) != len(want) || !float32SlicesRoughlyEqual(got.Output[0], want[0], 1e-4) || !float32SlicesRoughlyEqual(got.Output[1], want[1], 1e-4) {
		t.Fatalf("output = %+v, want %+v", got.Output, want)
	}
	if len(got.SelectedExpertIDs) != 2 || got.SelectedExpertIDs[0] != 1 || got.SelectedExpertIDs[1] != 2 {
		t.Fatalf("selected experts = %+v, want [1 2]", got.SelectedExpertIDs)
	}
	if got.LoadedPackedBytes != 6 {
		t.Fatalf("LoadedPackedBytes = %d, want two selected one-byte experts", got.LoadedPackedBytes)
	}
	events := recorder.Events()
	if len(events) != 2 || len(got.ProbeEvents) != 2 {
		t.Fatalf("events recorder/result = %d/%d, want 2", len(events), len(got.ProbeEvents))
	}
	if events[0].Kind != probe.KindRouterDecision || events[0].RouterDecision.TokenID != 101 || events[0].RouterDecision.Layer != 0 {
		t.Fatalf("first event = %+v, want router decision for token 101 layer 0", events[0])
	}
	if events[0].RouterDecision.ExpertIDs[0] != 1 || events[0].Meta["architecture"] != "minimax_m2" {
		t.Fatalf("first event router = %+v meta=%+v", events[0].RouterDecision, events[0].Meta)
	}
}

func TestMiniMaxM2_ForwardPackedLayerFromSafetensorsMetalProjectsRouter_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         2,
		IntermediateSize:   2,
		NumHiddenLayers:    1,
		NumAttentionHeads:  1,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    3,
		NumExpertsPerToken: 2,
		ScoringFunc:        "sigmoid",
		UseRoutingBias:     true,
	}
	plan, err := BuildTensorPlan(cfg, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	tensors := []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			-3, 0,
			0, 2,
			2, 0,
		}, 3, 2),
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.25, 0.5}, 3),
	}
	for _, tensor := range []miniMaxM2RawSafetensor{
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.up_proj.weight", []uint8{1, 1, 2, 0}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.1.down_proj.weight", []uint8{1, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.gate_proj.weight", []uint8{2, 0, 0, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.up_proj.weight", []uint8{0, 1, 1, 1}),
		miniMaxM2PackedRawTensor(t, "model.layers.0.block_sparse_moe.experts.2.down_proj.weight", []uint8{1, 1, 2, 0}),
	} {
		tensors = append(tensors,
			tensor,
			miniMaxM2F32RawTensor(tensor.Name+".scales", []float32{1}),
			miniMaxM2F32RawTensor(tensor.Name+".biases", []float32{0}),
		)
	}
	writeMiniMaxM2RawSafetensors(t, weights, tensors)
	hidden := [][]float32{{1, 2}, {2, 1}}
	recorder := probe.NewRecorder()

	got, err := ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:        plan,
		WeightFiles: []string{weights},
		Layer:       0,
		Hidden:      hidden,
		TokenIDs:    []int32{201, 202},
		ProbeSink:   recorder,
	})
	if err != nil {
		t.Fatalf("ForwardPackedLayerFromSafetensorsMetal() error = %v", err)
	}

	router, err := LoadRouter(plan, []string{weights}, 0)
	if err != nil {
		t.Fatalf("LoadRouter() error = %v", err)
	}
	scores, err := ProjectRouterScores(hidden, router)
	if err != nil {
		t.Fatalf("ProjectRouterScores() error = %v", err)
	}
	decisions, err := RouteTokens(cfg, scores, router.Bias)
	if err != nil {
		t.Fatalf("RouteTokens() error = %v", err)
	}
	experts, err := LoadPackedExpertsForDecisions(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadPackedExpertsForDecisions() error = %v", err)
	}
	want := miniMaxM2PackedDispatchReference(t, hidden, decisions, experts)
	if len(got.Output) != 2 || !float32SlicesRoughlyEqual(got.Output[0], want[0], 1e-4) || !float32SlicesRoughlyEqual(got.Output[1], want[1], 1e-4) {
		t.Fatalf("output = %+v, want %+v", got.Output, want)
	}
	if len(got.SelectedExpertIDs) != 2 || got.SelectedExpertIDs[0] != 1 || got.SelectedExpertIDs[1] != 2 {
		t.Fatalf("selected experts = %+v, want [1 2]", got.SelectedExpertIDs)
	}
	if got.LoadedPackedBytes != 6 {
		t.Fatalf("LoadedPackedBytes = %d, want two selected one-byte experts", got.LoadedPackedBytes)
	}
	events := recorder.Events()
	if len(events) != 2 || events[0].RouterDecision.TokenID != 201 {
		t.Fatalf("events = %+v, want router probes from computed scores", events)
	}
}

func miniMaxM2PackedExpertFixture(t *testing.T, gateValues, upValues, downValues []uint8) PackedExpertWeights {
	t.Helper()
	return PackedExpertWeights{
		GateProj: miniMaxM2PackedProjectionFixture(t, "gate_proj", gateValues),
		UpProj:   miniMaxM2PackedProjectionFixture(t, "up_proj", upValues),
		DownProj: miniMaxM2PackedProjectionFixture(t, "down_proj", downValues),
	}
}

func miniMaxM2PackedProjectionFixture(t *testing.T, projection string, values []uint8) JANGPackedProjectionTensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0." + projection + ".weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{2, 2},
		Elements:      4,
		Bits:          2,
		GroupSize:     4,
		Groups:        1,
		PackedBytes:   1,
		ValuesPerByte: 4,
		ScaleCount:    1,
		BiasCount:     1,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues(%s) error = %v", projection, err)
	}
	return JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1},
		Biases:     []float32{0},
	}
}

func miniMaxM2PackedDispatchReference(t *testing.T, hidden [][]float32, decisions []RouterDecision, experts map[int]PackedExpertWeights) [][]float32 {
	t.Helper()
	out := make([][]float32, len(hidden))
	for _, decision := range decisions {
		for i, expertID := range decision.ExpertIDs {
			expertOut := miniMaxM2PackedExpertReference(t, hidden[decision.TokenIndex], experts[expertID])
			if out[decision.TokenIndex] == nil {
				out[decision.TokenIndex] = make([]float32, len(expertOut))
			}
			for j, value := range expertOut {
				out[decision.TokenIndex][j] += decision.Weights[i] * value
			}
		}
	}
	return out
}

func miniMaxM2PackedExpertReference(t *testing.T, hidden []float32, expert PackedExpertWeights) []float32 {
	t.Helper()
	gate := miniMaxM2PackedProjectionReference(t, hidden, expert.GateProj)
	up := miniMaxM2PackedProjectionReference(t, hidden, expert.UpProj)
	if len(gate) != len(up) {
		t.Fatalf("gate len = %d, up len = %d", len(gate), len(up))
	}
	activated := make([]float32, len(gate))
	for i := range gate {
		activated[i] = float32(float64(gate[i])/(1+math.Exp(float64(-gate[i])))) * up[i]
	}
	return miniMaxM2PackedProjectionReference(t, activated, expert.DownProj)
}

func miniMaxM2PackedProjectionReference(t *testing.T, input []float32, projection JANGPackedProjectionTensor) []float32 {
	t.Helper()
	weight, err := jang.DequantizePackedTensor(projection.Descriptor, projection.Packed, projection.Scales, projection.Biases)
	if err != nil {
		t.Fatalf("jang.DequantizePackedTensor() error = %v", err)
	}
	outDim := int(projection.Descriptor.Shape[0])
	inDim := int(projection.Descriptor.Shape[1])
	return denseProjectionReference(input, 1, weight, outDim, inDim, projection.Bias)
}
