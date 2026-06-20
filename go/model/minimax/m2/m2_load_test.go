// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
)

func TestM2Load_BuildLayerForwardSkeleton_Good(t *testing.T) {
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

func TestM2Load_BuildLayerForwardSkeleton_Ugly(t *testing.T) {
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

func TestM2Load_LoadPackedExpertsForDecisions_Good(t *testing.T) {
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

func TestM2Load_LoadLazyExpertsForHidden_Good(t *testing.T) {
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

func TestM2Load_LazyExpertLoad_DequantizedExperts_Good(t *testing.T) {
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

func TestM2Load_LoadPackedExperts_Ugly(t *testing.T) {
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

func TestM2Load_LoadRouter_Good(t *testing.T) {
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

func TestM2Load_LoadRouter_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	// No weight files -> cheap pre-IO rejection.
	if _, err := LoadRouter(plan, nil, 0); err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
	// Out-of-range layer surfaces the LayerTensorSpecs range error.
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 0, []uint8{0, 1, 2, 3}))
	if _, err := LoadRouter(plan, []string{weights}, 99); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want layer out of range", err)
	}
}

func TestM2Load_LoadPackedExperts_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	if _, err := LoadPackedExperts(plan, nil, 0, []int{0}); err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
}

func TestM2Load_BuildLayerForwardSkeleton_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	if _, err := BuildLayerForwardSkeleton(plan, nil, 0); err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
}

// --- *Metal error legs (device-free: every assertion returns before kernels) ---

func TestM2Load_findPackedWeightRef_Good(t *testing.T) {
	// Canonical layout: the spec name is itself a tensor in the index.
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight"}
	index := miniMaxM2Index("experts.0.gate_proj.weight")
	ref, name, ok := findPackedWeightRef(index, spec)
	if !ok || name != "experts.0.gate_proj.weight" || ref.Name != name {
		t.Fatalf("findPackedWeightRef() = (%+v,%q,%v), want canonical hit", ref, name, ok)
	}
}

func TestM2Load_findPackedWeightRef_Ugly(t *testing.T) {
	// Spec name absent; resolution must fall through to the ".qweight" of a
	// trimmed alias — exercising the alias loop + trim(base)+".qweight" leg.
	spec := &TensorSpec{Name: "missing", Aliases: []string{"experts.0.up_proj.weight"}}
	index := miniMaxM2Index("experts.0.up_proj.qweight")
	ref, name, ok := findPackedWeightRef(index, spec)
	if !ok || name != "experts.0.up_proj.qweight" || ref.Name != name {
		t.Fatalf("findPackedWeightRef() = (%+v,%q,%v), want trimmed alias .qweight hit", ref, name, ok)
	}
}

func TestM2Load_findPackedWeightRef_Bad(t *testing.T) {
	// No candidate shape matches: full fan-out walked, miss returned.
	spec := &TensorSpec{Name: "experts.0.down_proj.weight", Aliases: []string{"legacy.down"}}
	index := miniMaxM2Index("some.other.tensor")
	if ref, name, ok := findPackedWeightRef(index, spec); ok {
		t.Fatalf("findPackedWeightRef() = (%+v,%q,%v), want full miss", ref, name, ok)
	}
}

func TestMiniMaxM2_TryPackedWeightName_PackedAndQweightSuffixes(t *testing.T) {
	// Base itself absent, but "base.packed" present → first suffix probe hit.
	if _, name, ok := tryPackedWeightName(miniMaxM2Index("w.packed"), nil, "w"); !ok || name != "w.packed" {
		t.Fatalf("tryPackedWeightName(.packed) = (%q,%v), want w.packed hit", name, ok)
	}
	// "base.qweight" present (no trim needed since base has no .weight).
	if _, name, ok := tryPackedWeightName(miniMaxM2Index("w.qweight"), nil, "w"); !ok || name != "w.qweight" {
		t.Fatalf("tryPackedWeightName(.qweight) = (%q,%v), want w.qweight hit", name, ok)
	}
}

func TestM2Load_findSidecarRef_Good(t *testing.T) {
	// Canonical: weightName + "." + sidecar.
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight"}
	index := miniMaxM2Index("experts.0.gate_proj.weight.scales")
	ref, name, ok := findSidecarRef(index, spec, "experts.0.gate_proj.weight", "scales")
	if !ok || name != "experts.0.gate_proj.weight.scales" || ref.Name != name {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want dotted scales hit", ref, name, ok)
	}
}

func TestM2Load_findSidecarRef_Ugly(t *testing.T) {
	// Underscore sidecar form on a packed weight name — drives the
	// trim(weightName) fall-through plus the name+"_"+sidecar leg.
	spec := &TensorSpec{Name: "missing"}
	index := miniMaxM2Index("experts.0.up_proj_biases")
	ref, name, ok := findSidecarRef(index, spec, "experts.0.up_proj.packed", "biases")
	if !ok || name != "experts.0.up_proj_biases" || ref.Name != name {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want underscore biases via trimmed packed name", ref, name, ok)
	}
}

func TestM2Load_findSidecarRef_Bad(t *testing.T) {
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Aliases: []string{"legacy.gate"}}
	index := miniMaxM2Index("unrelated.tensor")
	if ref, name, ok := findSidecarRef(index, spec, "experts.0.gate_proj.weight", "scales"); ok {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want full sidecar miss", ref, name, ok)
	}
}

func TestM2Load_findProjectionBiasRef_Good(t *testing.T) {
	// trim(weightName)+".bias" is the first projection-bias probe.
	spec := &TensorSpec{Name: "experts.0.down_proj.weight"}
	index := miniMaxM2Index("experts.0.down_proj.bias")
	ref, name, ok := findProjectionBiasRef(index, spec, "experts.0.down_proj.weight")
	if !ok || name != "experts.0.down_proj.bias" || ref.Name != name {
		t.Fatalf("findProjectionBiasRef() = (%+v,%q,%v), want trimmed .bias hit", ref, name, ok)
	}
}

func TestM2Load_findProjectionBiasRef_Ugly(t *testing.T) {
	// weightName has no bias; spec.Name (distinct) carries a ".proj_bias".
	spec := &TensorSpec{Name: "experts.0.down_proj"}
	index := miniMaxM2Index("experts.0.down_proj.proj_bias")
	ref, name, ok := findProjectionBiasRef(index, spec, "experts.0.down_proj.weight")
	if !ok || name != "experts.0.down_proj.proj_bias" || ref.Name != name {
		t.Fatalf("findProjectionBiasRef() = (%+v,%q,%v), want spec.Name .proj_bias hit", ref, name, ok)
	}
}

func TestM2Load_findProjectionBiasRef_Bad(t *testing.T) {
	// Projection bias is typically absent for MiniMax M2 — the common case.
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight", Aliases: []string{"legacy.gate"}}
	index := miniMaxM2Index("experts.0.gate_proj.weight")
	if ref, name, ok := findProjectionBiasRef(index, spec, "experts.0.gate_proj.weight"); ok {
		t.Fatalf("findProjectionBiasRef() = (%+v,%q,%v), want absent-bias miss", ref, name, ok)
	}
}

// --- DType / suffix leaves ---------------------------------------------

func TestMiniMaxM2_TrimWeightAndPackedSuffix(t *testing.T) {
	if got := trimWeightSuffix("experts.0.gate_proj.weight"); got != "experts.0.gate_proj" {
		t.Fatalf("trimWeightSuffix() = %q, want .weight stripped", got)
	}
	if got := trimWeightSuffix("experts.0.gate_proj"); got != "experts.0.gate_proj" {
		t.Fatalf("trimWeightSuffix(no suffix) = %q, want unchanged", got)
	}
	if got := trimPackedSuffix("w.packed"); got != "w" {
		t.Fatalf("trimPackedSuffix(.packed) = %q, want w", got)
	}
	if got := trimPackedSuffix("w.qweight"); got != "w" {
		t.Fatalf("trimPackedSuffix(.qweight) = %q, want w", got)
	}
	if got := trimPackedSuffix("w.bias"); got != "w.bias" {
		t.Fatalf("trimPackedSuffix(no packed suffix) = %q, want unchanged", got)
	}
}

func TestMiniMaxM2_FindProjectionBiasRef_AliasLeg(t *testing.T) {
	// weightName and spec.Name both miss; an alias carries the ".bias" — the
	// only path that walks the alias loop in findProjectionBiasRef.
	spec := &TensorSpec{Name: "experts.0.down_proj.weight", Aliases: []string{"mlp.experts.0.down_proj.weight"}}
	index := miniMaxM2Index("mlp.experts.0.down_proj.bias")
	ref, name, ok := findProjectionBiasRef(index, spec, "experts.0.down_proj.weight")
	if !ok || name != "mlp.experts.0.down_proj.bias" || ref.Name != name {
		t.Fatalf("findProjectionBiasRef() = (%+v,%q,%v), want alias .bias hit", ref, name, ok)
	}
}

func TestMiniMaxM2_FindSidecarRef_SpecNameLeg(t *testing.T) {
	// weightName (and its trim) miss; spec.Name resolves the sidecar — the
	// spec.Name fall-through distinct from the weightName/trim/alias legs.
	spec := &TensorSpec{Name: "experts.0.gate_proj.weight"}
	index := miniMaxM2Index("experts.0.gate_proj.scales")
	ref, name, ok := findSidecarRef(index, spec, "experts.0.gate_proj.packed", "scales")
	if !ok || name != "experts.0.gate_proj.scales" || ref.Name != name {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want spec.Name trimmed scales hit", ref, name, ok)
	}
}

func TestMiniMaxM2_FindSidecarRef_AliasLeg(t *testing.T) {
	// weightName, its trim, and spec.Name all miss; only an alias resolves the
	// sidecar — the alias loop, the last reachable findSidecarRef branch.
	spec := &TensorSpec{Name: "missing.gate.weight", Aliases: []string{"mlp.experts.0.gate_proj.weight"}}
	index := miniMaxM2Index("mlp.experts.0.gate_proj.scales")
	ref, name, ok := findSidecarRef(index, spec, "weight.packed", "scales")
	if !ok || name != "mlp.experts.0.gate_proj.scales" || ref.Name != name {
		t.Fatalf("findSidecarRef() = (%+v,%q,%v), want alias trimmed scales hit", ref, name, ok)
	}
}

func TestMiniMaxM2_TryProjectionBiasName_TrimmedProjBias(t *testing.T) {
	// Neither trim(name)+".bias" nor name+".proj_bias" hit, but
	// trim(name)+".proj_bias" does — the third probe, which only fires when
	// the name actually carries a ".weight" suffix to trim.
	ref, name, ok := tryProjectionBiasName(miniMaxM2Index("experts.0.down_proj.proj_bias"), nil, "experts.0.down_proj.weight")
	if !ok || name != "experts.0.down_proj.proj_bias" || ref.Name != name {
		t.Fatalf("tryProjectionBiasName() = (%+v,%q,%v), want trimmed .proj_bias hit", ref, name, ok)
	}
}

// --- Authored triplet completions (device-free; safetensors fixtures only) ---

func TestM2Load_LoadPackedExpertsForDecisions_Bad(t *testing.T) {
	// No weight files: LoadPackedExpertsForDecisions delegates to
	// LoadPackedExperts, which rejects before any I/O.
	plan := miniMaxM2SmallJANGTQPlan(t)
	_, err := LoadPackedExpertsForDecisions(plan, nil, 0, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: []int{1}, Weights: []float32{1}},
	})
	if err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
}

func TestM2Load_LoadPackedExpertsForDecisions_Ugly(t *testing.T) {
	// Decisions referencing no experts (all empty ExpertIDs) reduce to an empty
	// expert-id set, so with a valid (if unused) weight file present
	// LoadPackedExpertsForDecisions returns an empty map without reading any
	// expert payload.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	experts, err := LoadPackedExpertsForDecisions(plan, []string{weights}, 0, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: nil, Weights: nil},
	})
	if err != nil {
		t.Fatalf("LoadPackedExpertsForDecisions(no expert ids) error = %v", err)
	}
	if len(experts) != 0 {
		t.Fatalf("experts = %+v, want empty map when no experts are routed", experts)
	}
}

func TestM2Load_LoadLazyExpertsForHidden_Bad(t *testing.T) {
	// No weight files: the router load fails before routing or expert loading.
	plan := miniMaxM2SmallJANGTQPlan(t)
	_, err := LoadLazyExpertsForHidden(plan, nil, 0, [][]float32{{1, 0}}, nil, nil)
	if err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
}

func TestM2Load_LoadLazyExpertsForHidden_Ugly(t *testing.T) {
	// A hidden row whose width != hidden size fails in ProjectRouterScores after
	// the router loads — the post-load error path distinct from the no-files leg.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	// plan hidden size is 2; a width-3 row is rejected by the router projection.
	_, err := LoadLazyExpertsForHidden(plan, []string{weights}, 0, [][]float32{{1, 0, 0}}, nil, nil)
	if err == nil || !core.Contains(err.Error(), "hidden row") {
		t.Fatalf("error = %v, want hidden row width diagnostic from ProjectRouterScores", err)
	}
}

func TestM2Load_LoadPackedExperts_Good(t *testing.T) {
	// Load exactly the requested expert (id 2) and verify its packed descriptor,
	// packed bytes, and affine sidecars resolve from safetensors.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))

	experts, err := LoadPackedExperts(plan, []string{weights}, 0, []int{2})
	if err != nil {
		t.Fatalf("LoadPackedExperts() error = %v", err)
	}
	if len(experts) != 1 {
		t.Fatalf("experts = %+v, want exactly expert 2 loaded", experts)
	}
	expert, ok := experts[2]
	if !ok || expert.GateProj.Descriptor.Name == "" {
		t.Fatalf("expert 2 = %+v, want resolved gate descriptor", expert)
	}
	if len(expert.GateProj.Packed) != 1 || expert.GateProj.Descriptor.PackedBytes != 1 {
		t.Fatalf("expert 2 gate packed = %+v, want one packed byte", expert.GateProj.Packed)
	}
	if len(expert.GateProj.Scales) != 1 || expert.GateProj.Scales[0] != 0.5 || expert.GateProj.Biases[0] != 1 {
		t.Fatalf("expert 2 gate sidecars = scales:%+v biases:%+v, want fixture 0.5/1", expert.GateProj.Scales, expert.GateProj.Biases)
	}
}

func TestM2Load_LazyExpertLoad_DequantizedExperts_Bad(t *testing.T) {
	// A LazyExpertLoad carrying an expert with a malformed packed projection
	// (descriptor promises more elements than the sidecars/packed bytes support)
	// surfaces the dequantizer error rather than returning dense garbage.
	bad := LazyExpertLoad{
		Experts: map[int]PackedExpertWeights{
			0: {
				GateProj: JANGPackedProjectionTensor{
					Descriptor: jang.PackedTensorDescriptor{
						Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4,
						Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1,
					},
					// No packed bytes / no scales: validation inside the
					// dequantizer fails for this projection.
				},
			},
		},
	}
	if _, err := bad.DequantizedExperts(); err == nil {
		t.Fatal("DequantizedExperts() error = nil, want dequantizer error for malformed projection")
	}
}

func TestM2Load_LazyExpertLoad_DequantizedExperts_Ugly(t *testing.T) {
	// An empty load (no experts) dequantizes to an empty dense map without error.
	empty := LazyExpertLoad{}
	dense, err := empty.DequantizedExperts()
	if err != nil {
		t.Fatalf("DequantizedExperts(empty) error = %v", err)
	}
	if len(dense) != 0 {
		t.Fatalf("dense = %+v, want empty map for a load with no experts", dense)
	}
}

func TestM2Load_DequantizeJANGPackedProjection_Good(t *testing.T) {
	// A 2x2 affine-packed projection with scale 1 / bias 0 dequantizes its raw
	// quantised values straight back to float weights.
	projection := miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3})
	dense, err := DequantizeJANGPackedProjection(projection)
	if err != nil {
		t.Fatalf("DequantizeJANGPackedProjection() error = %v", err)
	}
	if !sameUint64Slice(dense.Descriptor.Shape, []uint64{2, 2}) {
		t.Fatalf("dense shape = %+v, want descriptor shape [2 2]", dense.Descriptor.Shape)
	}
	// scale 1, bias 0 → dequantized weight equals the quantised value.
	if !miniMaxM2Float32SlicesRoughlyEqual(dense.Weight, []float32{0, 1, 2, 3}, 1e-4) {
		t.Fatalf("dense weight = %+v, want identity dequant [0 1 2 3]", dense.Weight)
	}
}

func TestM2Load_DequantizeJANGPackedProjection_Bad(t *testing.T) {
	// A descriptor with no packed bytes and no sidecars cannot be dequantized;
	// the JANG dequantizer surfaces the error.
	if _, err := DequantizeJANGPackedProjection(JANGPackedProjectionTensor{
		Descriptor: jang.PackedTensorDescriptor{
			Name: "experts.0.gate_proj.weight", Shape: []uint64{2, 2}, Elements: 4,
			Bits: 2, GroupSize: 4, PackedBytes: 1, ScaleCount: 1, BiasCount: 1,
		},
	}); err == nil {
		t.Fatal("DequantizeJANGPackedProjection() error = nil, want error for missing packed payload")
	}
}

func TestM2Load_DequantizeJANGPackedProjection_Ugly(t *testing.T) {
	// A projection that also carries a per-output Bias: DequantizeJANGPackedProjection
	// clones the bias into the dense tensor alongside the dequantized weight.
	projection := miniMaxM2PackedProjectionFixture(t, "down_proj", []uint8{1, 1, 1, 1})
	projection.Bias = []float32{0.25, -0.5}
	dense, err := DequantizeJANGPackedProjection(projection)
	if err != nil {
		t.Fatalf("DequantizeJANGPackedProjection() error = %v", err)
	}
	if len(dense.Bias) != 2 || dense.Bias[0] != 0.25 || dense.Bias[1] != -0.5 {
		t.Fatalf("dense bias = %+v, want cloned projection bias [0.25 -0.5]", dense.Bias)
	}
	// The clone must be a copy, not an alias of the input slice.
	projection.Bias[0] = 99
	if dense.Bias[0] != 0.25 {
		t.Fatalf("dense bias mutated to %v, want an independent clone", dense.Bias[0])
	}
}

func TestM2Load_LoadRouter_Ugly(t *testing.T) {
	// The router gate is present but its on-disk shape disagrees with the plan
	// (2 experts on disk vs 3 in the config), so LoadRouter rejects the gate
	// shape rather than loading a mis-shaped router.
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 2,
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	// Only 2 expert rows on disk; the plan expects 3.
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0, 0, 1}, 2, 2),
	})
	if _, err := LoadRouter(plan, []string{weights}, 0); err == nil || !core.Contains(err.Error(), "gate shape") {
		t.Fatalf("error = %v, want router gate shape diagnostic", err)
	}
}
