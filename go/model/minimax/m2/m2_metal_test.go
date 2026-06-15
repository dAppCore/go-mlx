// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
)

func TestM2Metal_DispatchPackedExpertsMetal_Good(t *testing.T) {
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

func TestM2Metal_DispatchPackedExpertsMetal_Bad(t *testing.T) {
	_, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{7},
		Weights:    []float32{1},
	}}, nil)
	if err == nil || !core.Contains(err.Error(), "missing expert 7") {
		t.Fatalf("error = %v, want missing expert diagnostic", err)
	}
}

func TestM2Metal_ForwardLazyExpertLoadMetal_Bad(t *testing.T) {
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

func TestM2Metal_DispatchPackedExpertsFromSafetensorsMetal_Good(t *testing.T) {
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

func TestM2Metal_ForwardLazyExpertLoadMetal_Good(t *testing.T) {
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

func TestM2Metal_ForwardPackedLayerMetal_Good(t *testing.T) {
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

func TestM2Metal_ForwardPackedLayerFromSafetensorsMetal_Good(t *testing.T) {
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

func TestM2Metal_DispatchPackedExpertsMetal_Ugly(t *testing.T) {
	// Token index out of range and expert/weight length mismatch both return
	// before any Metal projection runs, so this needs no device.
	if _, err := DispatchPackedExpertsMetal([][]float32{{1}}, []RouterDecision{{TokenIndex: 4, ExpertIDs: []int{0}, Weights: []float32{1}}}, nil); err == nil || !core.Contains(err.Error(), "token index") {
		t.Fatalf("error = %v, want token index out of range", err)
	}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1}}, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0, 1}, Weights: []float32{1}}}, nil); err == nil || !core.Contains(err.Error(), "length mismatch") {
		t.Fatalf("error = %v, want expert/weight length mismatch", err)
	}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1}}, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{9}, Weights: []float32{1}}}, map[int]PackedExpertWeights{}); err == nil || !core.Contains(err.Error(), "missing expert") {
		t.Fatalf("error = %v, want missing expert", err)
	}
}

func TestM2Metal_DispatchPackedExpertsFromSafetensorsMetal_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	// Empty weight files fail in the loader before any dispatch.
	if _, err := DispatchPackedExpertsFromSafetensorsMetal(plan, nil, 0, [][]float32{{1, 0}}, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}); err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic", err)
	}
}

func TestM2Metal_ForwardPackedLayerMetal_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	// Hidden rows and router-score rows must match; the check precedes routing.
	_, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Plan:         plan,
		Hidden:       [][]float32{{1, 0}, {0, 1}},
		RouterScores: [][]float32{{0, 1}},
	})
	if err == nil || !core.Contains(err.Error(), "router rows") {
		t.Fatalf("error = %v, want hidden/router row mismatch", err)
	}
}

func TestM2Metal_ForwardPackedLayerFromSafetensorsMetal_Bad(t *testing.T) {
	plan := miniMaxM2SmallJANGTQPlan(t)
	// With a router bias supplied, the function takes the LoadRouter branch;
	// empty weight files make that branch fail cheaply (no device needed),
	// covering the bias-present path distinct from the lazy no-bias path.
	_, err := ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:       plan,
		Hidden:     [][]float32{{1, 0}},
		RouterBias: []float32{0, 0, 0},
	})
	if err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want router weight files diagnostic via bias branch", err)
	}
}

// --- Authored *Metal _Ugly completions (device-free error/edge legs) ---
//
// The Good legs above skip without a usable Metal device, so these _Ugly
// variants exercise the pre-dispatch guards that return before any kernel
// runs — no device, no model payload (AX-11).

func TestM2Metal_DispatchPackedExpertsFromSafetensorsMetal_Ugly(t *testing.T) {
	// An out-of-range layer makes the underlying LoadPackedExpertsForDecisions
	// fail in LayerTensorSpecs before any Metal dispatch, even with a real
	// weight file present.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	_, err := DispatchPackedExpertsFromSafetensorsMetal(plan, []string{weights}, 99, [][]float32{{1, 0}}, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: []int{2}, Weights: []float32{1}},
	})
	if err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want layer-range diagnostic before dispatch", err)
	}
}

func TestM2Metal_ForwardLazyExpertLoadMetal_Ugly(t *testing.T) {
	// A load whose decision references a token index outside the hidden batch
	// fails in the packed dispatch guard before any projection kernel runs.
	_, err := ForwardLazyExpertLoadMetal([][]float32{{1, 2}}, LazyExpertLoad{
		Decisions: []RouterDecision{{TokenIndex: 5, ExpertIDs: []int{0}, Weights: []float32{1}}},
		Experts:   map[int]PackedExpertWeights{0: {}},
	})
	if err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want token-index out-of-range from the dispatch guard", err)
	}
}

func TestM2Metal_ForwardPackedLayerMetal_Ugly(t *testing.T) {
	// Hidden and router-score row counts match (so the shape guard passes), but
	// an out-of-range layer makes the lazy expert load fail before any Metal
	// dispatch — the post-routing error leg distinct from the shape-mismatch Bad.
	plan := miniMaxM2SmallJANGTQPlan(t)
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, miniMaxM2LazyExpertFixtureTensors(t, 2, []uint8{0, 1, 2, 3}))
	_, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Plan:         plan,
		WeightFiles:  []string{weights},
		Layer:        99,
		Hidden:       [][]float32{{1, 0}},
		RouterScores: [][]float32{{0, 1, 2}},
	})
	if err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want layer-range diagnostic after routing", err)
	}
}

func TestM2Metal_ForwardPackedLayerFromSafetensorsMetal_Ugly(t *testing.T) {
	// No router bias takes the lazy LoadLazyExpertsForHidden branch; empty weight
	// files make that branch fail cheaply, the no-bias path distinct from the
	// bias-present LoadRouter Bad leg.
	plan := miniMaxM2SmallJANGTQPlan(t)
	_, err := ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:   plan,
		Hidden: [][]float32{{1, 0}},
	})
	if err == nil || !core.Contains(err.Error(), "weight files") {
		t.Fatalf("error = %v, want weight files diagnostic via the lazy no-bias branch", err)
	}
}

// --- Candidate-name resolvers (white-box, in-memory index) -------------
//
// findPackedWeightRef / findSidecarRef / findProjectionBiasRef and their
// try* helpers walk a fan-out of checkpoint name shapes against a
// safetensors.Index. The Good legs of the load tests hit the canonical
// "spec.Name is the tensor" path; these units drive the alias / suffix /
// sidecar-shape branches and the full-miss return. The index is a plain
// in-memory map[string]TensorRef — no file, no device, no model (AX-11).
