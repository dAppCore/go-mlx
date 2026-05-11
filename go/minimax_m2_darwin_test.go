// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestMiniMaxM2_DispatchPackedExpertsMetalUsesFusedProjection_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	hidden := [][]float32{{1, 2}}
	decisions := []MiniMaxM2RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{0.75, 0.25},
	}}
	experts := map[int]MiniMaxM2PackedExpertWeights{
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

	got, err := DispatchMiniMaxM2PackedExpertsMetal(hidden, decisions, experts)
	if err != nil {
		t.Fatalf("DispatchMiniMaxM2PackedExpertsMetal() error = %v", err)
	}

	want := miniMaxM2PackedDispatchReference(t, hidden, decisions, experts)
	if len(got) != 1 || !float32SlicesRoughlyEqual(got[0], want[0], 1e-4) {
		t.Fatalf("got = %+v, want %+v", got, want)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsMetalRejectsMissingExpert_Bad(t *testing.T) {
	_, err := DispatchMiniMaxM2PackedExpertsMetal([][]float32{{1, 2}}, []MiniMaxM2RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{7},
		Weights:    []float32{1},
	}}, nil)
	if err == nil || !core.Contains(err.Error(), "missing expert 7") {
		t.Fatalf("error = %v, want missing expert diagnostic", err)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsMetalRejectsMalformedDecisions_Bad(t *testing.T) {
	if _, err := DispatchMiniMaxM2PackedExpertsMetal([][]float32{{1, 2}}, []MiniMaxM2RouterDecision{{
		TokenIndex: 2,
		ExpertIDs:  []int{0},
		Weights:    []float32{1},
	}}, nil); err == nil || !core.Contains(err.Error(), "out of range") {
		t.Fatalf("out-of-range error = %v", err)
	}
	if _, err := DispatchMiniMaxM2PackedExpertsMetal([][]float32{{1, 2}}, []MiniMaxM2RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{1},
	}}, nil); err == nil || !core.Contains(err.Error(), "length mismatch") {
		t.Fatalf("length mismatch error = %v", err)
	}
	if _, err := ForwardMiniMaxM2LazyExpertLoadMetal([][]float32{{1, 2}}, MiniMaxM2LazyExpertLoad{
		Decisions: []MiniMaxM2RouterDecision{{TokenIndex: 0, ExpertIDs: []int{3}, Weights: []float32{1}}},
	}); err == nil || !core.Contains(err.Error(), "missing expert") {
		t.Fatalf("lazy load error = %v, want missing expert", err)
	}
	if _, err := ForwardMiniMaxM2PackedLayerMetal(MiniMaxM2PackedLayerForwardOptions{
		Hidden:       [][]float32{{1, 2}},
		RouterScores: [][]float32{{1}, {2}},
	}); err == nil || !core.Contains(err.Error(), "hidden rows") {
		t.Fatalf("packed layer shape error = %v", err)
	}
	if got := miniMaxM2SwiGLU(0.5, 2); math.IsNaN(float64(got)) || got == 0 {
		t.Fatalf("miniMaxM2SwiGLU() = %v, want finite non-zero", got)
	}
}

func TestMiniMaxM2_DispatchPackedExpertsFromSafetensorsMetal_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg := MiniMaxM2Config{
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
	plan, err := BuildMiniMaxM2TensorPlan(cfg, &JANGQuantizationInfo{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
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
	decisions := []MiniMaxM2RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{0, 1},
		Weights:    []float32{0.75, 0.25},
	}}

	got, err := DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal(plan, []string{weights}, 0, hidden, decisions)
	if err != nil {
		t.Fatalf("DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal() error = %v", err)
	}
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors() error = %v", err)
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
	load, err := LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors(plan, []string{weights}, 0, hidden, []int32{42}, nil)
	if err != nil {
		t.Fatalf("LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors() error = %v", err)
	}

	got, err := ForwardMiniMaxM2LazyExpertLoadMetal(hidden, load)
	if err != nil {
		t.Fatalf("ForwardMiniMaxM2LazyExpertLoadMetal() error = %v", err)
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

	cfg := MiniMaxM2Config{
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
	plan, err := BuildMiniMaxM2TensorPlan(cfg, &JANGQuantizationInfo{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
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
	recorder := NewProbeRecorder()

	got, err := ForwardMiniMaxM2PackedLayerMetal(MiniMaxM2PackedLayerForwardOptions{
		Plan:         plan,
		WeightFiles:  []string{weights},
		Layer:        0,
		Hidden:       hidden,
		RouterScores: routerScores,
		TokenIDs:     []int32{101, 102},
		ProbeSink:    recorder,
	})
	if err != nil {
		t.Fatalf("ForwardMiniMaxM2PackedLayerMetal() error = %v", err)
	}

	decisions, err := RouteMiniMaxM2Tokens(cfg, routerScores, nil)
	if err != nil {
		t.Fatalf("RouteMiniMaxM2Tokens() error = %v", err)
	}
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors() error = %v", err)
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
	if events[0].Kind != ProbeEventRouterDecision || events[0].RouterDecision.TokenID != 101 || events[0].RouterDecision.Layer != 0 {
		t.Fatalf("first event = %+v, want router decision for token 101 layer 0", events[0])
	}
	if events[0].RouterDecision.ExpertIDs[0] != 1 || events[0].Meta["architecture"] != "minimax_m2" {
		t.Fatalf("first event router = %+v meta=%+v", events[0].RouterDecision, events[0].Meta)
	}
}

func TestMiniMaxM2_ForwardPackedLayerFromSafetensorsMetalProjectsRouter_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg := MiniMaxM2Config{
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
	plan, err := BuildMiniMaxM2TensorPlan(cfg, &JANGQuantizationInfo{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
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
	recorder := NewProbeRecorder()

	got, err := ForwardMiniMaxM2PackedLayerFromSafetensorsMetal(MiniMaxM2PackedLayerForwardOptions{
		Plan:        plan,
		WeightFiles: []string{weights},
		Layer:       0,
		Hidden:      hidden,
		TokenIDs:    []int32{201, 202},
		ProbeSink:   recorder,
	})
	if err != nil {
		t.Fatalf("ForwardMiniMaxM2PackedLayerFromSafetensorsMetal() error = %v", err)
	}

	router, err := LoadMiniMaxM2RouterFromSafetensors(plan, []string{weights}, 0)
	if err != nil {
		t.Fatalf("LoadMiniMaxM2RouterFromSafetensors() error = %v", err)
	}
	scores, err := ProjectMiniMaxM2RouterScores(hidden, router)
	if err != nil {
		t.Fatalf("ProjectMiniMaxM2RouterScores() error = %v", err)
	}
	decisions, err := RouteMiniMaxM2Tokens(cfg, scores, router.Bias)
	if err != nil {
		t.Fatalf("RouteMiniMaxM2Tokens() error = %v", err)
	}
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, []string{weights}, 0, decisions)
	if err != nil {
		t.Fatalf("LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors() error = %v", err)
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

func miniMaxM2PackedExpertFixture(t *testing.T, gateValues, upValues, downValues []uint8) MiniMaxM2PackedExpertWeights {
	t.Helper()
	return MiniMaxM2PackedExpertWeights{
		GateProj: miniMaxM2PackedProjectionFixture(t, "gate_proj", gateValues),
		UpProj:   miniMaxM2PackedProjectionFixture(t, "up_proj", upValues),
		DownProj: miniMaxM2PackedProjectionFixture(t, "down_proj", downValues),
	}
}

func miniMaxM2PackedProjectionFixture(t *testing.T, projection string, values []uint8) JANGPackedProjectionTensor {
	t.Helper()
	desc := JANGPackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0." + projection + ".weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          JANGTensorRoleRoutedExpert,
		Shape:         []uint64{2, 2},
		Elements:      4,
		Bits:          2,
		GroupSize:     4,
		Groups:        1,
		PackedBytes:   1,
		ValuesPerByte: 4,
		ScaleCount:    1,
		BiasCount:     1,
		BitOrder:      JANGBitOrderLSB0,
		Encoding:      JANGEncodingAffine,
	}
	packed, err := PackJANGQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("PackJANGQuantizedValues(%s) error = %v", projection, err)
	}
	return JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1},
		Biases:     []float32{0},
	}
}

func miniMaxM2PackedDispatchReference(t *testing.T, hidden [][]float32, decisions []MiniMaxM2RouterDecision, experts map[int]MiniMaxM2PackedExpertWeights) [][]float32 {
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

func miniMaxM2PackedExpertReference(t *testing.T, hidden []float32, expert MiniMaxM2PackedExpertWeights) []float32 {
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
	weight, err := DequantizeJANGPackedTensor(projection.Descriptor, projection.Packed, projection.Scales, projection.Biases)
	if err != nil {
		t.Fatalf("DequantizeJANGPackedTensor() error = %v", err)
	}
	outDim := int(projection.Descriptor.Shape[0])
	inDim := int(projection.Descriptor.Shape[1])
	return denseProjectionReference(input, 1, weight, outDim, inDim, projection.Bias)
}
