// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
)

// These tests close remaining statement-coverage gaps in m2_metal.go. The
// device-live legs reuse the same tiny JANGTQ fixtures as the existing _Good
// metal tests (skipIfNoUsableMetal); the guard legs are device-free and return
// before any kernel runs (AX-11).

// --- ForwardPackedLayerFromSafetensorsMetal: bias-present (LoadRouter) leg ---

func TestM2MetalCover_ForwardPackedLayerFromSafetensorsMetal_BiasBranch(t *testing.T) {
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

	// Supplying RouterBias drives the LoadRouter + ProjectRouterScores branch
	// (distinct from the lazy no-bias path the existing _Good test takes).
	got, err := ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:        plan,
		WeightFiles: []string{weights},
		Layer:       0,
		Hidden:      hidden,
		RouterBias:  []float32{0, 0.25, 0.5},
		TokenIDs:    []int32{301, 302},
		ProbeSink:   recorder,
	})
	if err != nil {
		t.Fatalf("ForwardPackedLayerFromSafetensorsMetal(bias) error = %v", err)
	}
	if len(got.Output) != 2 {
		t.Fatalf("output rows = %d, want 2", len(got.Output))
	}
	if len(got.SelectedExpertIDs) != 2 || got.SelectedExpertIDs[0] != 1 || got.SelectedExpertIDs[1] != 2 {
		t.Fatalf("selected experts = %+v, want [1 2]", got.SelectedExpertIDs)
	}
	if len(recorder.Events()) != 2 {
		t.Fatalf("events = %d, want one router probe per token", len(recorder.Events()))
	}
}

// --- ForwardPackedLayerMetal: RouteTokens error (post shape-check) -------

func TestM2MetalCover_ForwardPackedLayerMetal_RouteError(t *testing.T) {
	// Hidden rows and router-score rows match (shape guard passes), but each
	// score row's width disagrees with the local expert count, so RouteTokens
	// rejects after the shape check and before any expert load — device-free.
	plan := miniMaxM2SmallJANGTQPlan(t) // NumLocalExperts = 3
	_, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Plan:         plan,
		Hidden:       [][]float32{{1, 0}},
		RouterScores: [][]float32{{0.1, 0.2}}, // width 2 ≠ 3 experts
	})
	if err == nil || !core.Contains(err.Error(), "expected 3") {
		t.Fatalf("error = %v, want RouteTokens row-width diagnostic", err)
	}
}

// --- runPackedExpertMetal: gate/up size mismatch ------------------------

func TestM2MetalCover_runPackedExpertMetal_GateUpSizeMismatch(t *testing.T) {
	skipIfNoUsableMetal(t)

	// Two well-formed packed projections whose output dimensions differ: the
	// gate projects to 2 rows, the up projection to 3. Both project cleanly on
	// device, but runPackedExpertMetal rejects the gate/up length mismatch.
	gate := miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3}) // shape [2,2]
	up := miniMaxM2WidePackedProjectionForCover(t, "up_proj")                     // shape [3,2]
	down := miniMaxM2PackedProjectionFixture(t, "down_proj", []uint8{0, 1, 2, 3})
	experts := map[int]PackedExpertWeights{
		0: {GateProj: gate, UpProj: up, DownProj: down},
	}
	decisions := []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, decisions, experts); err == nil || !core.Contains(err.Error(), "gate/up size mismatch") {
		t.Fatalf("error = %v, want gate/up size mismatch diagnostic", err)
	}
}

// --- ForwardPackedLayerFromSafetensorsMetal: bias-branch ProjectRouterScores error ---

func TestM2MetalCover_ForwardPackedLayerFromSafetensorsMetal_BiasProjectError(t *testing.T) {
	// RouterBias is supplied (so the LoadRouter branch runs), the router gate
	// loads, but a hidden row whose width disagrees with the router hidden size
	// makes ProjectRouterScores fail — the post-LoadRouter error leg of the
	// bias branch. ProjectRouterScores is a host projection, so no device is
	// needed.
	cfg := Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 2,
		UseRoutingBias: true,
	}
	plan, err := BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildTensorPlan() error = %v", err)
	}
	dir := t.TempDir()
	weights := core.PathJoin(dir, "model.safetensors")
	writeMiniMaxM2RawSafetensors(t, weights, []miniMaxM2RawSafetensor{
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0, 0, 1, 1, 1}, 3, 2),
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0, 0}, 3),
	})
	_, err = ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:        plan,
		WeightFiles: []string{weights},
		Layer:       0,
		Hidden:      [][]float32{{1, 0, 0}}, // width 3 ≠ router hidden size 2
		RouterBias:  []float32{0, 0, 0},
	})
	if err == nil || !core.Contains(err.Error(), "hidden row") {
		t.Fatalf("error = %v, want ProjectRouterScores hidden-row diagnostic in the bias branch", err)
	}
}

// --- DispatchPackedExpertsMetal: expert output shape mismatch -----------

func TestM2MetalCover_DispatchPackedExpertsMetal_OutputShapeMismatch(t *testing.T) {
	skipIfNoUsableMetal(t)

	// Two experts routed to the same token whose down projections yield
	// different output lengths: the first sets the token's output width, the
	// second's differing length trips the output-shape-mismatch guard. Both
	// experts are well-formed and project cleanly on device.
	narrowGate := miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3}) // [2,2]
	narrowUp := miniMaxM2PackedProjectionFixture(t, "up_proj", []uint8{0, 1, 2, 3})     // [2,2]
	narrowDown := miniMaxM2PackedProjectionFixture(t, "down_proj", []uint8{0, 1, 2, 3}) // [2,2] → 2 outputs

	// Wide expert: gate/up are [3,2] so swiGLU produces 3 activations, and the
	// down projection [2,3] maps them back to 2 outputs — but we instead use a
	// down projection that yields 3 outputs to force the length difference.
	wideGate := miniMaxM2WidePackedProjectionForCover(t, "gate_proj") // [3,2] → 3
	wideUp := miniMaxM2WidePackedProjectionForCover(t, "up_proj")     // [3,2] → 3
	wideDown := miniMaxM2WideDownPackedProjectionForCover(t)          // [3,3] → 3 outputs

	experts := map[int]PackedExpertWeights{
		0: {GateProj: narrowGate, UpProj: narrowUp, DownProj: narrowDown},
		1: {GateProj: wideGate, UpProj: wideUp, DownProj: wideDown},
	}
	decisions := []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0, 1}, Weights: []float32{0.5, 0.5}}}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, decisions, experts); err == nil || !core.Contains(err.Error(), "output shape mismatch") {
		t.Fatalf("error = %v, want expert output shape mismatch diagnostic", err)
	}
}

// miniMaxM2WideDownPackedProjectionForCover builds a valid [3,3] packed down
// projection (3 inputs → 3 outputs) so a wide expert produces a 3-wide output,
// differing from the 2-wide narrow expert in the same dispatch.
func miniMaxM2WideDownPackedProjectionForCover(t *testing.T) JANGPackedProjectionTensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.1.down_proj.weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{3, 3},
		Elements:      9,
		Bits:          2,
		GroupSize:     4,
		Groups:        3,
		PackedBytes:   3,
		ValuesPerByte: 4,
		ScaleCount:    3,
		BiasCount:     3,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, []uint8{0, 1, 2, 3, 1, 2, 3, 0, 1})
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues(down) error = %v", err)
	}
	return JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1, 1, 1},
		Biases:     []float32{0, 0, 0},
	}
}

// --- runPackedExpertMetal: gate / up / down projection error legs -------
//
// Each projection error is driven by a *valid* packed tensor whose declared
// input dimension disagrees with the activation width it is handed. The fused
// projection validates that mismatch and returns a clean error before any
// kernel dispatch (jang.projectPackedTensor checks input-last-dim vs
// weight-in-dim ahead of metal.FromValues), so no malformed descriptor ever
// reaches the GPU.

func TestM2MetalCover_runPackedExpertMetal_GateProjectError(t *testing.T) {
	skipIfNoUsableMetal(t)
	// Gate expects a 3-wide input but the hidden row is 2-wide → gate fails.
	gate := miniMaxM2InDimPackedProjectionForCover(t, "gate_proj", 3)
	expert := PackedExpertWeights{
		GateProj: gate,
		UpProj:   miniMaxM2PackedProjectionFixture(t, "up_proj", []uint8{0, 1, 2, 3}),
		DownProj: miniMaxM2PackedProjectionFixture(t, "down_proj", []uint8{0, 1, 2, 3}),
	}
	decisions := []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, decisions, map[int]PackedExpertWeights{0: expert}); err == nil || !core.Contains(err.Error(), "gate_proj") {
		t.Fatalf("error = %v, want gate_proj projection failure", err)
	}
}

func TestM2MetalCover_runPackedExpertMetal_UpProjectError(t *testing.T) {
	skipIfNoUsableMetal(t)
	// Gate is well-formed (2→2) so it projects, but up expects a 3-wide input
	// against the same 2-wide hidden row → up fails.
	expert := PackedExpertWeights{
		GateProj: miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3}),
		UpProj:   miniMaxM2InDimPackedProjectionForCover(t, "up_proj", 3),
		DownProj: miniMaxM2PackedProjectionFixture(t, "down_proj", []uint8{0, 1, 2, 3}),
	}
	decisions := []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, decisions, map[int]PackedExpertWeights{0: expert}); err == nil || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want up_proj projection failure", err)
	}
}

func TestM2MetalCover_runPackedExpertMetal_DownProjectError(t *testing.T) {
	skipIfNoUsableMetal(t)
	// Gate and up are well-formed (2→2) so swiGLU yields 2 activations, but the
	// down projection expects a 3-wide input → down fails.
	expert := PackedExpertWeights{
		GateProj: miniMaxM2PackedProjectionFixture(t, "gate_proj", []uint8{0, 1, 2, 3}),
		UpProj:   miniMaxM2PackedProjectionFixture(t, "up_proj", []uint8{0, 1, 2, 3}),
		DownProj: miniMaxM2InDimPackedProjectionForCover(t, "down_proj", 3),
	}
	decisions := []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}}}
	if _, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, decisions, map[int]PackedExpertWeights{0: expert}); err == nil || !core.Contains(err.Error(), "down_proj") {
		t.Fatalf("error = %v, want down_proj projection failure", err)
	}
}

// miniMaxM2InDimPackedProjectionForCover builds a valid packed projection whose
// input dimension is inDim (out-dim fixed at 2). It is itself well-formed, so
// the projection fails only on the input-width mismatch, never on validation.
func miniMaxM2InDimPackedProjectionForCover(t *testing.T, projection string, inDim int) JANGPackedProjectionTensor {
	t.Helper()
	elements := uint64(2 * inDim)
	values := make([]uint8, elements)
	for i := range values {
		values[i] = uint8(i % 4)
	}
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0." + projection + ".weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{2, uint64(inDim)},
		Elements:      elements,
		Bits:          2,
		GroupSize:     4,
		Groups:        int((elements + 3) / 4),
		PackedBytes:   int((elements*2 + 7) / 8),
		ValuesPerByte: 4,
		ScaleCount:    int((elements + 3) / 4),
		BiasCount:     int((elements + 3) / 4),
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues(%s) error = %v", projection, err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 1
	}
	return JANGPackedProjectionTensor{Descriptor: desc, Packed: packed, Scales: scales, Biases: biases}
}

// miniMaxM2WidePackedProjectionForCover builds a valid packed projection with a
// 3x2 shape (out-dim 3) so it projects cleanly on device but produces a
// different output length than the standard 2x2 fixtures.
func miniMaxM2WidePackedProjectionForCover(t *testing.T, projection string) JANGPackedProjectionTensor {
	t.Helper()
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0." + projection + ".weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{3, 2},
		Elements:      6,
		Bits:          2,
		GroupSize:     4,
		Groups:        2,
		PackedBytes:   2,
		ValuesPerByte: 4,
		ScaleCount:    2,
		BiasCount:     2,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, []uint8{0, 1, 2, 3, 1, 2})
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues(%s) error = %v", projection, err)
	}
	return JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1, 1},
		Biases:     []float32{0, 0},
	}
}
