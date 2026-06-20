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
