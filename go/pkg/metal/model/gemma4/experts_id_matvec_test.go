// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin Gemma4Experts.forward / forwardExpertIDMatVec — the expert-id
// matvec fast paths must match the gather-QMM reference across the fused
// gate_up, split gate/up, split fused-activation, and sorted-prefill variants.
// They moved here from package metal's expert_id_matvec_test.go with the
// Gemma4Experts type and its decode methods; the runtime gates are driven via
// the public metal.SetRuntimeGate seam.

func TestExpertIDMatVec_Gemma4ExpertsOptInMatchesGatherQMM_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec Gemma4ExpertsOptInMatchesGatherQMM"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 3
		routes    = 2
		hidden    = 8
		moeDim    = 8
		groupSize = 4
		bits      = 4
	)
	layer := &Gemma4Experts{
		GateUpProj: quantizedSwitchLinearExpertIDTest(t, experts, moeDim*2, hidden, groupSize, bits, 3),
		DownProj:   quantizedSwitchLinearExpertIDTest(t, experts, hidden, moeDim, groupSize, bits, 11),
	}
	defer func() {
		metal.FreeSwitchLinear(layer.GateUpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	x := metal.FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := metal.FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := metal.FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer metal.Free(x, topKIndices, topKWeights)

	restoreOff := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer metal.Free(want)

	phases := map[string]bool{}
	restoreOn := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*metal.Array) {
		phases[phase] = true
	})
	restoreOn()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the fused gate_up path")
	}
	defer metal.Free(got)
	metal.Materialize(want, got)

	if !phases["gate_up_id_matvec"] || !phases["activation_id_matvec"] || !phases["down_weighted_sum_id_matvec"] {
		t.Fatalf("expert id phases = %+v, want fused gate_up, activation, and weighted down", phases)
	}
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 5e-4)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, hidden)
	}
}

func TestExpertIDMatVec_Gemma4ExpertsSplitGateUpOptInMatchesGatherQMM_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec Gemma4ExpertsSplitGateUpOptInMatchesGatherQMM"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 3
		routes    = 2
		hidden    = 8
		moeDim    = 8
		groupSize = 4
		bits      = 4
	)
	layer := &Gemma4Experts{
		GateProj: quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 3),
		UpProj:   quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 5),
		DownProj: quantizedSwitchLinearExpertIDTest(t, experts, hidden, moeDim, groupSize, bits, 11),
	}
	quantizedSwitchLinearSidecarsAsType(layer.GateProj, metal.DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.UpProj, metal.DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.DownProj, metal.DTypeBFloat16)
	defer func() {
		metal.FreeSwitchLinear(layer.GateProj)
		metal.FreeSwitchLinear(layer.UpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	x := metal.FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := metal.FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := metal.FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer metal.Free(x, topKIndices, topKWeights)

	restoreOff := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer metal.Free(want)

	phases := map[string]bool{}
	restoreOn := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*metal.Array) {
		phases[phase] = true
	})
	restoreOn()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the split gate/up path")
	}
	defer metal.Free(got)
	metal.Materialize(want, got)

	if !phases["up_id_matvec"] || !phases["gate_id_matvec"] || !phases["activation_id_matvec"] || !phases["down_weighted_sum_id_matvec"] {
		t.Fatalf("expert id phases = %+v, want split gate/up, activation, and weighted down", phases)
	}
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 1e-3)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, hidden)
	}
}

func TestExpertIDMatVec_Gemma4ExpertsSplitGateUpFusedActivationMatchesGatherQMM_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec Gemma4ExpertsSplitGateUpFusedActivationMatchesGatherQMM"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 3
		routes    = 2
		hidden    = 8
		moeDim    = 8
		groupSize = 4
		bits      = 4
	)
	layer := &Gemma4Experts{
		GateProj: quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 3),
		UpProj:   quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 5),
		DownProj: quantizedSwitchLinearExpertIDTest(t, experts, hidden, moeDim, groupSize, bits, 11),
	}
	quantizedSwitchLinearSidecarsAsType(layer.GateProj, metal.DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.UpProj, metal.DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.DownProj, metal.DTypeBFloat16)
	defer func() {
		metal.FreeSwitchLinear(layer.GateProj)
		metal.FreeSwitchLinear(layer.UpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	x := metal.FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := metal.FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := metal.FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer metal.Free(x, topKIndices, topKWeights)

	restoreOff := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer metal.Free(want)

	phases := map[string]bool{}
	restoreMatVec := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	restoreFused := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION", "1")
	restoreUnrolled := metal.SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*metal.Array) {
		phases[phase] = true
	})
	restoreUnrolled()
	restoreFused()
	restoreMatVec()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the split fused-activation path")
	}
	defer metal.Free(got)
	metal.Materialize(want, got)

	if !phases["activation_split_id_matvec"] || !phases["down_weighted_sum_id_matvec"] {
		t.Fatalf("expert id phases = %+v, want split fused activation and weighted down", phases)
	}
	if phases["up_id_matvec"] || phases["gate_id_matvec"] {
		t.Fatalf("expert id phases = %+v, split fused activation should not materialise separate gate/up", phases)
	}
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 1e-3)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, hidden)
	}
}

func TestExpertIDMatVec_Gemma4SortedExpertPrefillMatchesGatherQMM_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec Gemma4SortedExpertPrefillMatchesGatherQMM"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 2
		seqLen    = 16
		topK      = 1
		hidden    = 8
		moeDim    = 8
		groupSize = 4
		bits      = 4
	)
	layer := &Gemma4Experts{
		GateProj: quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 3),
		UpProj:   quantizedSwitchLinearExpertIDTest(t, experts, moeDim, hidden, groupSize, bits, 5),
		DownProj: quantizedSwitchLinearExpertIDTest(t, experts, hidden, moeDim, groupSize, bits, 11),
	}
	defer func() {
		metal.FreeSwitchLinear(layer.GateProj)
		metal.FreeSwitchLinear(layer.UpProj)
		metal.FreeSwitchLinear(layer.DownProj)
	}()

	values := make([]float32, seqLen*hidden)
	for i := range values {
		values[i] = float32((i%11)-5) * 0.125
	}
	indices := make([]int32, seqLen*topK)
	weights := make([]float32, seqLen*topK)
	for i := range indices {
		indices[i] = int32((i + 1) % experts)
		weights[i] = 0.5 + 0.025*float32(i%5)
	}
	x := metal.FromValues(values, 1, seqLen, hidden)
	topKIndices := metal.FromValues(indices, 1, seqLen, topK)
	topKWeights := metal.FromValues(weights, 1, seqLen, topK)
	defer metal.Free(x, topKIndices, topKWeights)

	restoreOff := metal.SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer metal.Free(want)

	restoreOn := metal.SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "1")
	got := layer.forward(x, topKIndices, topKWeights, "")
	restoreOn()
	defer metal.Free(got)

	metal.Materialize(want, got)
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 6e-4)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 %d %d]", shape, seqLen, hidden)
	}
}
