// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	"dappco.re/go"
)

func TestExpertIDMatVec_QuantizedQ4MatchesCPUReference_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec QuantizedQ4MatchesCPUReference"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 3
		routes    = 2
		outDim    = 3
		inDim     = 8
		groupSize = 4
		bits      = 4
	)
	quantized := []uint8{
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 1, 0, 3, 4, 5, 6, 7,
		9, 8, 7, 6, 5, 4, 3, 2,

		0, 1, 1, 2, 3, 5, 8, 13,
		13, 8, 5, 3, 2, 1, 1, 0,
		4, 4, 4, 4, 2, 2, 2, 2,

		15, 14, 13, 12, 11, 10, 9, 8,
		8, 9, 10, 11, 12, 13, 14, 15,
		3, 6, 9, 12, 1, 4, 7, 10,
	}
	scales := []float32{
		0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
		0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
		0.12, 0.22, 0.32, 0.42, 0.52, 0.62,
	}
	qbiases := []float32{
		-0.5, 0.25, -0.25, 0.5, 0.75, -0.75,
		0.1, -0.2, 0.3, -0.4, 0.5, -0.6,
		-1.0, 1.0, -1.5, 1.5, -2.0, 2.0,
	}
	inputValues := []float32{
		0.25, -0.5, 1.25, 2.0, -1.0, 0.75, 0.5, -0.25,
		-0.75, 0.5, 1.5, -1.25, 0.25, 2.25, -0.5, 0.125,
	}
	ids := []int32{2, 0}

	input := FromValues(inputValues, routes, inDim)
	weight := FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits))
	scaleArray := FromValues(scales, experts, outDim, inDim/groupSize)
	biasArray := FromValues(qbiases, experts, outDim, inDim/groupSize)
	idArray := FromValues(ids, routes)
	defer Free(input, weight, scaleArray, biasArray, idArray)

	gotArray, err := quantizedExpertIDMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("quantizedExpertIDMatVec() error = %v", err)
	}
	defer Free(gotArray)
	Materialize(gotArray)

	want := quantizedExpertIDMatVecCPUReference(inputValues, quantized, scales, qbiases, ids, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, gotArray.Floats(), want, 1e-4)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != routes || shape[1] != outDim {
		t.Fatalf("shape = %+v, want [%d %d]", shape, routes, outDim)
	}
}

func TestExpertIDMatVec_QuantizedQ4SIMDWideInput_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec QuantizedQ4SIMDWideInput"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 4
		routes    = 3
		outDim    = 5
		inDim     = 64
		groupSize = 16
		bits      = 4
	)
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*7 + 3) & 15)
	}
	scales := make([]float32, experts*outDim*(inDim/groupSize))
	qbiases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.03125 * float32((i%11)+1)
		qbiases[i] = -0.75 + 0.125*float32(i%13)
	}
	inputValues := make([]float32, routes*inDim)
	for i := range inputValues {
		inputValues[i] = -1.5 + 0.0625*float32((i*5)%71)
	}
	ids := []int32{3, 1, 0}

	input := FromValues(inputValues, routes, inDim)
	weight := FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits))
	scaleArray := FromValues(scales, experts, outDim, inDim/groupSize)
	biasArray := FromValues(qbiases, experts, outDim, inDim/groupSize)
	idArray := FromValues(ids, routes)
	defer Free(input, weight, scaleArray, biasArray, idArray)

	gotArray, err := quantizedExpertIDMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("quantizedExpertIDMatVec() error = %v", err)
	}
	defer Free(gotArray)
	Materialize(gotArray)

	want := quantizedExpertIDMatVecCPUReference(inputValues, quantized, scales, qbiases, ids, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, gotArray.Floats(), want, 2e-4)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != routes || shape[1] != outDim {
		t.Fatalf("shape = %+v, want [%d %d]", shape, routes, outDim)
	}
}

func TestExpertIDMatVec_GELUGateUpMatchesCPUReference_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec GELUGateUpMatchesCPUReference"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 3
		routes    = 2
		outDim    = 8
		inDim     = 32
		groupSize = 8
		bits      = 4
	)
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*11 + 7) & 15)
	}
	scales := make([]float32, experts*outDim*(inDim/groupSize))
	qbiases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.02 * float32((i%13)+1)
		qbiases[i] = -0.5 + 0.0625*float32((i*3)%19)
	}
	inputValues := make([]float32, routes*inDim)
	for i := range inputValues {
		inputValues[i] = -1.25 + 0.03125*float32((i*7)%83)
	}
	ids := []int32{2, 0}

	input := FromValues(inputValues, routes, inDim)
	weight := FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits))
	scaleArray := FromValues(scales, experts, outDim, inDim/groupSize)
	biasArray := FromValues(qbiases, experts, outDim, inDim/groupSize)
	idArray := FromValues(ids, routes)
	defer Free(input, weight, scaleArray, biasArray, idArray)

	gotArray, err := quantizedExpertIDGELUGateUpMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("quantizedExpertIDGELUGateUpMatVec() error = %v", err)
	}
	defer Free(gotArray)
	Materialize(gotArray)

	want := quantizedExpertIDGELUGateUpMatVecCPUReference(inputValues, quantized, scales, qbiases, ids, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, gotArray.Floats(), want, 5e-4)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != routes || shape[1] != outDim/2 {
		t.Fatalf("shape = %+v, want [%d %d]", shape, routes, outDim/2)
	}
}

func TestExpertIDMatVec_WeightedMatVecSumMatchesCPUReference_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec WeightedMatVecSumMatchesCPUReference"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		experts   = 4
		routes    = 3
		outDim    = 6
		inDim     = 32
		groupSize = 8
		bits      = 4
	)
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*5 + 9) & 15)
	}
	scales := make([]float32, experts*outDim*(inDim/groupSize))
	qbiases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.04 * float32((i%7)+1)
		qbiases[i] = -0.35 + 0.075*float32(i%11)
	}
	inputValues := make([]float32, routes*inDim)
	for i := range inputValues {
		inputValues[i] = -1.0 + 0.05*float32((i*3)%59)
	}
	routeWeights := []float32{0.5, 0.3, 0.2}
	ids := []int32{2, 0, 3}

	input := FromValues(inputValues, routes, inDim)
	weightArray := FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits))
	scaleArray := FromValues(scales, experts, outDim, inDim/groupSize)
	biasArray := FromValues(qbiases, experts, outDim, inDim/groupSize)
	routeWeightArray := FromValues(routeWeights, routes)
	idArray := FromValues(ids, routes)
	defer Free(input, weightArray, scaleArray, biasArray, routeWeightArray, idArray)

	gotArray, err := quantizedExpertIDWeightedMatVecSum(input, routeWeightArray, weightArray, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("quantizedExpertIDWeightedMatVecSum() error = %v", err)
	}
	defer Free(gotArray)
	Materialize(gotArray)

	want := quantizedExpertIDWeightedMatVecSumCPUReference(inputValues, routeWeights, quantized, scales, qbiases, ids, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, gotArray.Floats(), want, 3e-4)
	if shape := gotArray.Shape(); len(shape) != 1 || shape[0] != outDim {
		t.Fatalf("shape = %+v, want [%d]", shape, outDim)
	}
}

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
		freeSwitchLinear(layer.GateUpProj)
		freeSwitchLinear(layer.DownProj)
	}()

	x := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer Free(x, topKIndices, topKWeights)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer Free(want)

	phases := map[string]bool{}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*Array) {
		phases[phase] = true
	})
	restoreOn()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the fused gate_up path")
	}
	defer Free(got)
	Materialize(want, got)

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
	quantizedSwitchLinearSidecarsAsType(layer.GateProj, DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.UpProj, DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.DownProj, DTypeBFloat16)
	defer func() {
		freeSwitchLinear(layer.GateProj)
		freeSwitchLinear(layer.UpProj)
		freeSwitchLinear(layer.DownProj)
	}()

	x := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer Free(x, topKIndices, topKWeights)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer Free(want)

	phases := map[string]bool{}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*Array) {
		phases[phase] = true
	})
	restoreOn()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the split gate/up path")
	}
	defer Free(got)
	Materialize(want, got)

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
	quantizedSwitchLinearSidecarsAsType(layer.GateProj, DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.UpProj, DTypeBFloat16)
	quantizedSwitchLinearSidecarsAsType(layer.DownProj, DTypeBFloat16)
	defer func() {
		freeSwitchLinear(layer.GateProj)
		freeSwitchLinear(layer.UpProj)
		freeSwitchLinear(layer.DownProj)
	}()

	x := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	topKIndices := FromValues([]int32{2, 0}, 1, 1, routes)
	topKWeights := FromValues([]float32{0.65, 0.35}, 1, 1, routes)
	defer Free(x, topKIndices, topKWeights)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer Free(want)

	phases := map[string]bool{}
	restoreMatVec := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")
	restoreFused := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION", "1")
	restoreUnrolled := SetRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4", "1")
	got, ok := layer.forwardExpertIDMatVec(x, topKIndices, topKWeights, func(phase string, _ ...*Array) {
		phases[phase] = true
	})
	restoreUnrolled()
	restoreFused()
	restoreMatVec()
	if !ok {
		t.Fatal("forwardExpertIDMatVec() did not take the split fused-activation path")
	}
	defer Free(got)
	Materialize(want, got)

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
		freeSwitchLinear(layer.GateProj)
		freeSwitchLinear(layer.UpProj)
		freeSwitchLinear(layer.DownProj)
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
	x := FromValues(values, 1, seqLen, hidden)
	topKIndices := FromValues(indices, 1, seqLen, topK)
	topKWeights := FromValues(weights, 1, seqLen, topK)
	defer Free(x, topKIndices, topKWeights)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "0")
	want := layer.forward(x, topKIndices, topKWeights, "")
	restoreOff()
	defer Free(want)

	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "1")
	got := layer.forward(x, topKIndices, topKWeights, "")
	restoreOn()
	defer Free(got)

	Materialize(want, got)
	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 6e-4)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 %d %d]", shape, seqLen, hidden)
	}
}

func TestExpertIDMatVec_KernelCacheReusesShape_Good(t *testing.T) {
	coverageTokens := "ExpertIDMatVec KernelCacheReusesShape"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 8)
	weight := FromValues([]uint32{0, 0}, 1, 2, 1)
	scales := FromValues([]float32{1, 1, 1, 1}, 1, 2, 2)
	biases := FromValues([]float32{0, 0, 0, 0}, 1, 2, 2)
	ids := FromValues([]int32{0}, 1)
	defer Free(input, weight, scales, biases, ids)

	meta, err := validateQuantizedExpertIDMatVec(input, weight, scales, biases, ids, 4, 4)
	if err != nil {
		t.Fatalf("validateQuantizedExpertIDMatVec() error = %v", err)
	}
	first := quantizedExpertIDMatVecKernel(meta, 4, 4)
	second := quantizedExpertIDMatVecKernel(meta, 4, 4)
	if first == nil || second == nil {
		t.Fatal("cached kernels should be non-nil")
	}
	if first != second {
		t.Fatal("same expert-id matvec shape should reuse the cached kernel")
	}

	routeWeights := FromValues([]float32{1}, 1)
	defer Free(routeWeights)
	firstWeighted := quantizedExpertIDWeightedMatVecSumKernel(meta, 4, 4)
	secondWeighted := quantizedExpertIDWeightedMatVecSumKernel(meta, 4, 4)
	if firstWeighted == nil || secondWeighted == nil {
		t.Fatal("cached weighted kernels should be non-nil")
	}
	if firstWeighted != secondWeighted {
		t.Fatal("same expert-id weighted matvec shape should reuse the cached kernel")
	}

	firstGateUp := quantizedExpertIDGELUGateUpMatVecKernel(meta, 4, 4)
	secondGateUp := quantizedExpertIDGELUGateUpMatVecKernel(meta, 4, 4)
	if firstGateUp == nil || secondGateUp == nil {
		t.Fatal("cached gate/up kernels should be non-nil")
	}
	if firstGateUp != secondGateUp {
		t.Fatal("same expert-id gate/up shape should reuse the cached kernel")
	}

	firstSplitGateUp := quantizedExpertIDGELUSplitGateUpMatVecKernel(meta, 4, 4)
	secondSplitGateUp := quantizedExpertIDGELUSplitGateUpMatVecKernel(meta, 4, 4)
	if firstSplitGateUp == nil || secondSplitGateUp == nil {
		t.Fatal("cached split gate/up kernels should be non-nil")
	}
	if firstSplitGateUp != secondSplitGateUp {
		t.Fatal("same expert-id split gate/up shape should reuse the cached kernel")
	}
}

func TestExpertIDMatVec_RejectsBadMetadata_Bad(t *testing.T) {
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	weight := FromValues([]uint32{0}, 1, 1, 1)
	scales := FromValues([]float32{1}, 1, 1, 1)
	biases := FromValues([]float32{0}, 1, 1, 1)
	ids := FromValues([]int32{0, 0, 0}, 3)
	defer Free(input, weight, scales, biases, ids)

	_, err := quantizedExpertIDMatVec(input, weight, scales, biases, ids, 4, 4)
	if err == nil || !core.Contains(err.Error(), "input row count") {
		t.Fatalf("error = %v, want input row count diagnostic", err)
	}

	validIDs := FromValues([]int32{0}, 1)
	defer Free(validIDs)
	_, err = quantizedExpertIDMatVec(input, weight, scales, biases, validIDs, 4, 3)
	if err == nil || !core.Contains(err.Error(), "unsupported bits") {
		t.Fatalf("error = %v, want unsupported bits diagnostic", err)
	}
}

func TestExpertIDMatVec_RejectsNonPackedShape_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 6)
	weight := FromValues([]uint32{0}, 1, 1, 1)
	scales := FromValues([]float32{1}, 1, 1, 1)
	biases := FromValues([]float32{0}, 1, 1, 1)
	ids := FromValues([]int32{0}, 1)
	defer Free(input, weight, scales, biases, ids)

	_, err := quantizedExpertIDMatVec(input, weight, scales, biases, ids, 4, 4)
	if err == nil || !core.Contains(err.Error(), "divide by group size") {
		t.Fatalf("error = %v, want group-size diagnostic", err)
	}
}

func packMLXAffineQ4TestRows(t *testing.T, values []uint8) []uint32 {
	t.Helper()
	if len(values)%8 != 0 {
		t.Fatalf("q4 test rows must have a multiple of 8 values, got %d", len(values))
	}
	packed := make([]uint32, len(values)/8)
	for i, value := range values {
		if value > 15 {
			t.Fatalf("q4 value %d exceeds 15", value)
		}
		packed[i/8] |= uint32(value) << uint((i%8)*4)
	}
	return packed
}

func quantizedExpertIDMatVecCPUReference(input []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	out := make([]float32, len(ids)*outDim)
	for route, expertID := range ids {
		expert := int(expertID)
		for outCol := 0; outCol < outDim; outCol++ {
			var sum float32
			for inCol := 0; inCol < inDim; inCol++ {
				weightIndex := (expert*outDim+outCol)*inDim + inCol
				group := inCol / groupSize
				scaleIndex := (expert*outDim+outCol)*groups + group
				w := float32(quantized[weightIndex])*scales[scaleIndex] + biases[scaleIndex]
				sum += input[route*inDim+inCol] * w
			}
			out[route*outDim+outCol] = sum
		}
	}
	return out
}

func quantizedExpertIDGELUGateUpMatVecCPUReference(input []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	halfOut := outDim / 2
	out := make([]float32, len(ids)*halfOut)
	for route, expertID := range ids {
		expert := int(expertID)
		for outCol := 0; outCol < halfOut; outCol++ {
			var gateSum, upSum float32
			for inCol := 0; inCol < inDim; inCol++ {
				group := inCol / groupSize
				gateWeightIndex := (expert*outDim+outCol)*inDim + inCol
				upWeightIndex := (expert*outDim+outCol+halfOut)*inDim + inCol
				gateScaleIndex := (expert*outDim+outCol)*groups + group
				upScaleIndex := (expert*outDim+outCol+halfOut)*groups + group
				gateWeight := float32(quantized[gateWeightIndex])*scales[gateScaleIndex] + biases[gateScaleIndex]
				upWeight := float32(quantized[upWeightIndex])*scales[upScaleIndex] + biases[upScaleIndex]
				inputValue := input[route*inDim+inCol]
				gateSum += inputValue * gateWeight
				upSum += inputValue * upWeight
			}
			out[route*halfOut+outCol] = geluApproxFloat32(gateSum) * upSum
		}
	}
	return out
}

func geluApproxFloat32(x float32) float32 {
	cube := x * x * x
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*cube)))))
}

func quantizedExpertIDWeightedMatVecSumCPUReference(input, routeWeights []float32, quantized []uint8, scales, biases []float32, ids []int32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	out := make([]float32, outDim)
	for route, expertID := range ids {
		expert := int(expertID)
		routeWeight := routeWeights[route]
		for outCol := 0; outCol < outDim; outCol++ {
			var sum float32
			for inCol := 0; inCol < inDim; inCol++ {
				weightIndex := (expert*outDim+outCol)*inDim + inCol
				group := inCol / groupSize
				scaleIndex := (expert*outDim+outCol)*groups + group
				w := float32(quantized[weightIndex])*scales[scaleIndex] + biases[scaleIndex]
				sum += input[route*inDim+inCol] * w
			}
			out[outCol] += routeWeight * sum
		}
	}
	return out
}

func quantizedSwitchLinearExpertIDTest(t *testing.T, experts, outDim, inDim, groupSize, bits, seed int) *SwitchLinear {
	t.Helper()
	if bits != 4 {
		t.Fatalf("test helper currently packs q4 only, got bits=%d", bits)
	}
	quantized := make([]uint8, experts*outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*seed + 5) & 15)
	}
	groups := inDim / groupSize
	scales := make([]float32, experts*outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.025 * float32((i%9)+1)
		biases[i] = -0.45 + 0.05*float32((i+seed)%17)
	}
	return NewQuantizedSwitchLinear(
		FromValues(packMLXAffineQ4TestRows(t, quantized), experts, outDim, inDim/(32/bits)),
		FromValues(scales, experts, outDim, groups),
		FromValues(biases, experts, outDim, groups),
		nil,
		groupSize,
		bits,
	)
}

func quantizedSwitchLinearSidecarsAsType(linear *SwitchLinear, dtype DType) {
	if linear == nil || linear.Scales == nil || linear.Biases == nil {
		return
	}
	scales := AsType(linear.Scales, dtype)
	biases := AsType(linear.Biases, dtype)
	Free(linear.Scales, linear.Biases)
	linear.Scales = scales
	linear.Biases = biases
}
