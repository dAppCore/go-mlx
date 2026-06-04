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

	gotArray, err := QuantizedExpertIDMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("QuantizedExpertIDMatVec() error = %v", err)
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

	gotArray, err := QuantizedExpertIDMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("QuantizedExpertIDMatVec() error = %v", err)
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

	gotArray, err := QuantizedExpertIDGELUGateUpMatVec(input, weight, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("QuantizedExpertIDGELUGateUpMatVec() error = %v", err)
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

	gotArray, err := QuantizedExpertIDWeightedMatVecSum(input, routeWeightArray, weightArray, scaleArray, biasArray, idArray, groupSize, bits)
	if err != nil {
		t.Fatalf("QuantizedExpertIDWeightedMatVecSum() error = %v", err)
	}
	defer Free(gotArray)
	Materialize(gotArray)

	want := quantizedExpertIDWeightedMatVecSumCPUReference(inputValues, routeWeights, quantized, scales, qbiases, ids, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, gotArray.Floats(), want, 3e-4)
	if shape := gotArray.Shape(); len(shape) != 1 || shape[0] != outDim {
		t.Fatalf("shape = %+v, want [%d]", shape, outDim)
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

	_, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 4, 4)
	if err == nil || !core.Contains(err.Error(), "input row count") {
		t.Fatalf("error = %v, want input row count diagnostic", err)
	}

	validIDs := FromValues([]int32{0}, 1)
	defer Free(validIDs)
	_, err = QuantizedExpertIDMatVec(input, weight, scales, biases, validIDs, 4, 3)
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

	_, err := QuantizedExpertIDMatVec(input, weight, scales, biases, ids, 4, 4)
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
		for outCol := range outDim {
			var sum float32
			for inCol := range inDim {
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
		for outCol := range halfOut {
			var gateSum, upSum float32
			for inCol := range inDim {
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
		for outCol := range outDim {
			var sum float32
			for inCol := range inDim {
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
