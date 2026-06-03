// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestGemma4RouterMatVecNativeMatchesQuantizedLinear_Good(t *testing.T) {
	coverageTokens := "Gemma4RouterMatVecNative MatchesQuantizedLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC", "1"))

	const (
		outDim    = 5
		inDim     = 16
		groupSize = 4
		bits      = 8
	)
	quantized := make([]uint8, outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*13 + 7) & 255)
	}
	groups := inDim / groupSize
	scales := make([]float32, outDim*groups)
	qbiases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.00390625 * float32((i%7)+1)
		qbiases[i] = -0.75 + 0.0625*float32(i%11)
	}
	inputValues := make([]float32, inDim)
	for i := range inputValues {
		inputValues[i] = -1.0 + 0.125*float32((i*5)%19)
	}

	input := FromValues(inputValues, 1, 1, inDim)
	weight := FromValues(packMLXAffineQ8TestRows(t, quantized), outDim, inDim/(32/bits))
	scaleRaw := FromValues(scales, outDim, groups)
	biasRaw := FromValues(qbiases, outDim, groups)
	scaleArray := AsType(scaleRaw, DTypeBFloat16)
	biasArray := AsType(biasRaw, DTypeBFloat16)
	Free(scaleRaw, biasRaw)
	defer Free(input, weight, scaleArray, biasArray)
	linear := NewQuantizedLinear(weight, scaleArray, biasArray, nil, groupSize, bits)

	want := linear.Forward(input)
	got, ok, err := NativeGemma4RouterMatVecScores(input, linear)
	if err != nil {
		t.Fatalf("NativeGemma4RouterMatVecScores() error = %v", err)
	}
	if !ok {
		t.Fatal("NativeGemma4RouterMatVecScores() ok = false, want true")
	}
	defer Free(want, got)
	Materialize(want, got)

	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 5e-3)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func TestMoERouterMatVecNativeMatchesQuantizedLinear_Good(t *testing.T) {
	coverageTokens := "MoERouterMatVecNative MatchesQuantizedLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		outDim    = 5
		inDim     = 16
		groupSize = 4
		bits      = 8
	)
	quantized := make([]uint8, outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*13 + 7) & 255)
	}
	groups := inDim / groupSize
	scales := make([]float32, outDim*groups)
	qbiases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.00390625 * float32((i%7)+1)
		qbiases[i] = -0.75 + 0.0625*float32(i%11)
	}
	inputValues := make([]float32, inDim)
	for i := range inputValues {
		inputValues[i] = -1.0 + 0.125*float32((i*5)%19)
	}

	input := FromValues(inputValues, 1, 1, inDim)
	weight := FromValues(packMLXAffineQ8TestRows(t, quantized), outDim, inDim/(32/bits))
	scaleRaw := FromValues(scales, outDim, groups)
	biasRaw := FromValues(qbiases, outDim, groups)
	scaleArray := AsType(scaleRaw, DTypeBFloat16)
	biasArray := AsType(biasRaw, DTypeBFloat16)
	Free(scaleRaw, biasRaw)
	defer Free(input, weight, scaleArray, biasArray)
	router := MoERouterProjection{
		Weight:    weight,
		Scales:    scaleArray,
		Biases:    biasArray,
		GroupSize: groupSize,
		Bits:      bits,
	}
	linear := router.Linear()

	want := linear.Forward(input)
	got, ok, err := nativeMoERouterProjectionScores(input, router)
	if err != nil {
		t.Fatalf("nativeMoERouterProjectionScores() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMoERouterProjectionScores() ok = false, want true")
	}
	defer Free(want, got)
	Materialize(want, got)

	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 5e-3)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func TestGemma4RouterTopKNative_Good(t *testing.T) {
	coverageTokens := "Gemma4RouterTopKNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK", "1"))

	scores := FromValues([]float32{1, 4, 2, -1}, 1, 1, 4)
	scale := FromValues([]float32{1, 2, 1, 3}, 4)
	defer Free(scores, scale)

	indices, weights, ok, err := NativeGemma4RouterTopK(scores, scale, 2)
	if err != nil {
		t.Fatalf("NativeGemma4RouterTopK() error = %v", err)
	}
	if !ok {
		t.Fatal("NativeGemma4RouterTopK() ok = false, want true")
	}
	defer Free(indices, weights)
	if err := Eval(indices, weights); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	gotIndices := indices.DataInt32()
	wantIndices := []int32{1, 2}
	for i := range wantIndices {
		if gotIndices[i] != wantIndices[i] {
			t.Fatalf("indices[%d] = %d, want %d", i, gotIndices[i], wantIndices[i])
		}
	}
	floatSliceApprox(t, weights.Floats(), []float32{1.7615942, 0.11920292})
}

func TestMoERouterTopKNative_Good(t *testing.T) {
	coverageTokens := "MoERouterTopKNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	scores := FromValues([]float32{1, 4, 2, -1}, 1, 1, 4)
	scale := FromValues([]float32{1, 2, 1, 3}, 4)
	defer Free(scores, scale)

	indices, weights, ok, err := nativeMoERouterTopK(scores, scale, 2)
	if err != nil {
		t.Fatalf("nativeMoERouterTopK() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMoERouterTopK() ok = false, want true")
	}
	defer Free(indices, weights)
	if err := Eval(indices, weights); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	gotIndices := indices.DataInt32()
	wantIndices := []int32{1, 2}
	for i := range wantIndices {
		if gotIndices[i] != wantIndices[i] {
			t.Fatalf("indices[%d] = %d, want %d", i, gotIndices[i], wantIndices[i])
		}
	}
	floatSliceApprox(t, weights.Floats(), []float32{1.7615942, 0.11920292})
}

func TestMoERouterTopKUnitScaleNative_Good(t *testing.T) {
	coverageTokens := "MoERouterTopKUnitScaleNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	scores := FromValues([]float32{1, 4, 2, -1}, 1, 1, 4)
	defer Free(scores)

	indices, weights, ok, err := nativeMoERouterTopK(scores, nil, 2)
	if err != nil {
		t.Fatalf("nativeMoERouterTopK(unit scale) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMoERouterTopK(unit scale) ok = false, want true")
	}
	defer Free(indices, weights)
	if err := Eval(indices, weights); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	gotIndices := indices.DataInt32()
	wantIndices := []int32{1, 2}
	for i := range wantIndices {
		if gotIndices[i] != wantIndices[i] {
			t.Fatalf("indices[%d] = %d, want %d", i, gotIndices[i], wantIndices[i])
		}
	}
	floatSliceApprox(t, weights.Floats(), []float32{0.8807971, 0.11920292})
}

func TestMoERouterTopKUnitScaleKernelCache_Good(t *testing.T) {
	coverageTokens := "MoERouterTopKUnitScale KernelCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	first := nativeMoERouterTopKUnitScaleKernel(8, 2)
	second := nativeMoERouterTopKUnitScaleKernel(8, 2)
	if first == nil || second == nil {
		t.Fatal("nativeMoERouterTopKUnitScaleKernel returned nil")
	}
	if first != second {
		t.Fatal("nativeMoERouterTopKUnitScaleKernel did not reuse cached kernel")
	}
}

func packMLXAffineQ8TestRows(t *testing.T, values []uint8) []uint32 {
	t.Helper()
	if len(values)%4 != 0 {
		t.Fatalf("q8 test rows must have a multiple of 4 values, got %d", len(values))
	}
	packed := make([]uint32, len(values)/4)
	for i, value := range values {
		packed[i/4] |= uint32(value) << uint((i%4)*8)
	}
	return packed
}
