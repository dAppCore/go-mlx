// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

func TestDenseMatVec_NativeMLPMatchesGoGraph_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeMLPMatchesGoGraph"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		hidden    = 8
		mlpDim    = 8
		groupSize = 4
		bits      = 4
	)
	inputValues := []float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}
	gate := quantizedLinearDenseMatVecFixture(t, mlpDim, hidden, groupSize, bits, 3)
	up := quantizedLinearDenseMatVecFixture(t, mlpDim, hidden, groupSize, bits, 5)
	down := quantizedLinearDenseMatVecFixture(t, hidden, mlpDim, groupSize, bits, 11)
	mlp := &MLP{
		GateProj: gate.linear,
		UpProj:   up.linear,
		DownProj: down.linear,
	}
	denseMatVecSidecarsAsType(mlp.GateProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.UpProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.DownProj, DTypeBFloat16)
	defer func() {
		freeLinear(mlp.GateProj)
		freeLinear(mlp.UpProj)
		freeLinear(mlp.DownProj)
	}()

	x := FromValues(inputValues, 1, 1, hidden)
	defer Free(x)

	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "1")
	got, ok, err := nativeMLPMatVec(x, mlp)
	restoreOn()
	if err != nil {
		t.Fatalf("nativeMLPMatVec() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMLPMatVec() ok = false, want true")
	}
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(nativeMLPMatVec) error = %v", err)
	}

	gateRef := quantizedDenseMatVecCPUReference(inputValues, gate.quantized, gate.scales, gate.biases, mlpDim, hidden, groupSize)
	upRef := quantizedDenseMatVecCPUReference(inputValues, up.quantized, up.scales, up.biases, mlpDim, hidden, groupSize)
	activated := make([]float32, mlpDim)
	for i := range activated {
		activated[i] = geluApproxFloat32(gateRef[i]) * upRef[i]
	}
	want := quantizedDenseMatVecCPUReference(activated, down.quantized, down.scales, down.biases, hidden, mlpDim, groupSize)

	assertFloat32SliceClose(t, got.Floats(), want, 2e-1)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != hidden {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, hidden)
	}
}

func TestDenseMatVec_NativeLinearForwardMatchesQuantizedMatmul_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeLinearForwardMatchesQuantizedMatmul"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		inDim     = 8
		outDim    = 6
		groupSize = 4
		bits      = 4
	)
	inputValues := []float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}
	fixture := quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, 7)
	linear := fixture.linear
	denseMatVecSidecarsAsType(linear, DTypeBFloat16)
	defer freeLinear(linear)

	x := FromValues(inputValues, 1, 1, inDim)
	defer Free(x)

	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "1")
	got := linear.Forward(x)
	restoreOn()
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(native linear matvec) error = %v", err)
	}

	want := quantizedDenseMatVecCPUReference(inputValues, fixture.quantized, fixture.scales, fixture.biases, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, got.Floats(), want, 1e-2)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func TestDenseMatVec_NativeLinearForwardSupportsQ6Default_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeLinearForwardSupportsQ6Default"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		inDim     = 10
		outDim    = 4
		groupSize = 5
		bits      = 6
	)
	inputValues := []float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875, 1.75, -0.25}
	fixture := quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, 13)
	linear := fixture.linear
	denseMatVecSidecarsAsType(linear, DTypeBFloat16)
	defer freeLinear(linear)

	x := FromValues(inputValues, 1, 1, inDim)
	defer Free(x)

	got, ok, err := quantizedDenseMatVec(x, linear)
	if err != nil {
		t.Fatalf("quantizedDenseMatVec(q6) error = %v", err)
	}
	if !ok {
		t.Fatal("quantizedDenseMatVec(q6) ok = false, want native q6 path for product default")
	}
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(native q6 linear matvec) error = %v", err)
	}

	want := quantizedDenseMatVecCPUReference(inputValues, fixture.quantized, fixture.scales, fixture.biases, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, got.Floats(), want, 2e-1)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func TestDenseMatVec_NativeLinearForwardSupportsQ6E2BShape_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeLinearForwardSupportsQ6E2BShape"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		inDim     = 1536
		outDim    = 4
		groupSize = 64
		bits      = 6
	)
	inputValues := make([]float32, inDim)
	for i := range inputValues {
		inputValues[i] = -1.25 + 0.03125*float32((i*11)%89)
	}
	fixture := quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, 17)
	linear := fixture.linear
	denseMatVecSidecarsAsType(linear, DTypeBFloat16)
	defer freeLinear(linear)

	x := FromValues(inputValues, 1, 1, inDim)
	defer Free(x)

	restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "1")
	got, ok, err := quantizedDenseMatVec(x, linear)
	restoreQ6()
	if err != nil {
		t.Fatalf("quantizedDenseMatVec(q6 E2B shape) error = %v", err)
	}
	if !ok {
		t.Fatal("quantizedDenseMatVec(q6 E2B shape) ok = false, want native q6 bitstream path")
	}
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(native q6 E2B linear matvec) error = %v", err)
	}

	want := quantizedDenseMatVecCPUReference(inputValues, fixture.quantized, fixture.scales, fixture.biases, outDim, inDim, groupSize)
	assertFloat32SliceClose(t, got.Floats(), want, 1.0)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func TestDenseMatVec_NativeLinearQ6E2BShapeDefaultFallsBack_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeLinearQ6E2BShapeDefaultFallsBack"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		inDim     = 1536
		outDim    = 1
		groupSize = 64
		bits      = 6
	)
	fixture := quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, 37)
	linear := fixture.linear
	denseMatVecSidecarsAsType(linear, DTypeBFloat16)
	defer freeLinear(linear)

	x := FromValues(make([]float32, inDim), 1, 1, inDim)
	defer Free(x)

	restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "0")
	got, ok, err := quantizedDenseMatVec(x, linear)
	restoreQ6()
	Free(got)
	if err != nil {
		t.Fatalf("quantizedDenseMatVec(q6 E2B default) error = %v", err)
	}
	if ok {
		t.Fatal("quantizedDenseMatVec(q6 E2B default) ok = true, want fallback until native q6 bitstream is faster")
	}
}

func TestDenseMatVec_NativeMLPSupportsQ6E2BShape_Good(t *testing.T) {
	coverageTokens := "DenseMatVec NativeMLPSupportsQ6E2BShape"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const (
		hidden    = 1536
		mlpDim    = 64
		outDim    = 8
		groupSize = 64
		bits      = 6
	)
	inputValues := make([]float32, hidden)
	for i := range inputValues {
		inputValues[i] = -0.03125 + 0.001953125*float32((i*13)%31)
	}
	gate := quantizedLinearDenseMatVecFixture(t, mlpDim, hidden, groupSize, bits, 19)
	up := quantizedLinearDenseMatVecFixture(t, mlpDim, hidden, groupSize, bits, 23)
	down := quantizedLinearDenseMatVecFixture(t, outDim, mlpDim, groupSize, bits, 29)
	mlp := &MLP{
		GateProj: gate.linear,
		UpProj:   up.linear,
		DownProj: down.linear,
	}
	denseMatVecSidecarsAsType(mlp.GateProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.UpProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.DownProj, DTypeBFloat16)
	defer func() {
		freeLinear(mlp.GateProj)
		freeLinear(mlp.UpProj)
		freeLinear(mlp.DownProj)
	}()

	x := FromValues(inputValues, 1, 1, hidden)
	defer Free(x)

	restoreQ6 := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC", "1")
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "1")
	got, ok, err := nativeMLPMatVec(x, mlp)
	restoreOn()
	restoreQ6()
	if err != nil {
		t.Fatalf("nativeMLPMatVec(q6 E2B shape) error = %v", err)
	}
	if !ok {
		t.Fatal("nativeMLPMatVec(q6 E2B shape) ok = false, want native q6 bitstream path")
	}
	defer Free(got)
	if err := Eval(got); err != nil {
		t.Fatalf("Eval(native q6 E2B MLP matvec) error = %v", err)
	}
	gotValues := got.Floats()
	for i, value := range gotValues {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("got[%d] = %v, want finite", i, value)
		}
	}
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

type denseMatVecLinearFixture struct {
	linear    *Linear
	quantized []uint8
	scales    []float32
	biases    []float32
}

func quantizedLinearDenseMatVecTest(t testing.TB, outDim, inDim, groupSize, bits, seed int) *Linear {
	return quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, seed).linear
}

func quantizedLinearDenseMatVecFixture(t testing.TB, outDim, inDim, groupSize, bits, seed int) denseMatVecLinearFixture {
	t.Helper()
	if bits <= 0 || bits > 16 || inDim <= 0 || outDim <= 0 || groupSize <= 0 || inDim%groupSize != 0 {
		t.Fatalf("test helper cannot pack out=%d in=%d group=%d bits=%d", outDim, inDim, groupSize, bits)
	}
	quantized := make([]uint8, outDim*inDim)
	maxValue := uint8((1 << bits) - 1)
	for i := range quantized {
		quantized[i] = uint8((i*seed + 5) & int(maxValue))
	}
	groups := inDim / groupSize
	scales := make([]float32, outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.025 * float32((i%9)+1)
		biases[i] = -0.45 + 0.05*float32((i+seed)%17)
	}
	packed := packMLXAffineTestRows(t, quantized, bits, inDim)
	if len(packed)%outDim != 0 {
		t.Fatalf("packed q%d rows length %d must divide outDim %d", bits, len(packed), outDim)
	}
	packedIn := len(packed) / outDim
	return denseMatVecLinearFixture{
		linear: NewQuantizedLinear(
			FromValues(packed, outDim, packedIn),
			FromValues(scales, outDim, groups),
			FromValues(biases, outDim, groups),
			nil,
			groupSize,
			bits,
		),
		quantized: quantized,
		scales:    scales,
		biases:    biases,
	}
}

func packMLXAffineTestRows(t testing.TB, values []uint8, bits, rowValues int) []uint32 {
	t.Helper()
	if bits <= 0 || bits > 16 || rowValues <= 0 || len(values)%rowValues != 0 {
		t.Fatalf("q%d test rows must use 1..16 bits and complete rows, row=%d values=%d", bits, rowValues, len(values))
	}
	maxValue := uint8((1 << bits) - 1)
	rows := len(values) / rowValues
	packedIn := quantizedDenseMatVecPackedIn(rowValues, bits)
	packed := make([]uint32, rows*packedIn)
	for row := 0; row < rows; row++ {
		for col := 0; col < rowValues; col++ {
			value := values[row*rowValues+col]
			if value > maxValue {
				t.Fatalf("q%d value %d exceeds %d", bits, value, maxValue)
			}
			bitOffset := col * bits
			packIndex := row*packedIn + bitOffset/32
			shift := uint(bitOffset % 32)
			packed[packIndex] |= uint32(value) << shift
			if spill := shift + uint(bits); spill > 32 {
				packed[packIndex+1] |= uint32(value) >> (32 - shift)
			}
		}
	}
	return packed
}

func quantizedDenseMatVecCPUReference(input []float32, quantized []uint8, scales, biases []float32, outDim, inDim, groupSize int) []float32 {
	groups := inDim / groupSize
	out := make([]float32, outDim)
	for outCol := 0; outCol < outDim; outCol++ {
		var sum float32
		for inCol := 0; inCol < inDim; inCol++ {
			weightIndex := outCol*inDim + inCol
			group := inCol / groupSize
			scaleIndex := outCol*groups + group
			w := float32(quantized[weightIndex])*scales[scaleIndex] + biases[scaleIndex]
			sum += input[inCol] * w
		}
		out[outCol] = sum
	}
	return out
}

func denseMatVecSidecarsAsType(linear *Linear, dtype DType) {
	if linear == nil || linear.Scales == nil || linear.Biases == nil {
		return
	}
	scales := AsType(linear.Scales, dtype)
	biases := AsType(linear.Biases, dtype)
	Free(linear.Scales, linear.Biases)
	linear.Scales = scales
	linear.Biases = biases
}
