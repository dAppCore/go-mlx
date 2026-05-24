// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

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

type denseMatVecLinearFixture struct {
	linear    *Linear
	quantized []uint8
	scales    []float32
	biases    []float32
}

func quantizedLinearDenseMatVecTest(t *testing.T, outDim, inDim, groupSize, bits, seed int) *Linear {
	return quantizedLinearDenseMatVecFixture(t, outDim, inDim, groupSize, bits, seed).linear
}

func quantizedLinearDenseMatVecFixture(t *testing.T, outDim, inDim, groupSize, bits, seed int) denseMatVecLinearFixture {
	t.Helper()
	if bits != 4 {
		t.Fatalf("test helper currently packs q4 only, got bits=%d", bits)
	}
	quantized := make([]uint8, outDim*inDim)
	for i := range quantized {
		quantized[i] = uint8((i*seed + 5) & 15)
	}
	groups := inDim / groupSize
	scales := make([]float32, outDim*groups)
	biases := make([]float32, len(scales))
	for i := range scales {
		scales[i] = 0.025 * float32((i%9)+1)
		biases[i] = -0.45 + 0.05*float32((i+seed)%17)
	}
	return denseMatVecLinearFixture{
		linear: NewQuantizedLinear(
			FromValues(packMLXAffineQ4TestRows(t, quantized), outDim, inDim/(32/bits)),
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
