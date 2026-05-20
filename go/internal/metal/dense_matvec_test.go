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
	mlp := &MLP{
		GateProj: quantizedLinearDenseMatVecTest(t, mlpDim, hidden, groupSize, bits, 3),
		UpProj:   quantizedLinearDenseMatVecTest(t, mlpDim, hidden, groupSize, bits, 5),
		DownProj: quantizedLinearDenseMatVecTest(t, hidden, mlpDim, groupSize, bits, 11),
	}
	denseMatVecSidecarsAsType(mlp.GateProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.UpProj, DTypeBFloat16)
	denseMatVecSidecarsAsType(mlp.DownProj, DTypeBFloat16)
	defer func() {
		freeLinear(mlp.GateProj)
		freeLinear(mlp.UpProj)
		freeLinear(mlp.DownProj)
	}()

	x := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, hidden)
	defer Free(x)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "0")
	want := mlp.forward(x)
	restoreOff()
	defer Free(want)

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
	Materialize(want, got)

	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 1e-3)
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
	linear := quantizedLinearDenseMatVecTest(t, outDim, inDim, groupSize, bits, 7)
	denseMatVecSidecarsAsType(linear, DTypeBFloat16)
	defer freeLinear(linear)

	x := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, inDim)
	defer Free(x)

	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "0")
	want := linear.Forward(x)
	restoreOff()
	defer Free(want)

	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "1")
	got := linear.Forward(x)
	restoreOn()
	defer Free(got)
	Materialize(want, got)

	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 5e-4)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != outDim {
		t.Fatalf("shape = %+v, want [1 1 %d]", shape, outDim)
	}
}

func quantizedLinearDenseMatVecTest(t *testing.T, outDim, inDim, groupSize, bits, seed int) *Linear {
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
	return NewQuantizedLinear(
		FromValues(packMLXAffineQ4TestRows(t, quantized), outDim, inDim/(32/bits)),
		FromValues(scales, outDim, groups),
		FromValues(biases, outDim, groups),
		nil,
		groupSize,
		bits,
	)
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
