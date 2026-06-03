// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestGemma4FFNResidual_NativeMatchesGoGraph_Good(t *testing.T) {
	coverageTokens := "Gemma4FFNResidual NativeMatchesGoGraph"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	residual := FromValues([]float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5, 0.125, -0.875}, 1, 1, 8)
	local := FromValues([]float32{0.5, -0.25, 1.0, 0.125, -0.75, 1.5, -1.25, 0.375}, 1, 1, 8)
	expert := FromValues([]float32{-0.125, 0.875, -1.5, 0.25, 1.25, -0.5, 0.625, -0.75}, 1, 1, 8)
	localNorm := FromValues([]float32{1.0, 0.75, 1.25, 1.5, 0.5, 1.75, 0.875, 1.125}, 8)
	expertNorm := FromValues([]float32{0.875, 1.5, 0.625, 1.25, 1.0, 0.75, 1.375, 0.5}, 8)
	combinedNorm := FromValues([]float32{1.125, 0.625, 1.5, 0.75, 1.25, 0.875, 1.0, 1.375}, 8)
	defer Free(residual, local, expert, localNorm, expertNorm, combinedNorm)

	localNormed := RMSNorm(local, localNorm, 1e-6)
	expertNormed := RMSNorm(expert, expertNorm, 1e-6)
	combined := Add(localNormed, expertNormed)
	combinedResidual := RMSNorm(combined, combinedNorm, 1e-6)
	want := Add(residual, combinedResidual)
	defer Free(localNormed, expertNormed, combined, combinedResidual, want)

	restore := SetRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL", "1")
	got, ok, err := nativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm, 1e-6)
	restore()
	if err != nil {
		t.Fatalf("nativeGemma4FFNResidual() error = %v", err)
	}
	if !ok {
		t.Fatal("nativeGemma4FFNResidual() ok = false, want true")
	}
	defer Free(got)
	Materialize(got, want)

	assertFloat32SliceClose(t, got.Floats(), want.Floats(), 1e-5)
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 8 {
		t.Fatalf("shape = %+v, want [1 1 8]", shape)
	}
}
