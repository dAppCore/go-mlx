// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestSquareICBMatchesUnarySquare(t *testing.T) {
	requireNativeRuntime(t)
	in := []float32{1, -2, 3, -4}
	got, err := squareICB(in)
	if err != nil {
		t.Fatalf("squareICB: %v", err)
	}
	for i, v := range in {
		if want := v * v; got[i] != want {
			t.Fatalf("squareICB[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestGemvICBMatchesMatVec(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	got, err := gemvICB(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("gemvICB: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gemvICB[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRebindProbeICBWritesEachReplayRow(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim, nRows = 16, 64, 3
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	got, err := rebindProbeICB(mat, vec, outDim, inDim, nRows)
	if err != nil {
		t.Fatalf("rebindProbeICB: %v", err)
	}
	for row := 0; row < nRows; row++ {
		for i := range want {
			if got[row*outDim+i] != want[i] {
				t.Fatalf("rebind row %d value %d = %v, want %v", row, i, got[row*outDim+i], want[i])
			}
		}
	}
}

func TestQMVICBMatchesQMVBF16(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim, groupSize, bits = 16, 64, 32, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 37)
	x := toBF16Bytes(syntheticFloat32(inDim, 53))
	want, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16: %v", err)
	}
	got, err := qmvICB(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("qmvICB: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("qmvICB byte %d = %#x, want %#x", i, got[i], want[i])
		}
	}
}

func TestRopeFreqsPipelineICBBuildsVariants(t *testing.T) {
	requireNativeRuntime(t)
	for _, traditional := range []bool{false, true} {
		pso, err := ropeFreqsPipelineICB(traditional)
		if err != nil {
			t.Fatalf("ropeFreqsPipelineICB(%v): %v", traditional, err)
		}
		if pso == nil || pso.GetID() == 0 {
			t.Fatalf("ropeFreqsPipelineICB(%v) returned nil pipeline", traditional)
		}
	}
}
