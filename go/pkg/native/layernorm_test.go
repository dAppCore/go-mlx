// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestLayerNormBF16 asserts native.LayerNormBF16 is BYTE-IDENTICAL to pkg/metal.LayerNorm over the
// last axis (parity_test.go pattern, eqBytes — not a tolerance). The gemma4 audio subsampler's
// scale-only LayerNorm (after each strided conv) goes through this.
func TestLayerNormBF16(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 20, 64
	eps := float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(rows*ax, 3))
	w := toBF16Bytes(syntheticFloat32(ax, 5))
	b := toBF16Bytes(syntheticFloat32(ax, 7))

	got, err := LayerNormBF16(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16: %v", err)
	}
	r := mc.AsType(mc.LayerNorm(marr(x, rows, ax), marr(w, ax), marr(b, ax), eps), mc.DTypeBFloat16)
	mc.Materialize(r)
	eqBytes(t, "LayerNormBF16 vs metal.LayerNorm", got, append([]byte(nil), r.RawBytes()...))
}

func TestLayerNormF32(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 7, 16
	eps := float32(1e-5)
	x := syntheticFloat32(rows*ax, 3)
	w := syntheticFloat32(ax, 5)
	b := syntheticFloat32(ax, 7)

	got, err := LayerNormF32(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormF32: %v", err)
	}
	r := mc.LayerNorm(mc.FromValues(x, rows, ax), mc.FromValues(w, ax), mc.FromValues(b, ax), eps)
	eqF32(t, "LayerNormF32 vs metal.LayerNorm", got, r)
}

func TestLayerNormF32LoopedAxis(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 2, 7000
	eps := float32(1e-5)
	x := syntheticFloat32(rows*ax, 23)
	w := syntheticFloat32(ax, 29)
	b := syntheticFloat32(ax, 31)

	got, err := LayerNormF32(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormF32 looped axis: %v", err)
	}
	want := hostLayerNormF32(x, w, b, rows, ax, eps)
	assertFloat32Near(t, "LayerNormF32 looped axis", got, want, 2e-4)
}

func TestLayerNormBF16LoopedAxis(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 1, 7000
	eps := float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(rows*ax, 37))
	w := toBF16Bytes(syntheticFloat32(ax, 41))
	b := toBF16Bytes(syntheticFloat32(ax, 43))

	got, err := LayerNormBF16(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16 looped axis: %v", err)
	}
	want := bf16Floats(toBF16Bytes(hostLayerNormF32(bf16Floats(x), bf16Floats(w), bf16Floats(b), rows, ax, eps)))
	assertFloat32Near(t, "LayerNormBF16 looped axis", bf16Floats(got), want, 0.035)
}

func hostLayerNormF32(x, weight, bias []float32, rows, axisSize int, eps float32) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*axisSize : (r+1)*axisSize]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(axisSize)
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(axisSize)
		invStd := float32(1 / math.Sqrt(variance+float64(eps)))
		dst := out[r*axisSize : (r+1)*axisSize]
		for i, v := range row {
			dst[i] = (v-float32(mean))*invStd*weight[i] + bias[i]
		}
	}
	return out
}
