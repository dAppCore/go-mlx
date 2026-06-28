// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkLayerNormBF16Rows4Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axisSize = 4, 512
	const eps = float32(1e-5)
	x, w, bias := layerNormBF16Fixture(rows, axisSize)
	b.SetBytes(int64(len(x) + len(w) + len(bias)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LayerNormBF16(x, w, bias, rows, axisSize, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLayerNormF32Rows4Axis512(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axisSize = 4, 512
	const eps = float32(1e-5)
	x := syntheticFloat32(rows*axisSize, 3)
	w := syntheticFloat32(axisSize, 5)
	bias := syntheticFloat32(axisSize, 7)
	b.SetBytes(int64((len(x) + len(w) + len(bias)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LayerNormF32(x, w, bias, rows, axisSize, eps); err != nil {
			b.Fatal(err)
		}
	}
}
