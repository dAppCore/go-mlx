// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoERouterTop2Of8(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel = 8, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	b.SetBytes(int64(len(x) + len(normW) + len(routerW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := MoERouter(x, normW, routerW, scale, numExperts, topK, dModel, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
