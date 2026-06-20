// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoEExpertsTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	b.SetBytes(int64(len(x) + len(gateW) + len(upW) + len(downW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
			b.Fatal(err)
		}
	}
}
