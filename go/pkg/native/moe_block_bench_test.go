// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoEBlockBF16Top2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
