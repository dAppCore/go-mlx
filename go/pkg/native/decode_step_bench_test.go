// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeStepKV64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim := nHeads*headDim, nKV*headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	_ = qDim
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kc := append([]byte(nil), kCache...)
		vc := append([]byte(nil), vCache...)
		if _, err := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}
