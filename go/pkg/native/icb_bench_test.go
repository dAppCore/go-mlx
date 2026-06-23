// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNormProjectICB128x256(b *testing.B) {
	requireNativeRuntime(b)

	const dIn, dOut = 128, 256
	x := syntheticFloat32(dIn, 3)
	normW := syntheticFloat32(dIn, 5)
	projW := syntheticFloat32(dOut*dIn, 7)
	b.SetBytes(int64((len(x) + len(normW) + len(projW)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBlockICB64(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}
