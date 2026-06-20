// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSDPAHeads8KV4Dim64Len16(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPA(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}
