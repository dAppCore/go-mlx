// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkGeluGateMulBF161024(b *testing.B) {
	requireNativeRuntime(b)

	gate := toBF16Bytes(syntheticFloat32(1024, 3))
	up := toBF16Bytes(syntheticFloat32(1024, 5))
	b.SetBytes(int64(len(gate) + len(up)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GeluGateMulBF16(gate, up); err != nil {
			b.Fatal(err)
		}
	}
}
