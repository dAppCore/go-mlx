// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRoPEHeads8Dim64(b *testing.B) {
	requireNativeRuntime(b)

	x := syntheticFloat32(8*64, 3)
	b.SetBytes(int64(len(x) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RoPE(x, 1, 8, 64, 10000, 1, 17, false); err != nil {
			b.Fatal(err)
		}
	}
}
