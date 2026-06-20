// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkBinaryAdd1024(b *testing.B) {
	requireNativeRuntime(b)

	a := syntheticFloat32(1024, 3)
	c := syntheticFloat32(1024, 5)
	b.SetBytes(int64(len(a) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Add(a, c); err != nil {
			b.Fatal(err)
		}
	}
}
