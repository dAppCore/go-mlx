// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRMSNormRows4Axis1024(b *testing.B) {
	requireNativeRuntime(b)

	const rows, axis = 4, 1024
	x := syntheticFloat32(rows*axis, 3)
	w := syntheticFloat32(axis, 5)
	b.SetBytes(int64(len(x) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNorm(x, w, rows, axis, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}
