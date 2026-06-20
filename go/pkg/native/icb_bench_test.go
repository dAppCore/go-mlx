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
