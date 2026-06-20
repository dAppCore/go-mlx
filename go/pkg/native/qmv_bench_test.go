// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkQMV64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := syntheticFloat32(inDim, 5)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QMV(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}
