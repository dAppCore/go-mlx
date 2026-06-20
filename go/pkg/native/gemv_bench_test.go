// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMatVec128x256(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 128, 256
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	b.SetBytes(int64((len(mat) + len(vec)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatVec(mat, vec, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}
