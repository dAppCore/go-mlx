// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkCopyViewSmallWeight(b *testing.B) {
	requireNativeRuntime(b)

	weight := toBF16Bytes(syntheticFloat32(64, 3))
	b.SetBytes(int64(len(weight)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = copyView(weight)
	}
}
