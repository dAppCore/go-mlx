// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNewGemma4Session(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	b.SetBytes(int64(len(g.Embed) + len(g.Layers[0].WGate)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewGemma4Session(g, arch, 4)
		if err != nil {
			b.Fatal(err)
		}
		_ = sess.Close()
	}
}
