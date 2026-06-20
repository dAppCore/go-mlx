// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNativeTokenModelEmbed(b *testing.B) {
	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	tm, err := NewBF16TokenModel(g, arch, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(arch.Hidden * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tm.Embed(int32(i % arch.Vocab)); err != nil {
			b.Fatal(err)
		}
	}
}
