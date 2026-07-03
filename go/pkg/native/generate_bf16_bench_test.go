// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkGenerateGemma4BF16OneToken(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	prompt := []int32{1, 5}
	b.SetBytes(int64(len(prompt) * arch.Hidden * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GenerateGemma4BF16(g, arch, prompt, 1, 4, -1); err != nil {
			b.Fatal(err)
		}
	}
}
