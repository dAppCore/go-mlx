// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkAssembleMistralBF16(b *testing.B) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	_, arch := mistralConfigFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	tensors := mistralTensorFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	b.SetBytes(int64(vocab*dModel*bf16Size + nLayers*dFF*dModel*bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AssembleMistralBF16(tensors, arch); err != nil {
			b.Fatal(err)
		}
	}
}
