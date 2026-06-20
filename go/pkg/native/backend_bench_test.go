// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNativeBackendBF16DecodeForward(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		b.Fatal(err)
	}
	inputs := decodeInputsFixture(2, dModel)
	b.SetBytes(int64(len(inputs) * dModel * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.DecodeForward(inputs); err != nil {
			b.Fatal(err)
		}
	}
}
