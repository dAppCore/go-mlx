// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/safetensors"
)

func BenchmarkLoadMistralBF16(b *testing.B) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	cfg, _ := mistralConfigFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	configJSON := core.JSONMarshal(cfg)
	if !configJSON.OK {
		b.Fatalf("marshal config: %s", configJSON.Error())
	}
	blob, err := safetensors.Encode(mistralTensorFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(configJSON.Value.([]byte)) + len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := LoadMistralBF16(configJSON.Value.([]byte), blob); err != nil {
			b.Fatal(err)
		}
	}
}
