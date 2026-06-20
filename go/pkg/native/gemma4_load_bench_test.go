// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

func BenchmarkLoadGemma4BF16(b *testing.B) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		b.Fatal(err)
	}
	configJSON := core.JSONMarshal(cfg)
	if !configJSON.OK {
		b.Fatalf("marshal config: %s", configJSON.Error())
	}
	blob, err := safetensors.Encode(gemma4TensorFixture(arch, false))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(configJSON.Value.([]byte)) + len(blob)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := LoadGemma4BF16(configJSON.Value.([]byte), blob); err != nil {
			b.Fatal(err)
		}
	}
}
