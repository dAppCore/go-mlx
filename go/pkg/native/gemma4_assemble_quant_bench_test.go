// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

func BenchmarkAssembleGemma4Quant(b *testing.B) {
	const groupSize, bits, nLayers = 32, 4, 2
	arch := archFixture(b, 64, 2, 1, 16, 128, 32, nLayers)
	tensors := quantTensors(arch, groupSize, bits)
	fill := func(n int, v byte) []byte {
		out := make([]byte, n)
		for i := range out {
			out[i] = v
		}
		return out
	}
	dModel, vocab := arch.Hidden, arch.Vocab
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{vocab, dModel * bits / 32}, Data: fill(vocab*dModel*bits/8, 41)}
	tensors["model.embed_tokens.scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, dModel / groupSize}, Data: fill(vocab*(dModel/groupSize)*bf16Size, 43)}
	tensors["model.embed_tokens.biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, dModel / groupSize}, Data: fill(vocab*(dModel/groupSize)*bf16Size, 47)}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dModel}, Data: fill(dModel*bf16Size, 53)}
	quant := &g4.QuantConfig{GroupSize: groupSize, Bits: bits}
	var bytes int
	for _, t := range tensors {
		bytes += len(t.Data)
	}
	b.SetBytes(int64(bytes))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AssembleGemma4Quant(tensors, arch, quant); err != nil {
			b.Fatal(err)
		}
	}
}
