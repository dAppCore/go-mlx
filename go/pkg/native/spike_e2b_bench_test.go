// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// THROWAWAY spike instrument. e2b-scale ICB quant decode (dModel=1536, gs64/b4, 6 layers, 64 tokens so
// the one-off ICB build amortises and the per-token REPLAY dominates). Real dims (input-rms fusion
// engages). Synthetic weights — the perf delta is the dispatch/barrier structure, not the values.
// Measures pkg/native (NOT lthn-mlx serve, which is the pkg/metal cgo engine).
func spikeE2BFixture(b *testing.B) (inputs [][]byte, layers []QuantizedLayerWeights, arch model.Arch) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 1536, 8, 1, 256, 6144, 32, 6
	const groupSize, bits = 64, 4
	const nTokens = 64
	arch = archFixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs = decodeInputsFixture(nTokens, dModel)
	layers = make([]QuantizedLayerWeights, nLayers)
	for li := range layers {
		layers[li] = quantizedLayerFixture(b, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, li+3)
	}
	b.SetBytes(int64(nTokens))
	return inputs, layers, arch
}

func spikeE2BDecode(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpikeE2BDecode — current barrier structure (input-rms fused per the recorder gate).
func BenchmarkSpikeE2BDecode(b *testing.B) { spikeE2BDecode(b) }

// BenchmarkSpikeE2BDecodeNoBarrier — ALL barriers off: the absolute no-barrier ceiling (the "311"
// floor; output is racy garbage, timing only). The gap to BenchmarkSpikeE2BDecode is the TOTAL barrier
// cost — how much is actually on the table, and whether element-wise (~4%) or the matmul tier owns it.
func BenchmarkSpikeE2BDecodeNoBarrier(b *testing.B) {
	allBarriersOffForTest = true
	defer func() { allBarriersOffForTest = false }()
	spikeE2BDecode(b)
}

// BenchmarkSpikeE2BDecodeReencode — the RE-ENCODE path (regular Metal encoder, fine-grained hazard
// tracking) instead of the ICB (COARSE wait-all-prior barriers). If this beats BenchmarkSpikeE2BDecode,
// the COARSE barrier is the cost and the lever is finer sync; if it's slower/equal, the barrier cost is
// inherent and only fusion removes it. This decides which matmul-tier fix to build.
func BenchmarkSpikeE2BDecodeReencode(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}
