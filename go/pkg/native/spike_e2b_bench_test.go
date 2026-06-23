// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"
	"unsafe"

	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/metal"
)

// decodeArchICBQuantPipelined records the stack into TWO ICBs (sharing the KV caches) and replays the
// sequence double-buffered: token t's host prep+submit on icb[t%2] overlaps token t-1's GPU compute on
// icb[(t-1)%2], reclaiming the per-token WaitUntilCompleted idle (pkg/metal's PipelinedDecode, no-cgo).
// The shared cache hazard correctly serialises the GPU side (t's attention waits t-1's KV write). No PLE.
func decodeArchICBQuantPipelined(tb testing.TB, inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int, base, scale, eps float32, valueNorm bool) [][]byte {
	nLayers := len(qlayers)
	kCaches := make([]metal.MTLBuffer, nLayers)
	vCaches := make([]metal.MTLBuffer, nLayers)
	for li := range specs {
		if specs[li].OwnsCache() {
			cb := uint(maxLen * nKVHeads * headDimOf(specs[li], headDim) * bf16Size)
			kCaches[li] = device.NewBufferWithLengthOptions(cb, metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(cb, metal.MTLResourceStorageModeShared)
		}
	}
	var rs [2]*archICBReplay
	for i := range rs {
		r, err := recordArchICBQuant(qlayers, specs, kCaches, vCaches, nil, 0, 0, 0, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, simpleICBRope(base, headDim), scale, eps, valueNorm)
		if err != nil {
			tb.Fatalf("record %d: %v", i, err)
		}
		rs[i] = r
	}

	outputs := make([][]byte, len(inputs))
	read := func(r *archICBReplay) []byte {
		o := make([]byte, r.dModel*bf16Size)
		copy(o, unsafe.Slice((*byte)(r.lastOut.Contents()), r.dModel*bf16Size))
		return o
	}
	withAutoreleasePool(func() {
		var prev *archICBReplay
		var prevCB metal.MTLCommandBuffer
		var prevT int
		for t := range inputs {
			r := rs[t%2]
			r.prepareStep(inputs[t], t, nil)
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			enc.ExecuteCommandsInBufferWithRange(r.icb, r.rng)
			enc.EndEncoding()
			cb.Commit() // submit t WITHOUT waiting — overlaps t-1's GPU compute
			if prevCB != nil {
				prevCB.WaitUntilCompleted()
				outputs[prevT] = read(prev)
			}
			prev, prevCB, prevT = r, cb, t
		}
		if prevCB != nil {
			prevCB.WaitUntilCompleted()
			outputs[prevT] = read(prev)
		}
	})
	return outputs
}

// TestSpikePipelinedMatchesSerial — the double-buffered pipelined replay must match the serial runBatch
// (same caches/ops; only the submission overlaps). Gates correctness before the perf number.
func TestSpikePipelinedMatchesSerial(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)
	serial, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("serial: %v", err)
	}
	pipe := decodeArchICBQuantPipelined(t, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	for tok := range serial {
		if cos := cosineBF16(pipe[tok], serial[tok]); cos < 0.9999 {
			t.Fatalf("tok%d: pipelined cosine=%.6f vs serial — overlap broke a dependency", tok, cos)
		}
	}
	t.Logf("pipelined replay matches serial across %d tokens", len(serial))
}

// BenchmarkSpikeE2BDecodePipelined — double-buffered pipelined replay vs the serial BenchmarkSpikeE2BDecode.
func BenchmarkSpikeE2BDecodePipelined(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeArchICBQuantPipelined(b, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	}
}

// TestSpikeGPUvsWall splits the decode wall into GPU-busy span vs host overhead (per-token
// WaitUntilCompleted turnaround + submit + read). A large host-overhead fraction is the idle that a
// pipelined / submit-ahead decode loop (pkg/metal's PipelinedDecode) reclaims — no kernel change.
func TestSpikeGPUvsWall(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)
	// warm
	_, _ = DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	pieceTimingOn = true
	icbGPUNs = 0
	start := time.Now()
	if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		pieceTimingOn = false
		t.Fatalf("decode: %v", err)
	}
	wall := time.Since(start)
	gpu := time.Duration(icbGPUNs)
	pieceTimingOn = false
	t.Logf("64-token decode: wall=%v  GPU-busy=%v  host-overhead=%v (%.0f%%)", wall, gpu, wall-gpu, 100*float64(wall-gpu)/float64(wall))
}

// THROWAWAY spike instrument. e2b-scale ICB quant decode (dModel=1536, gs64/b4, 6 layers, 64 tokens so
// the one-off ICB build amortises and the per-token REPLAY dominates). Real dims (input-rms fusion
// engages). Synthetic weights — the perf delta is the dispatch/barrier structure, not the values.
// Measures pkg/native (NOT lthn-mlx serve, which is the pkg/metal cgo engine).
func spikeE2BFixture(tb testing.TB) (inputs [][]byte, layers []QuantizedLayerWeights, arch model.Arch) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 1536, 8, 1, 256, 6144, 32, 6
	const groupSize, bits = 64, 4
	const nTokens = 64
	arch = archFixture(tb, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs = decodeInputsFixture(nTokens, dModel)
	layers = make([]QuantizedLayerWeights, nLayers)
	for li := range layers {
		layers[li] = quantizedLayerFixture(tb, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, li+3)
	}
	if b, ok := tb.(*testing.B); ok {
		b.SetBytes(int64(nTokens))
	}
	return inputs, layers, arch
}

// TestSpikeFineGrainedReplayMatchesCoarse — the fine-grained replay (encoder memory barriers) must
// produce byte-identical output to the coarse-barrier ICB replay (same recorded ops, same deps, just a
// different barrier mechanism). Gates correctness before trusting the perf number.
func TestSpikeFineGrainedReplayMatchesCoarse(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)

	fineGrainedReplay = false
	coarse, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("coarse: %v", err)
	}
	fineGrainedReplay = true
	fine, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	fineGrainedReplay = false
	if err != nil {
		t.Fatalf("fine: %v", err)
	}
	for tok := range coarse {
		if cos := cosineBF16(fine[tok], coarse[tok]); cos < 0.9999 {
			t.Fatalf("tok%d: fine-grained replay cosine=%.6f vs coarse — memory barriers don't enforce the deps", tok, cos)
		}
	}
	t.Logf("fine-grained replay matches coarse across %d tokens", len(coarse))
}

// BenchmarkSpikeE2BDecodeFineGrained — fine-grained (encoder memory-barrier) replay vs the coarse
// BenchmarkSpikeE2BDecode. If faster, the memory barrier pipelines where the coarse SetBarrier drains.
func BenchmarkSpikeE2BDecodeFineGrained(b *testing.B) {
	fineGrainedReplay = true
	defer func() { fineGrainedReplay = false }()
	spikeE2BDecode(b)
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
