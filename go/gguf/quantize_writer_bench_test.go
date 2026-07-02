// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the streaming GGUF quantize-write path.
//
// AX-11: writeQuantizedGGUFStream is the production safetensors→GGUF
// path. It loops every tensor and, per tensor, loops fixed-size element
// CHUNKS — for a multi-GB tensor that is millions of chunk iterations.
// The per-chunk cost (read a float32 window, quantise it, write it) is
// the high-multiplier hot path; the existing per-kernel benches measure
// a single whole-tensor call and never exercise the chunk loop.
//
// These benches force a SMALL chunkElements on several modest tensors so
// chunk count is high and the per-chunk allocations dominate the constant
// OpenReader / header-write setup — surfacing B/op and allocs/op the
// kernel benches are blind to. Inputs stay tiny (AX-11: no model loads).
//
// Run: go test -bench='BenchmarkWriteQuantizedStream' -benchmem -run='^$' ./go/gguf

package gguf

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// benchStreamFixture builds a small multi-tensor safetensors source on
// disk and returns the streaming tensor metadata + refs the writer
// consumes. Each tensor is sized so that, with a small chunkElements,
// the writer iterates many chunks per tensor.
func benchStreamFixture(b *testing.B, format QuantizeFormat, blockSize, elementsPerTensor, nTensors int) ([]ggufQuantizedTensor, []safetensors.TensorRef, []ggufMetadataEntry) {
	b.Helper()
	// First dim must be divisible by blockSize (buildStreamingGGUFQuantizedTensors
	// enforces it). Use Shape[0]=blockSize, Shape[1]=elementsPerTensor/blockSize.
	cols := elementsPerTensor / blockSize
	tensors := make([]safetensorTestTensor, nTensors)
	for t := range tensors {
		tensors[t] = safetensorTestTensor{
			Name:  "blk." + core.Sprintf("%d", t) + ".attn.weight",
			Shape: []int{blockSize, cols},
			Data:  benchQuantizeValues(elementsPerTensor),
		}
	}
	source := core.PathJoin(b.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(b, source, tensors)
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		b.Fatalf("index safetensors: %v", err)
	}
	quant, refs, err := buildStreamingGGUFQuantizedTensors(index, format)
	if err != nil {
		b.Fatalf("build streaming tensors: %v", err)
	}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, format, nil)
	return quant, refs, metadata
}

// benchWriteQuantizedStream runs the whole streaming writer with a forced
// small chunkElements so the per-chunk read+quantise+write path dominates.
func benchWriteQuantizedStream(b *testing.B, format QuantizeFormat, blockSize int) {
	const (
		elementsPerTensor = 256 * 64 // 16384 elements → 64 K-quant blocks / 512 Q-blocks
		nTensors          = 8
	)
	tensors, refs, metadata := benchStreamFixture(b, format, blockSize, elementsPerTensor, nTensors)
	// One chunk == one block: maximal chunk count, so the per-chunk
	// allocation behaviour is what the bench measures.
	chunkElements := blockSize
	dir := b.TempDir()
	out := core.PathJoin(dir, "streamed.gguf")
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := writeQuantizedGGUFStream(ctx, out, metadata, tensors, refs, format, chunkElements); err != nil {
			b.Fatalf("writeQuantizedGGUFStream: %v", err)
		}
	}
}

func BenchmarkWriteQuantizedStream_Q8_0(b *testing.B) { benchWriteQuantizedStream(b, QuantizeQ8_0, 32) }
func BenchmarkWriteQuantizedStream_Q4_0(b *testing.B) { benchWriteQuantizedStream(b, QuantizeQ4_0, 32) }
func BenchmarkWriteQuantizedStream_Q4_K(b *testing.B) {
	benchWriteQuantizedStream(b, QuantizeQ4_K, 256)
}
func BenchmarkWriteQuantizedStream_Q6_K(b *testing.B) {
	benchWriteQuantizedStream(b, QuantizeQ6_K, 256)
}
