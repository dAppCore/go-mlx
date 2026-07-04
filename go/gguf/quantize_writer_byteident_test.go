// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/safetensors"
)

// TestWriteQuantizedGGUFStream_ByteIdenticalToBuffered proves the chunked
// streaming writer — which reuses one set of per-MODEL scratch buffers
// (raw/values/out) across every tensor and every chunk — emits a file
// byte-for-byte identical to the buffered writer, which quantises each
// tensor whole via the one-shot quantizeQ* path. The two share offset/
// alignment/header logic, so a full-file compare covers payload bytes,
// tensor offsets, and inter-tensor padding together.
//
// The fixture is built to hit exactly the case the streaming bench does
// NOT: tensors of DIFFERING sizes (a large tensor's oversized backing
// array later reused for a small tensor) plus a last chunk that is a
// REMAINDER (chunkElements does not divide a tensor's element count).
// That is where a stale-tail leak from scratch reuse would surface; if
// the files match, reuse is byte-clean. chunkElements stays a multiple
// of blockSize so blocks never split across chunks and the whole-tensor
// quantizeQ* reference is exact.
func TestWriteQuantizedGGUFStream_ByteIdenticalToBuffered(t *testing.T) {
	formats := []struct {
		format    QuantizeFormat
		blockSize int
	}{
		{QuantizeQ8_0, 32},
		{QuantizeQ4_0, 32},
		{QuantizeQ5_0, 32},
		{QuantizeQ4_K, 256},
		{QuantizeQ5_K, 256},
		{QuantizeQ6_K, 256},
		{QuantizeQ8_K, 256},
		{QuantizeQ3_K, 256},
		{QuantizeQ2_K, 256},
	}
	for _, fc := range formats {
		t.Run(string(fc.format), func(t *testing.T) {
			bs := fc.blockSize
			// Mixed tensor sizes (in blocks): big, then small, then medium.
			// First dim = blockSize (divisible), second dim = block count.
			source := core.PathJoin(t.TempDir(), "source.safetensors")
			writeTestSafetensorsF32(t, source, []safetensorTestTensor{
				{Name: "blk.0.big.weight", Shape: []int{bs, 7}, Data: variedFloat32s(bs * 7)},
				{Name: "blk.1.small.weight", Shape: []int{bs, 1}, Data: variedFloat32s(bs * 1)},
				{Name: "blk.2.medium.weight", Shape: []int{bs, 3}, Data: variedFloat32s(bs * 3)},
			})
			index, err := safetensors.IndexFiles([]string{source})
			if err != nil {
				t.Fatalf("index safetensors: %v", err)
			}
			metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, fc.format, nil)

			// Streaming path with a small chunk that yields a remainder on
			// the 7-block and 3-block tensors: 2 blocks per chunk.
			chunkElements := bs * 2
			streamTensors, refs, err := buildStreamingGGUFQuantizedTensors(index, fc.format)
			if err != nil {
				t.Fatalf("build streaming tensors: %v", err)
			}
			streamPath := core.PathJoin(t.TempDir(), "streamed.gguf")
			if err := writeQuantizedGGUFStream(context.Background(), streamPath, metadata, streamTensors, refs, fc.format, chunkElements); err != nil {
				t.Fatalf("writeQuantizedGGUFStream: %v", err)
			}

			// Buffered reference: quantise each tensor whole (one-shot
			// quantizeQ*), same index order so offsets line up.
			dense, err := readDenseSafetensors(source)
			if err != nil {
				t.Fatalf("readDenseSafetensors: %v", err)
			}
			bufTensors := make([]ggufQuantizedTensor, 0, len(dense))
			for _, d := range dense {
				qt, err := quantizeGGUFTensor(d, fc.format)
				if err != nil {
					t.Fatalf("quantizeGGUFTensor %s: %v", d.Name, err)
				}
				bufTensors = append(bufTensors, qt)
			}
			bufferedPath := core.PathJoin(t.TempDir(), "buffered.gguf")
			if err := writeQuantizedGGUF(bufferedPath, metadata, bufTensors); err != nil {
				t.Fatalf("writeQuantizedGGUF: %v", err)
			}

			streamed := readFileOrFatal(t, streamPath)
			buffered := readFileOrFatal(t, bufferedPath)
			if !bytes.Equal(streamed, buffered) {
				t.Fatalf("%s: streamed file (%d B) != buffered file (%d B) — scratch reuse perturbed bytes; first diff at %d",
					fc.format, len(streamed), len(buffered), firstDiff(streamed, buffered))
			}
		})
	}
}

// variedFloat32s produces deterministic data whose per-block max-abs
// varies (so the scale path runs) and whose tail values differ from the
// head — making a stale-tail reuse leak observable rather than masked by
// constant data.
func variedFloat32s(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		x := float32((i*7)%251) - 125
		out[i] = x / 32
	}
	return out
}

func readFileOrFatal(t *testing.T, path string) []byte {
	t.Helper()
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("read %s: %v", path, read.Value)
	}
	return read.Value.([]byte)
}

func firstDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
