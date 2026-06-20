// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark for the OUTER safetensors→quantized-GGUF pipeline,
// QuantizeModelPack. The inner per-tensor streaming writer
// (writeQuantizedGGUFStream / writeQuantizedGGUFTensorStream) has its own
// bench (quantize_writer_bench_test.go) and is already alloc-minimal; this
// bench targets the ORCHESTRATION that wraps it — the safetensors index
// build, buildStreamingGGUFQuantizedTensors (per-tensor descriptor/ref
// construction), ggufQuantizeMetadata, the model-pack metadata copy, and
// the final ReadInfo validation re-parse.
//
// AX-11: QuantizeModelPack runs once per model conversion, but the
// per-tensor work inside it (one ggufQuantizedTensor + one TensorRef +
// previously one cloned Shape slice per tensor) scales with model size —
// exactly where unmeasured allocs hide. The fixture uses MANY tiny tensors
// so per-tensor FLAT pprof lines stand out above the per-model constant
// (mkdir, header, ReadInfo) rather than a few large tensors where payload
// decode would dwarf the orchestration.
//
// Each iteration converts to a UNIQUE output directory because
// QuantizeModelPack refuses a non-empty destination
// (ensureEmptyGGUFQuantizeDestination). The output paths are precomputed
// before ResetTimer so the strconv/PathJoin allocs for the path strings do
// not clutter the orchestration profile. The per-iteration MkdirAll +
// copyModelPackMetadata are INSIDE the function under test and therefore
// measured — that is inherent to the pipeline, not a target.
//
// Run:
//   go test -bench='BenchmarkQuantizeModelPack' -benchmem -benchtime=20x -run='^$' ./go/gguf
//   go test -bench='BenchmarkQuantizeModelPack_Q8_0' -benchmem -benchtime=50x -run='^$' \
//     -memprofile=/tmp/qmp.mem ./go/gguf
//   go tool pprof -alloc_objects -list 'QuantizeModelPack|buildStreamingGGUFQuantizedTensors' /tmp/qmp.mem

package gguf

import (
	"context"
	"strconv"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// Sinks defeat compiler DCE.
var (
	qmpSinkResult *QuantizeResult
	qmpSinkErr    error
)

// benchModelPackTensors lays out nTensors small F32 weights whose first
// dimension is blockSize (so every quantize format's divisibility check
// passes) and whose element count is a whole number of blocks. Each tensor
// is distinct so the per-tensor descriptor/ref/shape construction in
// buildStreamingGGUFQuantizedTensors runs nTensors times — the scaling
// stage AX-11 cares about.
func benchModelPackTensors(blockSize, cols, nTensors int) []safetensorTestTensor {
	tensors := make([]safetensorTestTensor, nTensors)
	elements := blockSize * cols
	for i := range tensors {
		tensors[i] = safetensorTestTensor{
			Name:  "model.layers." + strconv.Itoa(i) + ".self_attn.q_proj.weight",
			Shape: []int{blockSize, cols},
			Data:  benchQuantizeValues(elements),
		}
	}
	return tensors
}

// benchQuantizeModelPack drives the whole pipeline once per iteration into a
// fresh output directory. blockSize must match the format's GGUF block size
// (32 for *_0, 256 for K-quants) so the source tensors are valid input.
func benchQuantizeModelPack(b *testing.B, format QuantizeFormat, blockSize int) {
	const (
		cols     = 4  // 4 blocks per row → modest payload, many tensors
		nTensors = 48 // header walk + per-tensor descriptor build dominate
	)
	source := writeDenseSafetensorsPack(b, "qwen3", benchModelPackTensors(blockSize, cols, nTensors))
	pack := sourcePackFromDir(source)

	// Precompute unique output paths so the path-string allocs land outside
	// the measured region (each iteration needs an empty destination).
	base := b.TempDir()
	outs := make([]string, b.N)
	for i := range outs {
		outs[i] = core.PathJoin(base, "out"+strconv.Itoa(i))
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qmpSinkResult, qmpSinkErr = QuantizeModelPack(ctx, QuantizeOptions{
			SourcePack: pack,
			OutputPath: outs[i],
			Format:     format,
		})
		if qmpSinkErr != nil {
			b.Fatalf("QuantizeModelPack(%s): %v", format, qmpSinkErr)
		}
	}
}

// benchQuantizeModelPackLabels mirrors benchQuantizeModelPack but attaches
// labels so the ggufQuantizeMetadata label-append + sort path is exercised
// (it is skipped entirely when Labels is nil).
func benchQuantizeModelPackLabels(b *testing.B, format QuantizeFormat, blockSize int) {
	const (
		cols     = 4
		nTensors = 48
	)
	source := writeDenseSafetensorsPack(b, "qwen3", benchModelPackTensors(blockSize, cols, nTensors))
	pack := sourcePackFromDir(source)
	labels := map[string]string{"author": "lethean", "run": "bench", "stage": "q"}

	base := b.TempDir()
	outs := make([]string, b.N)
	for i := range outs {
		outs[i] = core.PathJoin(base, "out"+strconv.Itoa(i))
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qmpSinkResult, qmpSinkErr = QuantizeModelPack(ctx, QuantizeOptions{
			SourcePack: pack,
			OutputPath: outs[i],
			Format:     format,
			Labels:     labels,
		})
		if qmpSinkErr != nil {
			b.Fatalf("QuantizeModelPack(%s, labels): %v", format, qmpSinkErr)
		}
	}
	_ = mp.ModelPackFormatSafetensors // keep mp import meaningful if pack helper changes
}

// Q8_0 — the headline case (default format, 32-element blocks).
func BenchmarkQuantizeModelPack_Q8_0(b *testing.B) { benchQuantizeModelPack(b, QuantizeQ8_0, 32) }

// Q4_K — 256-element K-quant block; exercises the larger bytesPerBlock /
// Size path through buildStreamingGGUFQuantizedTensors.
func BenchmarkQuantizeModelPack_Q4_K(b *testing.B) { benchQuantizeModelPack(b, QuantizeQ4_K, 256) }

// Labels variant — exercises the metadata label key-sort + append branch.
func BenchmarkQuantizeModelPack_Q8_0_Labels(b *testing.B) {
	benchQuantizeModelPackLabels(b, QuantizeQ8_0, 32)
}
