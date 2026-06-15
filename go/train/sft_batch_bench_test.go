// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for sft.go — supervised LoRA fine-tuning pipeline.
// Per AX-11 — probe meta builds per gradient step (hundreds/thousands per
// training run); SFTLoRAMetadata clone fires per checkpoint + per final
// adapter save; sftBatchFromExamples runs once per accumulated batch
// (one per BatchSize samples). Pinning the alloc shape of these hot
// paths is the load-bearing AX commitment of this file.
//
// Run:    go test -bench='BenchmarkSFT' -benchmem -run='^$' ./go

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// BenchmarkSFT_BatchFromExamples mirrors sftBatchFromExamples — runs
// once per gradient accumulation flush (BatchSize examples).
func BenchmarkSFT_BatchFromExamples(b *testing.B) {
	examples := make([]sftExample, 8)
	for i := range examples {
		examples[i] = sftExample{
			inputs:  []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			targets: []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			mask:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkBatch = sftBatchFromExamples(examples)
	}
}

// BenchmarkSFT_HasTrainingTarget exercises the mask scan executed once
// per buildSFTExample.
func BenchmarkSFT_HasTrainingTarget(b *testing.B) {
	mask := make([]float32, 256)
	mask[200] = 1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = hasTrainingTarget(mask)
	}
}

// BenchmarkSFT_HasTrainingTarget_AllZero — worst case (full scan).
func BenchmarkSFT_HasTrainingTarget_AllZero(b *testing.B) {
	mask := make([]float32, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = hasTrainingTarget(mask)
	}
}

// BenchmarkSFT_BuildExample exercises buildSFTExample end-to-end with
// a fake tokenizer — the per-sample hot path of every SFT run.
func BenchmarkSFT_BuildExample(b *testing.B) {
	tok := spine.NewTokenizer(sftBenchTokenizer{
		encoded: map[string][]int32{
			"prompt":   {10, 11, 12, 13},
			"response": {20, 21, 22, 23, 24, 25, 26, 27},
		},
		eos: 2,
	})
	sample := dataset.Sample{Prompt: "prompt", Response: "response"}
	cfg := SFTConfig{BatchSize: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkExample, _, _ = buildSFTExample(tok, sample, cfg)
	}
}

// BenchmarkSFT_BatchBuilderFinish mirrors the final batch flush + clone.
func BenchmarkSFT_BatchBuilderFinish(b *testing.B) {
	example := sftExample{
		inputs:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		targets: []int{2, 3, 4, 5, 6, 7, 8, 9},
		mask:    []float32{0, 0, 1, 1, 1, 1, 1, 1},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := newSFTBatchBuilder(2)
		for range 8 {
			builder.add(example)
		}
		_ = builder.finish()
	}
}
