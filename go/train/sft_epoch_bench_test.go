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
)

// BenchmarkSFT_RunProbeMeta mirrors the runSFTBatchGroup probe.Event.Meta
// construction (6 string fields, all int-formatted today via Sprintf).
// Fires once per gradient step when a probe sink is attached.
func BenchmarkSFT_RunProbeMeta(b *testing.B) {
	cfg := SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, SequencePacking: true}
	cfg = normalizeSFTConfig(cfg)
	optimizerSteps := 1234
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkMap = sftBenchBuildProbeMeta(cfg, optimizerSteps)
	}
}

// BenchmarkSFT_StreamingPacker — exercise the per-sample packer add
// + final flush path. maxSeqLen=64, 8 samples of length 6 (no trim,
// no mid-add flush) → tests the pre-sized accumulator growth.
func BenchmarkSFT_StreamingPacker(b *testing.B) {
	ex := sftExample{
		inputs:  []int{1, 2, 3, 4, 5, 6},
		targets: []int{2, 3, 4, 5, 6, 7},
		mask:    []float32{0, 0, 0, 1, 1, 1},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packer := newSFTStreamingPacker(64, func(sftExample) error { return nil })
		for range 8 {
			_ = packer.add(ex)
		}
		_ = packer.finish()
	}
}
