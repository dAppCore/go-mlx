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

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// BenchmarkSFT_LoRAMetadata measures the per-checkpoint clone of
// TargetKeys/TargetLayers when persisting metadata.
func BenchmarkSFT_LoRAMetadata(b *testing.B) {
	cfg := spine.LoRAConfig{
		Rank:         8,
		Alpha:        16,
		TargetKeys:   []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		TargetLayers: []string{"layer.0", "layer.1", "layer.2", "layer.3"},
		DType:        metal.DTypeFloat32,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkLoRA = sftLoRAMetadata(cfg)
	}
}

// BenchmarkSFT_StepName tracks the checkpoint directory-name builder
// — runs every CheckpointEvery steps during long training runs.
func BenchmarkSFT_StepName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sftBenchSinkStepName = sftStepName(12345)
	}
}
