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

	core "dappco.re/go"
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

// sftLoadBenchConfig builds a representative SFT config the way a real
// LoRA run does — populated LoRA target keys/layers and a couple of eval
// prompts — so the persisted sidecar JSON is realistic rather than
// inflated. Shared by the two loader benches below.
func sftLoadBenchConfig() SFTConfig {
	return SFTConfig{
		LoRA: spine.LoRAConfig{
			Rank:         8,
			Alpha:        16,
			TargetKeys:   []string{"q_proj", "k_proj", "v_proj", "o_proj"},
			TargetLayers: []string{"layer.0", "layer.1", "layer.2", "layer.3"},
			DType:        metal.DTypeFloat32,
		},
		BatchSize:                 4,
		GradientAccumulationSteps: 2,
		LearningRate:              1e-5,
		MaxSeqLen:                 2048,
		EvalPrompts:               []string{"explain rope scaling", "summarise the diff"},
		EvalTemperature:           0.7,
	}
}

// BenchmarkSFT_LoadCheckpointMetadata — the public resume read path:
// core.ReadFile the sidecar then core.JSONUnmarshal it into a fresh
// SFTCheckpointMetadata (the richest of the three checkpoint structs —
// nested LoRA + AdamW + the eval-prompt slice). The sidecar is written
// once beside an adapter package from a realistic config; the loop
// measures the steady-state read+decode. The allocs are stdlib json +
// the file-read buffer + the returned *meta escape, none package-
// reducible — this bench pins that floor.
func BenchmarkSFT_LoadCheckpointMetadata(b *testing.B) {
	dir := b.TempDir()
	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", sftLoadBenchConfig(), &SFTResult{Steps: 120, OptimizerSteps: 60, Epochs: 2, Samples: 480, LastLoss: 0.42, LastValLoss: 0.51}, 2)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		b.Fatalf("seed SaveSFTCheckpointMetadata: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loaded, err := LoadSFTCheckpointMetadata(adapterPath)
		if err != nil {
			b.Fatalf("LoadSFTCheckpointMetadata: %v", err)
		}
		sftBenchSinkMeta = loaded
	}
}

// BenchmarkSFT_ApplyResumeMetadata — the resume-attach path. Distinct
// body from the public Load: it routes through loadSFTResumeMetadata
// (the missing-file -> (nil,nil) variant) and stamps result.ResumePath
// / result.ResumedFrom. A sidecar is seeded once; the loop reuses one
// *SFTResult and a ResumePath-set config (an empty ResumePath would take
// the early nil return and measure nothing).
func BenchmarkSFT_ApplyResumeMetadata(b *testing.B) {
	dir := b.TempDir()
	resumePath := core.PathJoin(dir, "prev.safetensors")
	prev := NewSFTCheckpointMetadata(resumePath, "gemma4", sftLoadBenchConfig(), &SFTResult{Steps: 90, Epochs: 1, Samples: 360, LastLoss: 0.55}, 1)
	if err := SaveSFTCheckpointMetadata(resumePath, prev); err != nil {
		b.Fatalf("seed SaveSFTCheckpointMetadata: %v", err)
	}
	cfg := SFTConfig{ResumePath: resumePath}
	result := &SFTResult{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ApplySFTResumeMetadata(result, cfg); err != nil {
			b.Fatalf("ApplySFTResumeMetadata: %v", err)
		}
	}
	sftBenchSinkMeta = result.ResumedFrom
}

var sftBenchSinkMeta *SFTCheckpointMetadata
