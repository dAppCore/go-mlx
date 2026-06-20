// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for distill.go — knowledge distillation pipeline.
// Per AX-11 — cloneDistillLogits fires on every teacher-cache Put
// (cache miss path) and every Get (cache hit path); for B*S*V tensors
// with B=4, S=128, V=32000, the alloc shape sets the per-step memory
// pressure of any distillation run with teacher caching enabled.
// emitDistillProbe / runDistillEpoch probe meta build per gradient
// step. Pinning these alloc shapes is the load-bearing AX commitment
// of this file.
//
// Run:    go test -bench='BenchmarkDistill' -benchmem -run='^$' ./go

package distill

import (
	"testing"
)

// BenchmarkDistill_FormatStepDir — per-checkpoint dirname builder.
// Runs once per checkpoint save and the alloc is the returned string
// itself; the int-to-decimal conversion fires on the hot path.
func BenchmarkDistill_FormatStepDir(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distillBenchStepSink = formatDistillStepDir(123456)
	}
}

// BenchmarkDistill_FormatStepDirSmall — small step value, exercising
// the zero-pad arm of formatDistillStepDir (step < 100000).
func BenchmarkDistill_FormatStepDirSmall(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distillBenchStepSink = formatDistillStepDir(42)
	}
}

// BenchmarkDistill_NewCheckpointMetadata — per-checkpoint metadata
// build (struct populate; no I/O). Fires on every checkpoint step
// inside maybeSaveDistillCheckpoint.
func BenchmarkDistill_NewCheckpointMetadata(b *testing.B) {
	cfg := DistillConfig{
		Temperature: 2,
		Loss:        DistillLossKL,
		ResumePath:  "/tmp/resume",
	}
	result := &DistillResult{
		Metrics: DistillMetrics{Steps: 100, Samples: 800, Tokens: 51200},
		Teacher: ModelInfo{Architecture: "qwen3", VocabSize: 32000},
		Student: ModelInfo{Architecture: "qwen3", VocabSize: 32000},
	}
	loss := DistillLoss{
		Value:            0.4,
		KL:               0.4,
		SoftCrossEntropy: 0.5,
		TeacherEntropy:   0.1,
		Tokens:           512,
		Temperature:      2,
		Kind:             DistillLossKL,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewDistillCheckpointMetadata("/tmp/ckpt", cfg, result, loss, 1)
	}
}

// BenchmarkDistill_LoadCheckpointMetadata — the resume read path:
// core.ReadFile the sidecar then core.JSONUnmarshal it into a fresh
// DistillCheckpointMetadata. The sidecar is written once into a temp
// dir from a realistic metadata value; the loop measures the steady-
// state read+decode. The allocs here are stdlib json + the file-read
// buffer + the returned *meta escape, none package-reducible — this
// bench pins that floor so a regression (e.g. an un-presized decode
// helper slipped into the loader body) would show up.
func BenchmarkDistill_LoadCheckpointMetadata(b *testing.B) {
	cfg := DistillConfig{
		Temperature: 2,
		Loss:        DistillLossKL,
		ResumePath:  "/tmp/resume",
	}
	result := &DistillResult{
		Metrics: DistillMetrics{Steps: 100, Samples: 800, Tokens: 51200, TeacherCacheHits: 9, TeacherCacheMisses: 3},
		Teacher: ModelInfo{Architecture: "qwen3", VocabSize: 32000},
		Student: ModelInfo{Architecture: "qwen3", VocabSize: 32000},
	}
	loss := DistillLoss{Value: 0.4, KL: 0.4, SoftCrossEntropy: 0.5, TeacherEntropy: 0.1}
	dir := b.TempDir()
	meta := NewDistillCheckpointMetadata(dir, cfg, result, loss, 1)
	if err := SaveDistillCheckpointMetadata(dir, meta); err != nil {
		b.Fatalf("seed SaveDistillCheckpointMetadata: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loaded, err := LoadDistillCheckpointMetadata(dir)
		if err != nil {
			b.Fatalf("LoadDistillCheckpointMetadata: %v", err)
		}
		distillBenchMetaSink = loaded
	}
}

var (
	distillBenchLossSink DistillLoss
	distillBenchMetaSink *DistillCheckpointMetadata
)
