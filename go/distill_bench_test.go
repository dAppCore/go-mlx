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

package mlx

import (
	"testing"

	"dappco.re/go/mlx/probe"
)

var (
	distillBenchSinkLogits DistillLogits
)

// BenchmarkDistill_CloneLogits — the per-step teacher-logit clone that
// runs on every cache Put + Get. Sized to a realistic mid-tier
// distillation step: B=4, S=128, V=32000 (~16MB float32 / batch).
// Tracks the per-alloc count + per-byte cost as the per-cell inner
// makes are the high-watermark allocators in production distillation.
func BenchmarkDistill_CloneLogits(b *testing.B) {
	const (
		batch  = 4
		seqLen = 128
		vocab  = 32000
	)
	src := make(DistillLogits, batch)
	for i := range src {
		src[i] = make([][]float32, seqLen)
		for j := range src[i] {
			src[i][j] = make([]float32, vocab)
			for k := range src[i][j] {
				src[i][j][k] = float32(k)
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distillBenchSinkLogits = cloneDistillLogits(src)
	}
}

// BenchmarkDistill_CloneLogitsSmall — smaller per-step shape that
// dominates short-context distillation (B=2, S=32, V=4096). Tracks
// the alloc-count overhead at smaller shapes where the per-row
// outer + per-cell inner allocations are the dominant cost.
func BenchmarkDistill_CloneLogitsSmall(b *testing.B) {
	const (
		batch  = 2
		seqLen = 32
		vocab  = 4096
	)
	src := make(DistillLogits, batch)
	for i := range src {
		src[i] = make([][]float32, seqLen)
		for j := range src[i] {
			src[i][j] = make([]float32, vocab)
			for k := range src[i][j] {
				src[i][j][k] = float32(k)
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distillBenchSinkLogits = cloneDistillLogits(src)
	}
}

// distillBenchProbeSink is a no-clone probe sink that captures the
// last event by value — used by benchmarks so the EmitProbe path
// stays free of the Recorder's clone-and-append cost.
type distillBenchProbeSink struct {
	last probe.Event
}

func (s *distillBenchProbeSink) EmitProbe(event probe.Event) {
	s.last = event
}

var (
	distillBenchSinkProbe distillBenchProbeSink
	distillBenchStepSink  string
)

// BenchmarkDistill_EmitProbe — per-gradient-step probe emission.
// Allocates a 7-entry meta map per call plus a probe.Training
// payload, calls strconv.FormatFloat once and core.Itoa twice. Runs
// once per training step inside runDistillEpoch when a ProbeSink is
// wired up, which is the typical "watch the run" production
// configuration.
func BenchmarkDistill_EmitProbe(b *testing.B) {
	cfg := DistillConfig{
		Temperature:  2.0,
		Loss:         DistillLossKL,
		LearningRate: 1e-4,
		ProbeSink:    &distillBenchSinkProbe,
	}
	result := &DistillResult{
		Metrics:     DistillMetrics{Steps: 1234},
		Checkpoints: []string{"a", "b", "c"},
		Evaluations: []DistillEvalResult{{Step: 1}, {Step: 2}},
	}
	loss := DistillLoss{
		Value:       0.4321,
		Tokens:      512,
		Temperature: 2.0,
		Kind:        DistillLossKL,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		emitDistillProbe(cfg, result, &loss, "miss", 1)
	}
}

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
