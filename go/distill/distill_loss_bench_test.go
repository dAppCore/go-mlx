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

// BenchmarkDistill_BatchLoss — per-step distillation loss kernel.
// Realistic short-context shape (B=2, S=8, V=128) — keeps each call
// fast enough for high b.N while still exercising the masked-path
// inner loop and the log-softmax + prob accumulator. Allocates the
// scratch buffers on the first call; subsequent calls reuse them.
func BenchmarkDistill_BatchLoss(b *testing.B) {
	const (
		batch  = 2
		seqLen = 8
		vocab  = 128
	)
	teacher := make(DistillLogits, batch)
	student := make(DistillLogits, batch)
	mask := make([][]float32, batch)
	for i := range batch {
		teacher[i] = make([][]float32, seqLen)
		student[i] = make([][]float32, seqLen)
		mask[i] = make([]float32, seqLen)
		for j := range seqLen {
			teacher[i][j] = make([]float32, vocab)
			student[i][j] = make([]float32, vocab)
			for k := range vocab {
				teacher[i][j][k] = float32((k * 7) % 13)
				student[i][j][k] = float32((k * 5) % 11)
			}
			mask[i][j] = 1
		}
	}
	cfg := DistillConfig{Loss: DistillLossKL, Temperature: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loss, err := DistillationBatchLoss(teacher, student, mask, cfg)
		if err != nil {
			b.Fatal(err)
		}
		distillBenchLossSink = loss
	}
}

// BenchmarkDistill_BatchLossNoMask — same shape, no mask (the
// teacher-forcing hot path that avoids the per-j maskRow[j] gate).
func BenchmarkDistill_BatchLossNoMask(b *testing.B) {
	const (
		batch  = 2
		seqLen = 8
		vocab  = 128
	)
	teacher := make(DistillLogits, batch)
	student := make(DistillLogits, batch)
	for i := range batch {
		teacher[i] = make([][]float32, seqLen)
		student[i] = make([][]float32, seqLen)
		for j := range seqLen {
			teacher[i][j] = make([]float32, vocab)
			student[i][j] = make([]float32, vocab)
			for k := range vocab {
				teacher[i][j][k] = float32((k * 7) % 13)
				student[i][j][k] = float32((k * 5) % 11)
			}
		}
	}
	cfg := DistillConfig{Loss: DistillLossKL, Temperature: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loss, err := DistillationBatchLoss(teacher, student, nil, cfg)
		if err != nil {
			b.Fatal(err)
		}
		distillBenchLossSink = loss
	}
}

var distillBenchCacheKeySink string

// BenchmarkDistill_BatchCacheKey — per-step teacher-cache key build.
// Fires once per step inside runDistillEpoch when TeacherCache is
// wired. JSON-marshals the SFTBatch + SHA256 over the result. The
// allocation bill is the marshal buffer + the hex-string return.
func BenchmarkDistill_BatchCacheKey(b *testing.B) {
	const (
		batch  = 2
		seqLen = 16
	)
	tokens := make([][]int, batch)
	targets := make([][]int, batch)
	mask := make([][]float32, batch)
	for i := range batch {
		tokens[i] = make([]int, seqLen)
		targets[i] = make([]int, seqLen)
		mask[i] = make([]float32, seqLen)
		for j := range seqLen {
			tokens[i][j] = i*seqLen + j
			targets[i][j] = (i*seqLen + j + 1) % 32000
			mask[i][j] = 1
		}
	}
	batchData := SFTBatch{
		Batch:   Batch{Tokens: tokens, LossMask: mask},
		Targets: targets,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distillBenchCacheKeySink = DistillBatchCacheKey(batchData)
	}
}
