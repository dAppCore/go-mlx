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
