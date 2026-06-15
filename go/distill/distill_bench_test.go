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
	"dappco.re/go/mlx/probe"
)

var (
	distillBenchSinkLogits DistillLogits
)

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
