// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for distill.go — knowledge distillation pipeline.
// emitDistillProbe / runDistillEpoch probe meta build per gradient step —
// the load-bearing AX commitment of this file. The teacher-logit clone
// benchmarks moved to the shared dappco.re/go/inference/distill engine's
// own MemoryLogitCache benchmarks, since MemoryDistillLogitCache is now an
// alias onto that implementation.
//
// Run:    go test -bench='BenchmarkDistill' -benchmem -run='^$' ./go

package distill

import (
	"dappco.re/go/inference/probe"
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
