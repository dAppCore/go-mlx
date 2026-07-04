// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
	"time"
)

var traceBenchPhaseSink []TokenPhaseTrace

func BenchmarkTokenPhaseTraceAppend_Nil1024(b *testing.B) {
	start := time.Now()
	phase := TokenPhaseTrace{Step: 1, ForwardDuration: time.Millisecond}

	b.ReportAllocs()
	for b.Loop() {
		var phases []TokenPhaseTrace
		for range 1024 {
			phases = appendTokenPhaseTrace(phases, phase, start)
		}
		traceBenchPhaseSink = phases
	}
}

func BenchmarkTokenPhaseTraceAppend_Preallocated1024(b *testing.B) {
	start := time.Now()
	phase := TokenPhaseTrace{Step: 1, ForwardDuration: time.Millisecond}
	cfg := GenerateConfig{MaxTokens: 1024, TraceTokenPhases: true}

	b.ReportAllocs()
	for b.Loop() {
		phases := newTokenPhaseTraceBuffer(cfg)
		for range 1024 {
			phases = appendTokenPhaseTrace(phases, phase, start)
		}
		traceBenchPhaseSink = phases
	}
}
