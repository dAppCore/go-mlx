// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the spine dispatch shufflers — ToMetalGenerateConfig and
// ToMetalProbeSink. Per AX-11 — both fire on every Generate / Chat /
// Classify / BatchGenerate call, so the per-call allocation budget for
// the inference hot path runs through here.
//
// Run:    go test -bench='BenchmarkSpine_' -benchmem -run='^$' ./spine

package spine

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE.
var (
	spineBenchSinkMetalCfg    metal.GenerateConfig
	spineBenchSinkMetalSink   metal.ProbeSink
	spineBenchSinkProbeLogits []probe.Logit
)

// noopProbeSink is a minimal probe.Sink that drops every event — used by
// the ToMetalProbeSink benchmark to exercise the non-nil dispatch path
// without paying for downstream event-conversion work.
type noopProbeSink struct{}

// EmitProbe drops the event.
func (noopProbeSink) EmitProbe(probe.Event) {}

// --- ToMetalGenerateConfig ---
// Per-call shuffler from the spine GenerateConfig into the metal package
// equivalent. Inlined into every Generate / Chat / Classify entry — the
// per-call allocation pattern here drives the dispatch-side budget.

func BenchmarkSpine_ToMetalGenerateConfig_NoSink(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		Seed:          42,
		SeedSet:       true,
		RepeatPenalty: 1.1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkMetalCfg = ToMetalGenerateConfig(cfg)
	}
}

func BenchmarkSpine_ToMetalGenerateConfig_WithSink(b *testing.B) {
	sink := noopProbeSink{}
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		Seed:          42,
		SeedSet:       true,
		RepeatPenalty: 1.1,
		ProbeSink:     sink,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkMetalCfg = ToMetalGenerateConfig(cfg)
	}
}

// --- ToMetalProbeSink ---
// Per-call closure/adapter allocator. Fires once per Generate / Chat /
// Classify entry. The nil-sink path is the steady-state (most calls
// don't request probes); the non-nil path is the trace hot path.

func BenchmarkSpine_ToMetalProbeSink_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkMetalSink = ToMetalProbeSink(nil)
	}
}

func BenchmarkSpine_ToMetalProbeSink_NonNil(b *testing.B) {
	sink := noopProbeSink{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkMetalSink = ToMetalProbeSink(sink)
	}
}

// --- toProbeLogits (W10-AN) ---
// Per-probe-event slice clone — metal.ProbeLogit and probe.Logit have
// bit-identical layout (int32 + float32 + float64). Top-K is commonly
// 50-100 entries per probe.Logits, emitted per-token when ProbeSink is
// enabled. Benches the empty / typical / large fan-outs to surface the
// per-element struct unpacking cost vs a direct slab copy.

func BenchmarkSpine_ToProbeLogits_Empty(b *testing.B) {
	var logits []metal.ProbeLogit
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeLogits = toProbeLogits(logits)
	}
}

func BenchmarkSpine_ToProbeLogits_Typical(b *testing.B) {
	logits := make([]metal.ProbeLogit, 50)
	for i := range logits {
		logits[i] = metal.ProbeLogit{TokenID: int32(i), Logit: float32(i) * 0.1, Probability: float64(i) * 0.001}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeLogits = toProbeLogits(logits)
	}
}

func BenchmarkSpine_ToProbeLogits_Large(b *testing.B) {
	logits := make([]metal.ProbeLogit, 256)
	for i := range logits {
		logits[i] = metal.ProbeLogit{TokenID: int32(i), Logit: float32(i) * 0.1, Probability: float64(i) * 0.001}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeLogits = toProbeLogits(logits)
	}
}
