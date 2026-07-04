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
	"dappco.re/go/inference/probe"
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

// --- toProbeEvent (W10-AN) ---
// The per-probe-event metal→probe converter the metalProbeSinkAdapter calls
// on every event the engine emits. Under a ProbeSink the metal emit* helpers
// fire several events PER GENERATION TOKEN — emitProbeToken (Token-only),
// emitProbeLogits (a Logits-only event + an Entropy-only event), plus
// cache/memory pressure. Each metal emit* helper populates exactly ONE
// payload pointer, so the realistic per-call cost is a single-payload event,
// NOT the all-fields shape the correctness test exercises. Bench the shapes
// that actually occur so the per-token probe budget is measured, not a
// phantom 12-payload event that never happens in production.

var spineBenchSinkProbeEvent probe.Event

func BenchmarkSpine_ToProbeEvent_TokenOnly(b *testing.B) {
	// emitProbeToken — the common per-token event. One output struct + the
	// &probe.Token{} child; no slices, no maps.
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Step:  6,
		Token: &metal.ProbeToken{ID: 42, Text: "tok", PromptTokens: 12, GeneratedTokens: 3},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeEvent = toProbeEvent(event)
	}
}

func BenchmarkSpine_ToProbeEvent_LogitsOnly(b *testing.B) {
	// emitProbeLogits — the heavy per-token event: Shape clone, Top slab,
	// Values clone, and the inner Meta map. topK defaults to 8 in the engine.
	top := make([]metal.ProbeLogit, 8)
	values := make([]float32, 8)
	for i := range top {
		top[i] = metal.ProbeLogit{TokenID: int32(i), Logit: float32(i) * 0.1, Probability: float64(i) * 0.001}
		values[i] = float32(i) * 0.1
	}
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Step:  6,
		Logits: &metal.ProbeLogits{
			Shape:      []int32{1, 16},
			VocabSize:  16,
			MaxTokenID: 4,
			MaxLogit:   1.5,
			MinTokenID: 5,
			MinLogit:   -1.5,
			MeanLogit:  0.25,
			Top:        top,
			Values:     values,
			Meta:       map[string]string{"cpu_transfer": "compact_topk"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeEvent = toProbeEvent(event)
	}
}

func BenchmarkSpine_ToProbeEvent_EntropyOnly(b *testing.B) {
	// emitProbeLogits emits this alongside the Logits event — scalar-only
	// payload, so the floor is the output struct + the &probe.Entropy child.
	event := metal.ProbeEvent{
		Kind:    metal.ProbeEventEntropy,
		Phase:   metal.ProbePhaseDecode,
		Step:    6,
		Entropy: &metal.ProbeEntropy{Value: 0.4, Unit: "nats"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeEvent = toProbeEvent(event)
	}
}

func BenchmarkSpine_ToProbeEvent_CacheOnly(b *testing.B) {
	// emitProbeCachePressure — periodic per-step scalar payload.
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventCachePressure,
		Phase: metal.ProbePhaseDecode,
		Step:  6,
		Cache: &metal.ProbeCachePressure{PromptTokens: 10, GeneratedTokens: 2, LayerCount: 6, CacheTokens: 12, ProcessedTokens: 14, MaxCacheTokens: 20, Utilization: 0.6, Rotating: true},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeEvent = toProbeEvent(event)
	}
}

// --- cloneProbeMeta — fires for event.Meta and logits.Meta per event ---

func BenchmarkSpine_CloneProbeMeta_Empty(b *testing.B) {
	var meta map[string]string
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeMeta = cloneProbeMeta(meta)
	}
}

func BenchmarkSpine_CloneProbeMeta_Single(b *testing.B) {
	meta := map[string]string{"cpu_transfer": "compact_topk"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spineBenchSinkProbeMeta = cloneProbeMeta(meta)
	}
}

var spineBenchSinkProbeMeta map[string]string
