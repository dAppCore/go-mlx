// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for backend.go dispatch helpers — toMetalGenerateConfig and
// toMetalProbeSink. Per AX-11 — both fire on every Generate / Chat /
// Classify / BatchGenerate call, so the per-call allocation budget for
// the inference hot path runs through here.
//
// Run:    go test -bench='BenchmarkBackend_ToMetal' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE.
var (
	backendBenchSinkMetalCfg      metal.GenerateConfig
	backendBenchSinkMetalSink     metal.ProbeSink
	backendBenchSinkHint          parser.Hint
	backendBenchSinkProbeLogits   []probe.Logit
	backendBenchSinkProbeEvent    probe.Event
	backendBenchSinkRootMetrics   Metrics
	backendBenchSinkRootToken     Token
	backendBenchSinkRootAdapter   lora.AdapterInfo
	backendBenchSinkChatMessages  []metal.ChatMessage
)

// noopProbeSink is a minimal probe.Sink that drops every event — used by
// the toMetalProbeSink benchmark to exercise the non-nil dispatch path
// without paying for downstream event-conversion work.
type noopProbeSink struct{}

// EmitProbe drops the event.
func (noopProbeSink) EmitProbe(probe.Event) {}

// --- toMetalGenerateConfig ---
// Per-call shuffler from the root GenerateConfig into the metal package
// equivalent. Inlined into every Generate / Chat / Classify entry — the
// per-call allocation pattern here drives the dispatch-side budget.

func BenchmarkBackend_ToMetalGenerateConfig_NoSink(b *testing.B) {
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
		backendBenchSinkMetalCfg = toMetalGenerateConfig(cfg)
	}
}

func BenchmarkBackend_ToMetalGenerateConfig_WithSink(b *testing.B) {
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
		backendBenchSinkMetalCfg = toMetalGenerateConfig(cfg)
	}
}

// --- toMetalProbeSink ---
// Per-call closure/adapter allocator. Fires once per Generate / Chat /
// Classify entry. The nil-sink path is the steady-state (most calls
// don't request probes); the non-nil path is the trace hot path.

func BenchmarkBackend_ToMetalProbeSink_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkMetalSink = toMetalProbeSink(nil)
	}
}

func BenchmarkBackend_ToMetalProbeSink_NonNil(b *testing.B) {
	sink := noopProbeSink{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkMetalSink = toMetalProbeSink(sink)
	}
}

// --- hintForParser cache (Wave6-W1A) ---
// Per-Generate parser.Hint dispatch — pre-cached at LoadModel + on LoRA
// mutation; the cached read is the hot-path replacement for the prior
// per-call m.model.Info() fan-out (which itself cloned the native
// AdapterInfo.TargetKeys slice).

func BenchmarkBackend_HintForParser_Cached(b *testing.B) {
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "qwen3",
				Adapter:      metal.AdapterInfo{Name: "probe-lora"},
			},
		},
		adapterInfo: lora.AdapterInfo{Name: "probe-lora"},
	}
	// Warm the cache so we measure the steady-state read, not the
	// one-time lazy build.
	model.refreshParserHint()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkHint = model.hintForParser()
	}
}

func BenchmarkBackend_HintForParser_Build(b *testing.B) {
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "qwen3",
				Adapter:      metal.AdapterInfo{Name: "probe-lora"},
			},
		},
		adapterInfo: lora.AdapterInfo{Name: "probe-lora"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkHint = model.buildParserHint()
	}
}

// --- toRootProbeLogits (W10-AN) ---
// Per-probe-event slice clone — metal.ProbeLogit and probe.Logit have
// bit-identical layout (int32 + float32 + float64). Top-K is commonly
// 50-100 entries per probe.Logits, emitted per-token when ProbeSink is
// enabled. Benches the empty / typical / large fan-outs to surface the
// per-element struct unpacking cost vs a direct slab copy.

func BenchmarkBackend_ToRootProbeLogits_Empty(b *testing.B) {
	var logits []metal.ProbeLogit
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkProbeLogits = toRootProbeLogits(logits)
	}
}

func BenchmarkBackend_ToRootProbeLogits_Typical(b *testing.B) {
	logits := make([]metal.ProbeLogit, 50)
	for i := range logits {
		logits[i] = metal.ProbeLogit{TokenID: int32(i), Logit: float32(i) * 0.1, Probability: float64(i) * 0.001}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkProbeLogits = toRootProbeLogits(logits)
	}
}

func BenchmarkBackend_ToRootProbeLogits_Large(b *testing.B) {
	logits := make([]metal.ProbeLogit, 256)
	for i := range logits {
		logits[i] = metal.ProbeLogit{TokenID: int32(i), Logit: float32(i) * 0.1, Probability: float64(i) * 0.001}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkProbeLogits = toRootProbeLogits(logits)
	}
}

// --- toRootToken (W10-AN) ---
// Per-token shuffler used by toRootClassifyResults / toRootBatchResults /
// every *Stream entry. Tiny but fires once per emitted token.

func BenchmarkBackend_ToRootToken(b *testing.B) {
	token := metal.Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootToken = toRootToken(token)
	}
}

// --- toRootAdapterInfo (W10-AN) ---
// Called from toRootMetrics on every Metrics() read AND from
// adapterFromNativeInfo on every Info() read. Clones TargetKeys slice.

func BenchmarkBackend_ToRootAdapterInfo_Empty(b *testing.B) {
	info := metal.AdapterInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootAdapter = toRootAdapterInfo(info)
	}
}

func BenchmarkBackend_ToRootAdapterInfo_Typical(b *testing.B) {
	info := metal.AdapterInfo{
		Name:       "probe-lora",
		Path:       "/models/lora.safetensors",
		Hash:       "sha256:abc",
		Rank:       16,
		Alpha:      32.0,
		Scale:      2.0,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootAdapter = toRootAdapterInfo(info)
	}
}

// --- toRootMetrics (W10-AN) ---
// Per-Metrics() call: field-by-field shuffler. Fires on every read of
// Model.Metrics() — typically once per Generate but call sites vary.

func BenchmarkBackend_ToRootMetrics_Simple(b *testing.B) {
	metrics := metal.Metrics{
		PromptTokens:        128,
		GeneratedTokens:     64,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  100.0,
		Adapter:             metal.AdapterInfo{Name: "probe-lora"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootMetrics = toRootMetrics(metrics)
	}
}

// --- chatMessagesAsMetal (W10-AN) ---
// Per-Chat call shuffler from []inference.Message to []metal.ChatMessage.
// W10-AN replaced a make + per-message copy with a layout-guarded
// unsafe.Slice reinterpret — the bench surfaces the cost going from
// O(N) struct copy + 1 alloc to 0 / 0.

func BenchmarkBackend_ChatMessagesAsMetal_Short(b *testing.B) {
	messages := []inference.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkChatMessages = chatMessagesAsMetal(messages)
	}
}

func BenchmarkBackend_ChatMessagesAsMetal_Long(b *testing.B) {
	messages := make([]inference.Message, 20)
	for i := range messages {
		messages[i] = inference.Message{Role: "user", Content: "turn"}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkChatMessages = chatMessagesAsMetal(messages)
	}
}
