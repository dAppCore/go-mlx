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

	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE.
var (
	backendBenchSinkMetalCfg  metal.GenerateConfig
	backendBenchSinkMetalSink metal.ProbeSink
	backendBenchSinkHint      parser.Hint
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
