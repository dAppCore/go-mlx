// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for state_chapter_smoke.go — the runner-build path that
// pre-folds GenerateConfig into GenerateOption closures. Per AX-11 —
// stateKVChapterGenerateOptions fires once per chapter-smoke runner
// (one runner per RunModelStateKVChapterSmoke invocation), and each
// of its closures fires once per chapter Generate call (often dozens
// per smoke session over a long-corpus harness).
//
// Run:    go test -bench='BenchmarkStateChapterSmoke' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/chaptersmoke"
	"dappco.re/go/mlx/probe"
)

var chapterSmokeConfigSnapshot = chaptersmoke.Config{
	AnswerMaxTokens: 64,
	Temperature:     0.7,
}

// Sinks defeat compiler DCE.
var (
	stateChapterSmokeBenchSinkOpts []GenerateOption
	stateChapterSmokeBenchSinkCfg  GenerateConfig
)

type stateChapterSmokeStubSink struct{}

func (stateChapterSmokeStubSink) EmitProbe(_ probe.Event) {}

// --- stateKVChapterGenerateOptions: minimum config (always-on fields only) ---

func BenchmarkStateChapterSmoke_GenerateOptions_Minimum(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.7,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stateChapterSmokeBenchSinkOpts = stateKVChapterGenerateOptions(cfg)
	}
}

// --- stateKVChapterGenerateOptions: typical chapter sampling profile ---

func BenchmarkStateChapterSmoke_GenerateOptions_Typical(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2, 0},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stateChapterSmokeBenchSinkOpts = stateKVChapterGenerateOptions(cfg)
	}
}

// --- stateKVChapterGenerateOptions: full sampling profile (all 8 fields) ---

func BenchmarkStateChapterSmoke_GenerateOptions_Full(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2, 0},
		ProbeSink:     stateChapterSmokeStubSink{},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stateChapterSmokeBenchSinkOpts = stateKVChapterGenerateOptions(cfg)
	}
}

// --- Applied: the chapter-runner consumer pattern ---

func BenchmarkStateChapterSmoke_Apply_Typical(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2, 0},
	}
	opts := stateKVChapterGenerateOptions(cfg)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stateChapterSmokeBenchSinkCfg = applyGenerateOptions(opts)
	}
}

// --- chapterGenerateConfig: the cfg→GenerateConfig narrow projection ---

func BenchmarkStateChapterSmoke_ChapterGenerateConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stateChapterSmokeBenchSinkCfg = chapterGenerateConfig(chapterSmokeConfigSnapshot)
	}
}
