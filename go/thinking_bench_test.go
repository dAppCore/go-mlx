// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the root-package thinking-mode GenerateOption
// builders + parserHint. Per AX-11 — every Generate / Chat call
// constructs a fresh GenerateConfig by applying the option chain;
// the With* builders fire on every dispatch. parserHint also fires
// per dispatch inside FilterThinkingTokens + every wire handler
// that resolves the architecture-specific reasoning parser.
//
// FilterThinkingTokens itself takes a *Tokenizer (Metal-backed) and
// is excluded — its CPU path is covered by the parser bench tree.
//
// Run:    go test -bench='BenchmarkThinking' -benchtime=100ms -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/lora"
)

// Sinks defeat compiler DCE.
var (
	thinkingBenchSinkOption GenerateOption
	thinkingBenchSinkConfig GenerateConfig
	thinkingBenchSinkHint   parser.Hint
)

// --- Single-option builders — pure closure constructors ---

func BenchmarkThinking_WithThinkingMode(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkOption = WithThinkingMode(parser.Capture)
	}
}

func BenchmarkThinking_WithShowThinking(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkOption = WithShowThinking()
	}
}

func BenchmarkThinking_WithHideThinking(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkOption = WithHideThinking()
	}
}

func BenchmarkThinking_WithCaptureThinking(b *testing.B) {
	capture := func(parser.Chunk) {}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkOption = WithCaptureThinking(capture)
	}
}

func BenchmarkThinking_WithThinkingCapture_Alias(b *testing.B) {
	capture := func(parser.Chunk) {}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkOption = WithThinkingCapture(capture)
	}
}

// --- Option application — measures what callers actually pay
// per Generate call: build the option, then apply to a fresh
// config. Mirrors the inner loop of `ApplyGenerateOpts`. ---

func BenchmarkThinking_ApplyShowThinking(b *testing.B) {
	option := WithShowThinking()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := DefaultGenerateConfig()
		option(&cfg)
		thinkingBenchSinkConfig = cfg
	}
}

func BenchmarkThinking_ApplyHideThinking(b *testing.B) {
	option := WithHideThinking()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := DefaultGenerateConfig()
		option(&cfg)
		thinkingBenchSinkConfig = cfg
	}
}

func BenchmarkThinking_ApplyCaptureThinking(b *testing.B) {
	option := WithCaptureThinking(func(parser.Chunk) {})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := DefaultGenerateConfig()
		option(&cfg)
		thinkingBenchSinkConfig = cfg
	}
}

// --- parserHint — fires per FilterThinkingTokens call + per wire
// dispatch when the parser needs to pick reasoning markers. ---

func BenchmarkThinking_ParserHint_QwenBare(b *testing.B) {
	info := ModelInfo{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkHint = parserHint(info)
	}
}

func BenchmarkThinking_ParserHint_QwenWithAdapter(b *testing.B) {
	info := ModelInfo{
		Architecture: "qwen3",
		Adapter:      lora.AdapterInfo{Name: "probe-lora", Path: "/models/lora/probe", Rank: 16, Alpha: 32},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkHint = parserHint(info)
	}
}

func BenchmarkThinking_ParserHint_Gemma4(b *testing.B) {
	info := ModelInfo{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchSinkHint = parserHint(info)
	}
}
