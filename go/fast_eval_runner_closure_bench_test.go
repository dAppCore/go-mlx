// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the modelDecodeGenerate / benchModelDecodeGenerate closure
// construction in fast_eval_runner.go. Per AX-11 — the closure is built once
// per generator per decode.Speculative / decode.PromptLookup call (so twice
// per GenerateSpeculative on the production speculative.go:GenerateSpeculative
// path, and once per fast-eval bench dispatch on
// modelBenchSpeculativeDecode / modelBenchPromptLookupDecode).
//
// These benches measure pure construction — they pass a zero-value *Model so
// the closure body short-circuits on the nil-model guard if invoked. The
// invocation benches deliberately fire the guard once per loop iteration to
// surface the per-call dispatch cost separately from the construction cost.
//
// Run:    go test -bench='BenchmarkFastEvalRunner_(BenchModelDecodeGenerate|ModelDecodeGenerate)' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference/decode"
)

// Sinks defeat compiler DCE for the closure-construction benches.
var (
	fastEvalRunnerBenchSinkFunc decode.GenerateFunc
	fastEvalRunnerBenchSinkGen  decode.Generation
	fastEvalRunnerBenchSinkErr  error
)

// --- modelDecodeGenerate — construction-only allocs ---
//
// Each call returns a decode.GenerateFunc. After the *GenerateConfig
// signature change the closure captures only two 8-byte pointers (model
// + base) and the GenerateConfig is owned by the caller, so the inner
// closure object is the only per-call alloc. The bench fixture below
// pre-allocates `base` on the stack so its escape (forced by the closure
// capturing it via &base) is hoisted outside the benchmark loop — this
// is the shape speculative.go:GenerateSpeculative drives where one
// generateCfg is shared by both target + draft modelDecodeGenerate calls.

func BenchmarkFastEvalRunner_ModelDecodeGenerate_Construct(b *testing.B) {
	model := &Model{}
	base := DefaultGenerateConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalRunnerBenchSinkFunc = modelDecodeGenerate(model, &base)
	}
}

// benchModelDecodeGenerate still constructs its own GenerateConfig
// locally; that local escapes to heap because the inner closure captures
// its address. The bench retains both surfaces so a future further win
// (e.g. taking a *GenerateConfig argument here too) shows up cleanly
// against this baseline.
func BenchmarkFastEvalRunner_BenchModelDecodeGenerate_Construct(b *testing.B) {
	model := &Model{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalRunnerBenchSinkFunc = benchModelDecodeGenerate(model)
	}
}

// --- modelDecodeGenerate — invocation guard cost (no real model required) ---
//
// The closure body short-circuits on `model == nil || model.model == nil`.
// Pairing construct + invoke per iteration mirrors the shape decode.Speculative
// drives — it calls the GenerateFunc once per (target | draft) per Speculative
// pass. The bench captures the combined cost the production hot path pays
// when speculative decode is dispatched repeatedly.

func BenchmarkFastEvalRunner_ModelDecodeGenerate_ConstructAndInvoke(b *testing.B) {
	model := &Model{}
	base := DefaultGenerateConfig()
	ctx := context.Background()
	cfg := decode.GenerateConfig{MaxTokens: 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalRunnerBenchSinkFunc = modelDecodeGenerate(model, &base)
		fastEvalRunnerBenchSinkGen, fastEvalRunnerBenchSinkErr = fastEvalRunnerBenchSinkFunc(ctx, "prompt", cfg)
	}
}

// --- speculative.GenerateSpeculative shape — two closures sharing one base ---
//
// Mirrors the production target+draft pair pattern that speculative.go drives
// on every Model.GenerateSpeculative entry. Previously each leg paid its own
// heap-spilled `base` copy (2 closures × 2 allocs = 4). After the *GenerateConfig
// signature change, both legs share one heap-resident base, so the pair pays
// 2 closure allocs + a sink reference for the second closure value (the first
// is overwritten by the second when both go through one sink).

func BenchmarkFastEvalRunner_ModelDecodeGenerate_SpeculativePairConstruct(b *testing.B) {
	target := &Model{}
	draft := &Model{}
	base := DefaultGenerateConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalRunnerBenchSinkFunc = modelDecodeGenerate(target, &base)
		fastEvalRunnerBenchSinkFunc = modelDecodeGenerate(draft, &base)
	}
}
