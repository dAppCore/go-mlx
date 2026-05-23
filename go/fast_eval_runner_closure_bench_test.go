// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the modelDecodeGenerator pool path in fast_eval_runner.go.
// Per AX-11 — production dispatch (modelBenchSpeculativeDecode +
// modelBenchPromptLookupDecode + speculative.GenerateSpeculative) acquires
// a *modelDecodeGenerator from the pool, configures (model, base), passes
// it as the decode.Generator interface, and releases on defer. The pre-
// W11-M shape was a closure-returning helper (one heap-allocated closure
// per call); the W11-M shape replaces the closure with a pool Get/Put on
// a struct that implements decode.Generator directly.
//
// These benches measure the construction surface — they pass a zero-value
// *Model so Generate short-circuits on the nil-model guard if invoked.
// Bench names retain the `_Construct` / `_ConstructAndInvoke` /
// `_SpeculativePairConstruct` suffixes so the W11-L baseline numbers from
// /tmp/wave11-W11M-baseline.txt diff cleanly against the W11-M result.
//
// Run: go test -bench='BenchmarkFastEvalRunner_(BenchModelDecodeGenerate|ModelDecodeGenerate)' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference/decode"
)

// Sinks defeat compiler DCE for the bench loops.
var (
	fastEvalRunnerBenchSinkGenerator decode.Generator
	fastEvalRunnerBenchSinkGen       decode.Generation
	fastEvalRunnerBenchSinkErr       error
)

// --- modelDecodeGenerator — pool acquire/release allocs ---
//
// Each iteration acquires a generator from the pool, parks the
// (model, base) pair, and releases. Once the pool is warm the per-call
// alloc count drops to zero — the previous closure-returning shape paid
// one heap alloc per call to materialise the closure object.

func BenchmarkFastEvalRunner_ModelDecodeGenerate_Construct(b *testing.B) {
	model := &Model{}
	base := DefaultGenerateConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := acquireModelDecodeGenerator(model, &base)
		fastEvalRunnerBenchSinkGenerator = g
		releaseModelDecodeGenerator(g)
	}
}

// benchModelDecodeGenerate constructs a fresh *modelDecodeGenerator each
// call (it owns its own base config) — kept benched separately so the
// non-pooled test entry point's cost stays visible alongside the pooled
// hot path.
func BenchmarkFastEvalRunner_BenchModelDecodeGenerate_Construct(b *testing.B) {
	model := &Model{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalRunnerBenchSinkGenerator = benchModelDecodeGenerate(model)
	}
}

// --- modelDecodeGenerator — invocation guard cost (no real model required) ---
//
// Generate short-circuits on `g.model == nil || g.model.model == nil`.
// Pairing acquire + Generate + release per iteration mirrors the shape
// decode.PromptLookup drives — one generator dispatch per call.

func BenchmarkFastEvalRunner_ModelDecodeGenerate_ConstructAndInvoke(b *testing.B) {
	model := &Model{}
	base := DefaultGenerateConfig()
	ctx := context.Background()
	cfg := decode.GenerateConfig{MaxTokens: 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := acquireModelDecodeGenerator(model, &base)
		fastEvalRunnerBenchSinkGen, fastEvalRunnerBenchSinkErr = g.Generate(ctx, "prompt", cfg)
		releaseModelDecodeGenerator(g)
	}
}

// --- speculative.GenerateSpeculative shape — two pooled generators sharing one base ---
//
// Mirrors the production target+draft pattern speculative.go drives on
// every Model.GenerateSpeculative entry. The pre-W11-M shape paid two
// closure allocs per dispatch (target + draft); the pool shape pays zero
// steady-state — both legs acquire from the same sync.Pool and release
// on defer.

func BenchmarkFastEvalRunner_ModelDecodeGenerate_SpeculativePairConstruct(b *testing.B) {
	target := &Model{}
	draft := &Model{}
	base := DefaultGenerateConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := acquireModelDecodeGenerator(target, &base)
		d := acquireModelDecodeGenerator(draft, &base)
		fastEvalRunnerBenchSinkGenerator = t
		fastEvalRunnerBenchSinkGenerator = d
		releaseModelDecodeGenerator(d)
		releaseModelDecodeGenerator(t)
	}
}
