// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for production-lane descriptor builders. Per AX-11 — the
// DefaultProductionLane + Gemma4FastRuntimeGates* helpers are queried
// per dispatch by the agentic driver to determine what gates apply to
// the requested context length. The cost is dominated by the per-call
// slice clone (defensive copy) — important to know because some
// callers query these on every prompt, not just at boot.
//
// Run:    go test -bench='BenchmarkProdLane' -benchmem -run='^$' ./go

package mlx

import "testing"

// Sinks defeat compiler DCE. Distinct names from root_bench_test.go +
// adapter_bench_test.go to avoid collisions in package mlx.
var (
	prodLaneBenchSinkPlan  ProductionLane
	prodLaneBenchSinkGates []string
)

// --- DefaultProductionLane — fires per dispatch to seed the request shape ---

func BenchmarkProdLane_DefaultProductionLane(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkPlan = DefaultProductionLane()
	}
}

// --- DefaultGemma4FastRuntimeGates — defensive slice clone over the
// 10-element gate set. Hit on every dispatch decision.

func BenchmarkProdLane_DefaultGemma4FastRuntimeGates(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = DefaultGemma4FastRuntimeGates()
	}
}

// --- LongContextGemma4FastRuntimeGates — single-element clone. Same
// pattern as DefaultGemma4FastRuntimeGates but smaller; documents the
// minimum-cost defensive-copy floor for this helper family.

func BenchmarkProdLane_LongContextGemma4FastRuntimeGates(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = LongContextGemma4FastRuntimeGates()
	}
}

// --- Gemma4FastRuntimeGatesForContext — context-conditional gate
// resolver. Short-circuit branch (<= LongFormContextLength) returns
// the default set as-is; the long-context branch rewrites the slice
// by stripping fixed-cache gates and appending the paged-decode gate.

func BenchmarkProdLane_GatesForContext_DefaultBranch(b *testing.B) {
	// 4096 is the production driver default — hits the short-circuit
	// path that just returns DefaultGemma4FastRuntimeGates() verbatim.
	contextLength := ProductionLaneContextLength
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = Gemma4FastRuntimeGatesForContext(contextLength)
	}
}

func BenchmarkProdLane_GatesForContext_LongFormBranch(b *testing.B) {
	// 65536 — exactly at the LongFormContextLength boundary, still
	// short-circuit.
	contextLength := ProductionLaneLongFormContextLength
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = Gemma4FastRuntimeGatesForContext(contextLength)
	}
}

func BenchmarkProdLane_GatesForContext_HyperLongBranch(b *testing.B) {
	// 131072 — above LongFormContextLength so the helper rewrites
	// the slice (drop fixed-cache gates, append paged-decode gate).
	contextLength := ProductionLaneHyperLongContextLength
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = Gemma4FastRuntimeGatesForContext(contextLength)
	}
}
