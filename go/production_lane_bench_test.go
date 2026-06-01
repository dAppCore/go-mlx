// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for production-lane descriptor builders. Per AX-11 — the
// DefaultProductionLane + DefaultGemma4FastRuntimeGates helpers are queried
// per dispatch by the agentic driver. Context length must not select a
// different gate family. The cost is dominated by the per-call shared
// read-only gate slice — important to know because some callers query these
// on every prompt, not just at boot.
//
// Run:    go test -bench='BenchmarkProdLane' -benchmem -run='^$' ./go

package mlx

import "testing"

// Sinks defeat compiler DCE. Distinct names from root_bench_test.go +
// adapter_bench_test.go to avoid collisions in package mlx.
var (
	prodLaneBenchSinkPlan           ProductionLane
	prodLaneBenchSinkGates          []string
	prodLaneBenchSinkQuantPolicy    ProductionQuantizationPolicy
	prodLaneBenchSinkMTPPolicy      ProductionMTPPolicy
	prodLaneBenchSinkTurboPolicy    ProductionTurboQuantPolicy
	prodLaneBenchSinkCombinedPolicy ProductionCombinedMTPAndTurboQuantPolicy
)

// --- DefaultProductionLane — fires per dispatch to seed the request shape ---

func BenchmarkProdLane_DefaultProductionLane(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkPlan = DefaultProductionLane()
	}
}

// --- DefaultGemma4FastRuntimeGates — read-only gate set. Hit on every
// dispatch decision.

func BenchmarkProdLane_DefaultGemma4FastRuntimeGates(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkGates = DefaultGemma4FastRuntimeGates()
	}
}

func BenchmarkProdLane_DefaultProductionQuantizationPolicy(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkQuantPolicy = DefaultProductionQuantizationPolicy()
	}
}

func BenchmarkProdLane_DefaultProductionMTPPolicy(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkMTPPolicy = DefaultProductionMTPPolicy()
	}
}

func BenchmarkProdLane_DefaultProductionTurboQuantPolicy(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkTurboPolicy = DefaultProductionTurboQuantPolicy()
	}
}

func BenchmarkProdLane_DefaultProductionCombinedMTPAndTurboQuantPolicy(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkCombinedPolicy = DefaultProductionCombinedMTPAndTurboQuantPolicy()
	}
}
