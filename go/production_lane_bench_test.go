// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for production-lane descriptor builders. Per AX-11 — the
// DefaultProductionLane + DefaultGemma4FastRuntimeGates helpers are queried
// per dispatch by the agentic driver. Context length must not select a
// different gate family. Slice-bearing defaults return defensive copies, so
// the allocation cost stays visible here instead of leaking into assumptions.
//
// Run:    go test -bench='BenchmarkProdLane' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/memory"
)

// Sinks defeat compiler DCE. Distinct names from root_bench_test.go +
// adapter_bench_test.go to avoid collisions in package mlx.
var (
	prodLaneBenchSinkPlan           ProductionLane
	prodLaneBenchSinkGates          []string
	prodLaneBenchSinkQuantPolicy    ProductionQuantizationPolicy
	prodLaneBenchSinkQuantChoice    ProductionQuantizationChoice
	prodLaneBenchSinkMTPPolicy      ProductionMTPPolicy
	prodLaneBenchSinkTurboPolicy    ProductionTurboQuantPolicy
	prodLaneBenchSinkCombinedPolicy ProductionCombinedMTPAndTurboQuantPolicy
	prodLaneBenchSinkMTPDecision    ProductionMTPPromotionDecision
	prodLaneBenchSinkTurboDecision  ProductionTurboQuantPromotionDecision
	prodLaneBenchSinkComboDecision  ProductionCombinedMTPAndTurboQuantDecision
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

func BenchmarkProdLane_SelectProductionQuantizationTier_DefaultQ6(b *testing.B) {
	input := ProductionQuantizationSelectionInput{
		Device: memory.DeviceInfo{
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		ContextLength: ProductionLaneLongContextLength,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkQuantChoice = SelectProductionQuantizationTier(input)
	}
}

func BenchmarkProdLane_SelectProductionQuantizationTier_QualityUnknownHeadroom(b *testing.B) {
	input := ProductionQuantizationSelectionInput{QualityFirst: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkQuantChoice = SelectProductionQuantizationTier(input)
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

func BenchmarkProdLane_EvaluateProductionMTPPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionMTPPolicy()
	evidence := productionCombinedMTPPassEvidence(memory.KVCacheModePaged)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkMTPDecision = EvaluateProductionMTPPromotion(policy, evidence)
	}
}

func BenchmarkProdLane_EvaluateProductionTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionTurboQuantPolicy()
	evidence := productionCombinedTurboQuantPassEvidence()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkTurboDecision = EvaluateProductionTurboQuantPromotion(policy, evidence)
	}
}

func BenchmarkProdLane_EvaluateProductionCombinedMTPAndTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	mtpEvidence := productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant)
	turboEvidence := productionCombinedTurboQuantPassEvidence()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prodLaneBenchSinkComboDecision = EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence)
	}
}
