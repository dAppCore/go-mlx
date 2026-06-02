// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/memory"
)

var (
	benchmarkProductionQuantizationChoice ProductionQuantizationChoice
	benchmarkProductionQuantizationPack   ProductionQuantizationPackSupport
	benchmarkProductionArchitectureStatus ProductionArchitectureStatusReport
	benchmarkProductionMTPDecision        ProductionMTPPromotionDecision
	benchmarkProductionTurboDecision      ProductionTurboQuantPromotionDecision
	benchmarkProductionCombinedDecision   ProductionCombinedMTPAndTurboQuantDecision
)

func BenchmarkSelectProductionQuantizationTier_DefaultQ6(b *testing.B) {
	input := ProductionQuantizationSelectionInput{
		Device:        memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 90 * memory.GiB},
		ContextLength: ProductionLaneLongContextLength,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkProductionQuantizationChoice = SelectProductionQuantizationTier(input)
	}
}

func BenchmarkProductionQuantizationPackByName_MXFP8(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pack, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-e2b-it-mxfp8")
		if !ok {
			b.Fatal("missing mxfp8 pack")
		}
		benchmarkProductionQuantizationPack = pack
	}
}

func BenchmarkDefaultProductionArchitectureStatus(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkProductionArchitectureStatus = DefaultProductionArchitectureStatus()
	}
}

func BenchmarkEvaluateProductionMTPPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionMTPPolicy()
	evidence := productionCombinedMTPPassEvidence(memory.KVCacheModePaged)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkProductionMTPDecision = EvaluateProductionMTPPromotion(policy, evidence)
	}
}

func BenchmarkEvaluateProductionTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionTurboQuantPolicy()
	evidence := productionCombinedTurboQuantPassEvidence()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkProductionTurboDecision = EvaluateProductionTurboQuantPromotion(policy, evidence)
	}
}

func BenchmarkEvaluateProductionCombinedMTPAndTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	mtpEvidence := productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant)
	turboEvidence := productionCombinedTurboQuantPassEvidence()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkProductionCombinedDecision = EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence)
	}
}
