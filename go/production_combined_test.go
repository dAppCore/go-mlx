// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
)

func TestProductionCombinedMTPAndTurboQuantPolicy_Defaults_Good(t *testing.T) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()

	if policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID || policy.AssistantModelID != OfficialGemma4E2BAssistantLock().ModelID {
		t.Fatalf("policy identity = %+v, want official target+assistant IDs", policy)
	}
	if policy.Mode != ProductionCombinedMTPAndTurboQuantMode || policy.CacheMode != memory.KVCacheModeTurboQuant {
		t.Fatalf("policy mode = %q cache=%q, want combined MTP+TurboQuant lane", policy.Mode, policy.CacheMode)
	}
	if policy.EnabledByDefault || !policy.RequiresExplicitOptIn {
		t.Fatalf("policy default = enabled:%v explicit:%v, want explicit opt-in only", policy.EnabledByDefault, policy.RequiresExplicitOptIn)
	}
	if !policy.RequiresMTPPromotion || !policy.RequiresTurboQuantPromotion || !policy.RequiresGreedyParity || !policy.RequiresTurboQuantQualityParity {
		t.Fatalf("policy requirements = %+v, want both component promotion gates plus quality/parity checks", policy)
	}
	for _, metric := range []string{
		"mtp_visible_tokens_per_sec",
		"mtp_target_tokens_per_sec",
		"mtp_warm_decode_tokens_per_sec",
		"mtp_wall_duration",
		"mtp_restore_duration",
		"mtp_active_plus_cache_memory_bytes",
		"mtp_energy_joules",
		"mtp_draft_token_schedule",
		"mtp_observed_draft_token_sweeps",
		"mtp_accepted_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"assistant_token_ordering_dtype",
		"assistant_token_ordering_shape",
		"estimated_power_watts",
		"turboquant_baseline_cache_mode",
		"turboquant_quality_flags",
		"turboquant_normal_context_validated",
		"turboquant_stress_context_validated",
		"turboquant_candidate_layout_version",
		"turboquant_candidate_qjl_residual",
		"turboquant_candidate_metadata_bytes",
		"turboquant_baseline_visible_tokens_per_sec",
		"turboquant_candidate_visible_tokens_per_sec",
		"turboquant_baseline_input_output_tokens_per_sec",
		"turboquant_candidate_input_output_tokens_per_sec",
		"turboquant_baseline_wall_duration",
		"turboquant_candidate_wall_duration",
		"turboquant_baseline_restore_duration",
		"turboquant_candidate_restore_duration",
		"turboquant_baseline_active_plus_cache_memory_bytes",
		"turboquant_candidate_active_plus_cache_memory_bytes",
		"turboquant_baseline_energy_joules",
		"turboquant_candidate_energy_joules",
		"quality_flags",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_AllowsOptInCandidate_Good(t *testing.T) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant),
		productionCombinedTurboQuantPassEvidence(),
	)

	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v, want combined production candidate after both lanes pass", decision)
	}
	if decision.EnableByDefault {
		t.Fatalf("EnableByDefault = true, want combined lane to remain explicit opt-in")
	}
	if !decision.MTPEligible || !decision.TurboQuantEligible {
		t.Fatalf("component eligibility = mtp:%v turbo:%v, want both true", decision.MTPEligible, decision.TurboQuantEligible)
	}
	if decision.MTPAcceptanceRate <= 0 || decision.TurboQuantMemorySavingsRatio <= 0 {
		t.Fatalf("decision metrics = %+v, want assistant acceptance and TurboQuant memory savings recorded", decision)
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_RejectsNonTurboQuantMTP_Bad(t *testing.T) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassEvidence(memory.KVCacheModePaged),
		productionCombinedTurboQuantPassEvidence(),
	)

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "must run target-only and MTP with TurboQuant") {
		t.Fatalf("decision = %+v, want combined lane to require TurboQuant cache mode in MTP comparison", decision)
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_RejectsAssistantAcceptanceRegression_Bad(t *testing.T) {
	mtpEvidence := productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant)
	mtpEvidence.MTPAcceptedTokens = 0
	mtpEvidence.MTPRejectedTokens = mtpEvidence.MTPProposedTokens

	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		mtpEvidence,
		productionCombinedTurboQuantPassEvidence(),
	)

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "MTP must pass") || !core.Contains(decision.Reason, "accepted draft tokens") {
		t.Fatalf("decision = %+v, want assistant acceptance gate", decision)
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_RejectsTurboQuantQualityLoss_Bad(t *testing.T) {
	turboEvidence := productionCombinedTurboQuantPassEvidence()
	turboEvidence.QualityFlags = []string{"long_output_drift"}

	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant),
		turboEvidence,
	)

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "TurboQuant must pass") || !core.Contains(decision.Reason, "quality flags") {
		t.Fatalf("decision = %+v, want TurboQuant quality gate", decision)
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_AllocFree_Good(t *testing.T) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	mtpEvidence := productionCombinedMTPPassEvidence(memory.KVCacheModeTurboQuant)
	turboEvidence := productionCombinedTurboQuantPassEvidence()

	allocs := testing.AllocsPerRun(100, func() {
		decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence)
		if !decision.ProductionCandidate {
			t.Fatalf("decision = %+v, want combined production candidate", decision)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs/op = %.0f, want zero for policy hot-path evaluation", allocs)
	}
}

func productionCombinedMTPPassEvidence(cacheMode memory.KVCacheMode) ProductionMTPPromotionEvidence {
	return ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                ProductionMTPPromotionMinRetainedTurns,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyFirstTokenDuration:         120 * time.Millisecond,
		MTPFirstTokenDuration:                90 * time.Millisecond,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SameLoadPolicy:                       true,
		TargetOnlyCachePolicy:                "full",
		MTPCachePolicy:                       "full",
		TargetOnlyCacheMode:                  string(cacheMode),
		MTPCacheMode:                         string(cacheMode),
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               ProductionMTPDefaultDraftTokens,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		AssistantFourLayerDrafter:            true,
		AssistantTokenOrderingDType:          "int64",
		AssistantTokenOrderingShape:          []int{2048, 128},
		OfficialPairVerified:                 true,
		OfficialTargetModelID:                OfficialGemma4E2BTargetLock().ModelID,
		OfficialTargetRevision:               OfficialGemma4E2BTargetLock().Revision,
		OfficialAssistantModelID:             OfficialGemma4E2BAssistantLock().ModelID,
		OfficialAssistantRevision:            OfficialGemma4E2BAssistantLock().Revision,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	}
}

func productionCombinedTurboQuantPassEvidence() ProductionTurboQuantPromotionEvidence {
	policy := DefaultProductionTurboQuantPolicy()
	return ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		CandidateLayoutVersion:              policy.RequiredLayoutVersion,
		CandidateKeyAlgorithm:               policy.RequiredKeyAlgorithm,
		CandidateValueAlgorithm:             policy.RequiredValueAlgorithm,
		CandidateOutlierPolicy:              policy.RequiredOutlierPolicy,
		CandidateEffectiveBitsMilli:         policy.TargetEffectiveBitsMilli,
		CandidateQJLResidual:                true,
		CandidateMetadataBytes:              64 * 1024,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        120,
		BaselineInputOutputTokensPerSec:     33000,
		CandidateInputOutputTokensPerSec:    36000,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
	}
}
