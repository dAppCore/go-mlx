// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"time"

	"dappco.re/go/mlx/memory"
)

// ProductionTurboQuantPolicy describes the evidence required before the
// explicit TurboQuant KV-cache mode can move from research lane to production
// candidate. It remains non-default even after promotion.
type ProductionTurboQuantPolicy struct {
	TargetModelID                   string               `json:"target_model_id"`
	CacheMode                       memory.KVCacheMode   `json:"cache_mode"`
	Mode                            string               `json:"mode"`
	TargetEffectiveBitsMilli        int                  `json:"target_effective_bits_milli"`
	EnabledByDefault                bool                 `json:"enabled_by_default"`
	RequiresExplicitOptIn           bool                 `json:"requires_explicit_opt_in"`
	RequiresRetainedWorkflow        bool                 `json:"requires_retained_workflow"`
	RequiresQualityParity           bool                 `json:"requires_quality_parity"`
	RequiresSideBySideBenchmark     bool                 `json:"requires_side_by_side_benchmark"`
	RequiresNormalContextValidation bool                 `json:"requires_normal_context_validation"`
	RequiresStressContextValidation bool                 `json:"requires_stress_context_validation"`
	MinimumRetainedTurns            int                  `json:"minimum_retained_turns"`
	NormalContextLength             int                  `json:"normal_context_length"`
	StressContextLength             int                  `json:"stress_context_length"`
	CompareAgainstCacheModes        []memory.KVCacheMode `json:"compare_against_cache_modes"`
	RequiredMetrics                 []string             `json:"required_metrics"`
}

// ProductionTurboQuantPromotionEvidence is the measured retained-workflow row
// used to decide whether TurboQuant can be treated as a production candidate.
type ProductionTurboQuantPromotionEvidence struct {
	RetainedWorkflow                    bool                 `json:"retained_workflow"`
	Turns                               int                  `json:"turns"`
	QualityMatches                      bool                 `json:"quality_matches"`
	QualityFlags                        []string             `json:"quality_flags,omitempty"`
	BaselineCacheMode                   memory.KVCacheMode   `json:"baseline_cache_mode"`
	CandidateCacheMode                  memory.KVCacheMode   `json:"candidate_cache_mode"`
	SameLoadPolicy                      bool                 `json:"same_load_policy"`
	BaselineCachePolicy                 string               `json:"baseline_cache_policy"`
	CandidateCachePolicy                string               `json:"candidate_cache_policy"`
	BaselineContextLength               int                  `json:"baseline_context_length"`
	CandidateContextLength              int                  `json:"candidate_context_length"`
	ComparedCacheModes                  []memory.KVCacheMode `json:"compared_cache_modes,omitempty"`
	NormalContextValidated              bool                 `json:"normal_context_validated"`
	StressContextValidated              bool                 `json:"stress_context_validated"`
	BaselineVisibleTokensPerSec         float64              `json:"baseline_visible_tokens_per_sec,omitempty"`
	CandidateVisibleTokensPerSec        float64              `json:"candidate_visible_tokens_per_sec,omitempty"`
	BaselineInputOutputTokensPerSec     float64              `json:"baseline_input_output_tokens_per_sec,omitempty"`
	CandidateInputOutputTokensPerSec    float64              `json:"candidate_input_output_tokens_per_sec,omitempty"`
	BaselineWallDuration                time.Duration        `json:"baseline_wall_duration,omitempty"`
	CandidateWallDuration               time.Duration        `json:"candidate_wall_duration,omitempty"`
	BaselineRestoreDuration             time.Duration        `json:"baseline_restore_duration,omitempty"`
	CandidateRestoreDuration            time.Duration        `json:"candidate_restore_duration,omitempty"`
	BaselinePeakMemoryBytes             uint64               `json:"baseline_peak_memory_bytes,omitempty"`
	CandidatePeakMemoryBytes            uint64               `json:"candidate_peak_memory_bytes,omitempty"`
	BaselineActivePlusCacheMemoryBytes  uint64               `json:"baseline_active_plus_cache_memory_bytes,omitempty"`
	CandidateActivePlusCacheMemoryBytes uint64               `json:"candidate_active_plus_cache_memory_bytes,omitempty"`
	BaselineEnergyJoules                float64              `json:"baseline_energy_joules,omitempty"`
	CandidateEnergyJoules               float64              `json:"candidate_energy_joules,omitempty"`
	EstimatedPowerWatts                 float64              `json:"estimated_power_watts,omitempty"`
}

// ProductionTurboQuantPromotionDecision records the audited result. A
// ProductionCandidate may be exposed as an explicit cache option, but
// EnableByDefault remains false until a separate product decision changes it.
type ProductionTurboQuantPromotionDecision struct {
	ProductionCandidate bool    `json:"production_candidate"`
	EnableByDefault     bool    `json:"enable_by_default"`
	Reason              string  `json:"reason"`
	WallSpeedup         float64 `json:"wall_speedup,omitempty"`
	VisibleSpeedup      float64 `json:"visible_speedup,omitempty"`
	RestoreSpeedup      float64 `json:"restore_speedup,omitempty"`
	MemorySavingsRatio  float64 `json:"memory_savings_ratio,omitempty"`
	EnergySavingsRatio  float64 `json:"energy_savings_ratio,omitempty"`
}

// DefaultProductionTurboQuantPolicy returns the opt-in TurboQuant validation
// policy from GOAL.md. The 3.5 bits/channel target is a validation hypothesis,
// not a default runtime setting.
func DefaultProductionTurboQuantPolicy() ProductionTurboQuantPolicy {
	return ProductionTurboQuantPolicy{
		TargetModelID:                   OfficialGemma4E2BTargetLock().ModelID,
		CacheMode:                       memory.KVCacheModeTurboQuant,
		Mode:                            "turboquant-kv",
		TargetEffectiveBitsMilli:        3500,
		EnabledByDefault:                false,
		RequiresExplicitOptIn:           true,
		RequiresRetainedWorkflow:        true,
		RequiresQualityParity:           true,
		RequiresSideBySideBenchmark:     true,
		RequiresNormalContextValidation: true,
		RequiresStressContextValidation: true,
		MinimumRetainedTurns:            ProductionMTPPromotionMinRetainedTurns,
		NormalContextLength:             ProductionLaneLongContextLength,
		StressContextLength:             ProductionLaneHyperLongContextLength,
		CompareAgainstCacheModes: []memory.KVCacheMode{
			memory.KVCacheModeFP16,
			memory.KVCacheModePaged,
			memory.KVCacheModeQ8,
			memory.KVCacheModeKQ8VQ4,
		},
		RequiredMetrics: []string{
			"baseline_cache_mode",
			"candidate_cache_mode",
			"same_load_policy",
			"baseline_cache_policy",
			"candidate_cache_policy",
			"baseline_context_length",
			"candidate_context_length",
			"normal_context_validated",
			"stress_context_validated",
			"candidate_peak_memory_bytes",
			"baseline_peak_memory_bytes",
			"candidate_active_plus_cache_memory_bytes",
			"baseline_active_plus_cache_memory_bytes",
			"candidate_wall_duration",
			"baseline_wall_duration",
			"candidate_restore_duration",
			"baseline_restore_duration",
			"candidate_visible_tokens_per_sec",
			"baseline_visible_tokens_per_sec",
			"candidate_input_output_tokens_per_sec",
			"baseline_input_output_tokens_per_sec",
			"candidate_energy_joules",
			"baseline_energy_joules",
			"estimated_power_watts",
			"quality_flags",
		},
	}
}

// EvaluateProductionTurboQuantPromotion applies the production rule from
// GOAL.md: TurboQuant must stay explicit, and can become a production candidate
// only after retained-workflow quality parity plus normal and stress-context
// memory/wall evidence.
func EvaluateProductionTurboQuantPromotion(policy ProductionTurboQuantPolicy, evidence ProductionTurboQuantPromotionEvidence) ProductionTurboQuantPromotionDecision {
	if policy.CacheMode == "" {
		policy = DefaultProductionTurboQuantPolicy()
	}
	decision := ProductionTurboQuantPromotionDecision{
		EnableByDefault:    false,
		WallSpeedup:        durationSpeedup(evidence.BaselineWallDuration, evidence.CandidateWallDuration),
		VisibleSpeedup:     ratioSpeedup(evidence.CandidateVisibleTokensPerSec, evidence.BaselineVisibleTokensPerSec),
		RestoreSpeedup:     durationSpeedup(evidence.BaselineRestoreDuration, evidence.CandidateRestoreDuration),
		MemorySavingsRatio: byteSavingsRatio(evidence.BaselinePeakMemoryBytes, evidence.CandidatePeakMemoryBytes),
		EnergySavingsRatio: ratioSavings(evidence.BaselineEnergyJoules, evidence.CandidateEnergyJoules),
	}
	if policy.RequiresExplicitOptIn && policy.EnabledByDefault {
		decision.Reason = "TurboQuant policy must remain explicit opt-in"
		return decision
	}
	if evidence.CandidateCacheMode != policy.CacheMode {
		decision.Reason = "TurboQuant candidate cache mode is required"
		return decision
	}
	if evidence.BaselineCacheMode == "" || evidence.BaselineCacheMode == policy.CacheMode || !turboQuantModeInSlice(policy.CompareAgainstCacheModes, evidence.BaselineCacheMode) {
		decision.Reason = "TurboQuant baseline cache mode must be one of fp16, paged, q8, or k-q8-v-q4"
		return decision
	}
	if policy.RequiresRetainedWorkflow && !evidence.RetainedWorkflow {
		decision.Reason = "retained workflow evidence is required before TurboQuant promotion"
		return decision
	}
	if evidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "retained workflow turn count is below the TurboQuant promotion minimum"
		return decision
	}
	if policy.RequiresQualityParity && !evidence.QualityMatches {
		decision.Reason = "quality parity is required before TurboQuant promotion"
		return decision
	}
	if len(evidence.QualityFlags) > 0 {
		decision.Reason = "quality flags must be empty before TurboQuant promotion"
		return decision
	}
	if policy.RequiresSideBySideBenchmark && !turboQuantComparedAllModes(policy.CompareAgainstCacheModes, evidence.ComparedCacheModes) {
		decision.Reason = "TurboQuant must be compared side by side against fp16, paged, q8, and k-q8-v-q4 cache modes"
		return decision
	}
	if policy.RequiresNormalContextValidation && !evidence.NormalContextValidated {
		decision.Reason = "normal 30k-40k retained-context validation is required before TurboQuant promotion"
		return decision
	}
	if policy.RequiresStressContextValidation && !evidence.StressContextValidated {
		decision.Reason = "100k stress-context validation is required before TurboQuant promotion"
		return decision
	}
	if decision.WallSpeedup == 0 || decision.MemorySavingsRatio <= 0 || decision.EnergySavingsRatio <= 0 || evidence.EstimatedPowerWatts <= 0 {
		decision.Reason = "TurboQuant wall, memory, and estimated-energy evidence are required"
		return decision
	}
	if evidence.BaselineActivePlusCacheMemoryBytes == 0 || evidence.CandidateActivePlusCacheMemoryBytes == 0 {
		decision.Reason = "TurboQuant active+cache memory evidence is required"
		return decision
	}
	if evidence.BaselineVisibleTokensPerSec <= 0 || evidence.CandidateVisibleTokensPerSec <= 0 {
		decision.Reason = "TurboQuant visible throughput evidence is required"
		return decision
	}
	if !productionTurboQuantHasLoadPolicyEvidence(evidence) {
		decision.Reason = "TurboQuant load policy evidence is required"
		return decision
	}
	if evidence.BaselineInputOutputTokensPerSec <= 0 || evidence.CandidateInputOutputTokensPerSec <= 0 {
		decision.Reason = "TurboQuant input+output throughput evidence is required"
		return decision
	}
	if decision.WallSpeedup <= 1 && decision.RestoreSpeedup <= 1 {
		decision.Reason = "TurboQuant must improve retained wall time or restore time before promotion"
		return decision
	}
	decision.ProductionCandidate = true
	decision.Reason = "TurboQuant retained workflow saves memory/energy with quality parity"
	return decision
}

func turboQuantComparedAllModes(required, actual []memory.KVCacheMode) bool {
	for _, want := range required {
		found := false
		for _, got := range actual {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func turboQuantModeInSlice(values []memory.KVCacheMode, needle memory.KVCacheMode) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func productionTurboQuantHasLoadPolicyEvidence(evidence ProductionTurboQuantPromotionEvidence) bool {
	return evidence.SameLoadPolicy &&
		evidence.BaselineCachePolicy != "" &&
		evidence.BaselineCachePolicy == evidence.CandidateCachePolicy &&
		evidence.BaselineContextLength > 0 &&
		evidence.BaselineContextLength == evidence.CandidateContextLength
}

func byteSavingsRatio(baseline, candidate uint64) float64 {
	if baseline == 0 || candidate == 0 || candidate >= baseline {
		return 0
	}
	return 1 - float64(candidate)/float64(baseline)
}

func ratioSavings(baseline, candidate float64) float64 {
	if baseline <= 0 || candidate <= 0 || candidate >= baseline {
		return 0
	}
	return 1 - candidate/baseline
}
