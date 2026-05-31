// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "dappco.re/go/mlx/memory"

const ProductionCombinedMTPAndTurboQuantMode = "mtp+turboquant-kv"

// ProductionCombinedMTPAndTurboQuantPolicy describes the evidence required
// before the app can expose MTP drafting on top of TurboQuant K/V as a
// production candidate. It remains explicit opt-in even when both component
// lanes pass.
type ProductionCombinedMTPAndTurboQuantPolicy struct {
	TargetModelID                   string             `json:"target_model_id"`
	AssistantModelID                string             `json:"assistant_model_id"`
	Mode                            string             `json:"mode"`
	CacheMode                       memory.KVCacheMode `json:"cache_mode"`
	EnabledByDefault                bool               `json:"enabled_by_default"`
	RequiresExplicitOptIn           bool               `json:"requires_explicit_opt_in"`
	RequiresRetainedWorkflow        bool               `json:"requires_retained_workflow"`
	RequiresGreedyParity            bool               `json:"requires_greedy_parity"`
	RequiresTurboQuantQualityParity bool               `json:"requires_turboquant_quality_parity"`
	RequiresMTPPromotion            bool               `json:"requires_mtp_promotion"`
	RequiresTurboQuantPromotion     bool               `json:"requires_turboquant_promotion"`
	MinimumRetainedTurns            int                `json:"minimum_retained_turns"`
	RequiredMetrics                 []string           `json:"required_metrics,omitempty"`
}

// ProductionCombinedMTPAndTurboQuantDecision records the audited combined-lane
// result. A production candidate can be shown as an advanced opt-in path, but
// enabling it by default requires a later product decision.
type ProductionCombinedMTPAndTurboQuantDecision struct {
	ProductionCandidate          bool    `json:"production_candidate"`
	EnableByDefault              bool    `json:"enable_by_default"`
	Reason                       string  `json:"reason"`
	MTPEligible                  bool    `json:"mtp_eligible"`
	TurboQuantEligible           bool    `json:"turboquant_eligible"`
	MTPWallSpeedup               float64 `json:"mtp_wall_speedup,omitempty"`
	MTPVisibleSpeedup            float64 `json:"mtp_visible_speedup,omitempty"`
	MTPAcceptanceRate            float64 `json:"mtp_acceptance_rate,omitempty"`
	TurboQuantMemorySavingsRatio float64 `json:"turboquant_memory_savings_ratio,omitempty"`
	TurboQuantEnergySavingsRatio float64 `json:"turboquant_energy_savings_ratio,omitempty"`
}

// DefaultProductionCombinedMTPAndTurboQuantPolicy returns the GOAL.md
// intersection gate: MTP must still pass while using TurboQuant K/V, and
// TurboQuant must still pass quality and memory gates under retained workflows.
func DefaultProductionCombinedMTPAndTurboQuantPolicy() ProductionCombinedMTPAndTurboQuantPolicy {
	return ProductionCombinedMTPAndTurboQuantPolicy{
		TargetModelID:                   OfficialGemma4E2BTargetLock().ModelID,
		AssistantModelID:                OfficialGemma4E2BAssistantLock().ModelID,
		Mode:                            ProductionCombinedMTPAndTurboQuantMode,
		CacheMode:                       memory.KVCacheModeTurboQuant,
		EnabledByDefault:                false,
		RequiresExplicitOptIn:           true,
		RequiresRetainedWorkflow:        true,
		RequiresGreedyParity:            true,
		RequiresTurboQuantQualityParity: true,
		RequiresMTPPromotion:            true,
		RequiresTurboQuantPromotion:     true,
		MinimumRetainedTurns:            ProductionMTPPromotionMinRetainedTurns,
		RequiredMetrics: []string{
			"mtp_target_only_cache_mode",
			"mtp_cache_mode",
			"mtp_proposed_tokens",
			"mtp_accepted_tokens",
			"mtp_rejected_tokens",
			"mtp_target_verify_calls",
			"mtp_draft_calls",
			"turboquant_candidate_cache_mode",
			"turboquant_candidate_layout_version",
			"turboquant_candidate_key_algorithm",
			"turboquant_candidate_value_algorithm",
			"turboquant_candidate_effective_bits_milli",
			"turboquant_candidate_qjl_residual",
			"turboquant_candidate_metadata_bytes",
			"turboquant_quality_flags",
			"turboquant_active_plus_cache_memory_savings",
			"quality_flags",
		},
	}
}

// EvaluateProductionCombinedMTPAndTurboQuantPromotion applies the combined
// production rule from GOAL.md: neither component lane may hide the other's
// quality, memory, acceptance, or verify-loop regressions.
func EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy ProductionCombinedMTPAndTurboQuantPolicy, mtpEvidence ProductionMTPPromotionEvidence, turboEvidence ProductionTurboQuantPromotionEvidence) ProductionCombinedMTPAndTurboQuantDecision {
	if policy.CacheMode == "" {
		policy = DefaultProductionCombinedMTPAndTurboQuantPolicy()
	}
	mtpDecision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), mtpEvidence)
	turboDecision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), turboEvidence)
	decision := ProductionCombinedMTPAndTurboQuantDecision{
		EnableByDefault:              false,
		MTPEligible:                  mtpDecision.EnableByDefault,
		TurboQuantEligible:           turboDecision.ProductionCandidate,
		MTPWallSpeedup:               mtpDecision.WallSpeedup,
		MTPVisibleSpeedup:            mtpDecision.VisibleSpeedup,
		MTPAcceptanceRate:            mtpDecision.AcceptanceRate,
		TurboQuantMemorySavingsRatio: turboDecision.MemorySavingsRatio,
		TurboQuantEnergySavingsRatio: turboDecision.EnergySavingsRatio,
	}
	if policy.RequiresExplicitOptIn && policy.EnabledByDefault {
		decision.Reason = "combined MTP+TurboQuant policy must remain explicit opt-in"
		return decision
	}
	if policy.RequiresRetainedWorkflow && (!mtpEvidence.RetainedWorkflow || !turboEvidence.RetainedWorkflow) {
		decision.Reason = "combined MTP+TurboQuant retained workflow evidence is required"
		return decision
	}
	if mtpEvidence.Turns < policy.MinimumRetainedTurns || turboEvidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "combined MTP+TurboQuant retained workflow turn count is below the promotion minimum"
		return decision
	}
	if policy.RequiresGreedyParity && !mtpEvidence.GreedyOutputMatches {
		decision.Reason = "combined MTP+TurboQuant requires MTP greedy output parity"
		return decision
	}
	if policy.RequiresTurboQuantQualityParity && !turboEvidence.QualityMatches {
		decision.Reason = "combined MTP+TurboQuant requires TurboQuant quality parity"
		return decision
	}
	if mtpEvidence.TargetOnlyCacheMode != string(policy.CacheMode) || mtpEvidence.MTPCacheMode != string(policy.CacheMode) {
		decision.Reason = "combined MTP benchmark must run target-only and MTP with TurboQuant cache mode"
		return decision
	}
	if turboEvidence.CandidateCacheMode != policy.CacheMode {
		decision.Reason = "combined MTP+TurboQuant requires a TurboQuant candidate cache mode"
		return decision
	}
	if policy.RequiresMTPPromotion && !mtpDecision.EnableByDefault {
		decision.Reason = "MTP must pass target-only retained workflow under TurboQuant: " + mtpDecision.Reason
		return decision
	}
	if policy.RequiresTurboQuantPromotion && !turboDecision.ProductionCandidate {
		decision.Reason = "TurboQuant must pass retained quality/memory gates before combined promotion: " + turboDecision.Reason
		return decision
	}
	decision.ProductionCandidate = true
	decision.Reason = "combined MTP+TurboQuant retained workflow passes both lanes and remains explicit opt-in"
	return decision
}
