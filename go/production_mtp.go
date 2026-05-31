// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "time"

const (
	// ProductionMTPDefaultDraftTokens is the conservative official Gemma 4 E2B
	// assistant block size used for promotion benchmarks.
	ProductionMTPDefaultDraftTokens = 2
	// ProductionMTPPromotionMinRetainedTurns is the minimum retained workflow
	// length before MTP can be considered for default interactive use.
	ProductionMTPPromotionMinRetainedTurns = 10
)

// ProductionMTPPolicy describes when the app may promote the official Gemma 4
// E2B assistant path from opt-in benchmark lane to default interactive mode.
type ProductionMTPPolicy struct {
	TargetModelID               string   `json:"target_model_id"`
	AssistantModelID            string   `json:"assistant_model_id"`
	Mode                        string   `json:"mode"`
	DefaultDraftTokens          int      `json:"default_draft_tokens"`
	RequiredDraftTokenSweeps    []int    `json:"required_draft_token_sweeps,omitempty"`
	MinimumRetainedTurns        int      `json:"minimum_retained_turns"`
	EnabledByDefault            bool     `json:"enabled_by_default"`
	RequiresRetainedWorkflow    bool     `json:"requires_retained_workflow"`
	RequiresGreedyParity        bool     `json:"requires_greedy_parity"`
	RequiresSideBySideBenchmark bool     `json:"requires_side_by_side_benchmark"`
	RequiredMetrics             []string `json:"required_metrics"`
}

// ProductionMTPPromotionEvidence is the measured target-only versus MTP row
// used to decide whether assistant drafting can become the default.
type ProductionMTPPromotionEvidence struct {
	RetainedWorkflow                     bool          `json:"retained_workflow"`
	Turns                                int           `json:"turns"`
	GreedyOutputMatches                  bool          `json:"greedy_output_matches"`
	QualityFlags                         []string      `json:"quality_flags,omitempty"`
	TargetOnlyVisibleTokensPerSec        float64       `json:"target_only_visible_tokens_per_sec,omitempty"`
	MTPVisibleTokensPerSec               float64       `json:"mtp_visible_tokens_per_sec,omitempty"`
	TargetOnlyInputOutputTokensPerSec    float64       `json:"target_only_input_output_tokens_per_sec,omitempty"`
	MTPInputOutputTokensPerSec           float64       `json:"mtp_input_output_tokens_per_sec,omitempty"`
	MTPTargetTokensPerSec                float64       `json:"mtp_target_tokens_per_sec,omitempty"`
	MTPWarmDecodeTokensPerSec            float64       `json:"mtp_warm_decode_tokens_per_sec,omitempty"`
	TargetOnlyWallDuration               time.Duration `json:"target_only_wall_duration,omitempty"`
	MTPWallDuration                      time.Duration `json:"mtp_wall_duration,omitempty"`
	TargetOnlyRestoreDuration            time.Duration `json:"target_only_restore_duration,omitempty"`
	MTPRestoreDuration                   time.Duration `json:"mtp_restore_duration,omitempty"`
	TargetOnlyPeakMemoryBytes            uint64        `json:"target_only_peak_memory_bytes,omitempty"`
	MTPPeakMemoryBytes                   uint64        `json:"mtp_peak_memory_bytes,omitempty"`
	TargetOnlyActivePlusCacheMemoryBytes uint64        `json:"target_only_active_plus_cache_memory_bytes,omitempty"`
	MTPActivePlusCacheMemoryBytes        uint64        `json:"mtp_active_plus_cache_memory_bytes,omitempty"`
	TargetOnlyEnergyJoules               float64       `json:"target_only_energy_joules,omitempty"`
	MTPEnergyJoules                      float64       `json:"mtp_energy_joules,omitempty"`
	EstimatedPowerWatts                  float64       `json:"estimated_power_watts,omitempty"`
	SameLoadPolicy                       bool          `json:"same_load_policy"`
	TargetOnlyCachePolicy                string        `json:"target_only_cache_policy"`
	MTPCachePolicy                       string        `json:"mtp_cache_policy"`
	TargetOnlyCacheMode                  string        `json:"target_only_cache_mode"`
	MTPCacheMode                         string        `json:"mtp_cache_mode"`
	TargetOnlyContextLength              int           `json:"target_only_context_length"`
	MTPContextLength                     int           `json:"mtp_context_length"`
	SpeculativeDraftModelPath            string        `json:"speculative_draft_model_path,omitempty"`
	SpeculativeDraftTokens               int           `json:"speculative_draft_tokens,omitempty"`
	MTPDraftTokenSchedule                []int         `json:"mtp_draft_token_schedule,omitempty"`
	MTPObservedDraftTokenSweeps          []int         `json:"mtp_observed_draft_token_sweeps,omitempty"`
	MTPProposedTokens                    int           `json:"mtp_proposed_tokens,omitempty"`
	MTPAcceptedTokens                    int           `json:"mtp_accepted_tokens,omitempty"`
	MTPRejectedTokens                    int           `json:"mtp_rejected_tokens,omitempty"`
	MTPTargetVerifyCalls                 int           `json:"mtp_target_verify_calls,omitempty"`
	MTPDraftCalls                        int           `json:"mtp_draft_calls,omitempty"`
}

// ProductionMTPPromotionDecision is the audited promotion result. A false
// decision keeps MTP opt-in even if a short smoke run completed successfully.
type ProductionMTPPromotionDecision struct {
	EnableByDefault bool    `json:"enable_by_default"`
	Reason          string  `json:"reason"`
	WallSpeedup     float64 `json:"wall_speedup,omitempty"`
	VisibleSpeedup  float64 `json:"visible_speedup,omitempty"`
	RestoreSpeedup  float64 `json:"restore_speedup,omitempty"`
	EnergySavings   float64 `json:"energy_savings_ratio,omitempty"`
	AcceptanceRate  float64 `json:"acceptance_rate,omitempty"`
}

// DefaultProductionMTPPolicy returns the active official Gemma 4 E2B assistant
// policy. It deliberately does not enable MTP by default; promotion requires
// retained-workflow evidence against target-only generation.
func DefaultProductionMTPPolicy() ProductionMTPPolicy {
	return ProductionMTPPolicy{
		TargetModelID:               OfficialGemma4E2BTargetLock().ModelID,
		AssistantModelID:            OfficialGemma4E2BAssistantLock().ModelID,
		Mode:                        SpeculativeDecodeModeMTP,
		DefaultDraftTokens:          ProductionMTPDefaultDraftTokens,
		RequiredDraftTokenSweeps:    defaultProductionMTPDraftTokenSweeps(),
		MinimumRetainedTurns:        ProductionMTPPromotionMinRetainedTurns,
		EnabledByDefault:            false,
		RequiresRetainedWorkflow:    true,
		RequiresGreedyParity:        true,
		RequiresSideBySideBenchmark: true,
		RequiredMetrics: []string{
			"speculative_draft_model_path",
			"speculative_draft_tokens",
			"target_only_visible_tokens_per_sec",
			"mtp_visible_tokens_per_sec",
			"target_only_input_output_tokens_per_sec",
			"mtp_input_output_tokens_per_sec",
			"mtp_target_tokens_per_sec",
			"mtp_warm_decode_tokens_per_sec",
			"target_only_wall_duration",
			"mtp_wall_duration",
			"target_only_restore_duration",
			"mtp_restore_duration",
			"target_only_peak_memory_bytes",
			"mtp_peak_memory_bytes",
			"target_only_active_plus_cache_memory_bytes",
			"mtp_active_plus_cache_memory_bytes",
			"target_only_energy_joules",
			"mtp_energy_joules",
			"estimated_power_watts",
			"same_load_policy",
			"target_only_cache_policy",
			"mtp_cache_policy",
			"target_only_cache_mode",
			"mtp_cache_mode",
			"target_only_context_length",
			"mtp_context_length",
			"mtp_draft_token_schedule",
			"mtp_observed_draft_token_sweeps",
			"mtp_proposed_tokens",
			"mtp_accepted_tokens",
			"mtp_rejected_tokens",
			"mtp_target_verify_calls",
			"mtp_draft_calls",
			"quality_flags",
		},
	}
}

// EvaluateProductionMTPPromotion applies the production rule from GOAL.md:
// assistant drafting can become the default only when it beats target-only on a
// retained workflow without changing greedy output quality.
func EvaluateProductionMTPPromotion(policy ProductionMTPPolicy, evidence ProductionMTPPromotionEvidence) ProductionMTPPromotionDecision {
	if policy.MinimumRetainedTurns == 0 {
		policy = DefaultProductionMTPPolicy()
	}
	decision := ProductionMTPPromotionDecision{
		WallSpeedup:     durationSpeedup(evidence.TargetOnlyWallDuration, evidence.MTPWallDuration),
		VisibleSpeedup:  ratioSpeedup(evidence.MTPVisibleTokensPerSec, evidence.TargetOnlyVisibleTokensPerSec),
		RestoreSpeedup:  durationSpeedup(evidence.TargetOnlyRestoreDuration, evidence.MTPRestoreDuration),
		EnergySavings:   ratioSavings(evidence.TargetOnlyEnergyJoules, evidence.MTPEnergyJoules),
		AcceptanceRate:  ratioSpeedup(float64(evidence.MTPAcceptedTokens), float64(evidence.MTPProposedTokens)),
		EnableByDefault: false,
	}
	if policy.RequiresRetainedWorkflow && !evidence.RetainedWorkflow {
		decision.Reason = "retained workflow evidence is required before MTP promotion"
		return decision
	}
	if evidence.Turns < policy.MinimumRetainedTurns {
		decision.Reason = "retained workflow turn count is below the MTP promotion minimum"
		return decision
	}
	if policy.RequiresGreedyParity && !evidence.GreedyOutputMatches {
		decision.Reason = "greedy output parity is required before MTP promotion"
		return decision
	}
	if len(evidence.QualityFlags) > 0 {
		decision.Reason = "quality flags must be empty before MTP promotion"
		return decision
	}
	if policy.RequiresSideBySideBenchmark && (decision.WallSpeedup == 0 || decision.VisibleSpeedup == 0) {
		decision.Reason = "side-by-side target-only and MTP wall/visible metrics are required"
		return decision
	}
	if evidence.SpeculativeDraftModelPath == "" || evidence.SpeculativeDraftTokens <= 0 || len(evidence.MTPDraftTokenSchedule) == 0 {
		decision.Reason = "MTP draft model, draft token count, and schedule evidence are required"
		return decision
	}
	for _, draftTokens := range evidence.MTPDraftTokenSchedule {
		if draftTokens <= 0 {
			decision.Reason = "MTP draft token schedule must contain positive draft counts"
			return decision
		}
	}
	if len(missingProductionMTPDraftTokenSweeps(requiredProductionMTPDraftTokenSweeps(policy), evidence.MTPObservedDraftTokenSweeps)) > 0 {
		decision.Reason = "MTP draft-token sweep evidence is incomplete"
		return decision
	}
	if evidence.MTPTargetTokensPerSec <= 0 || evidence.MTPWarmDecodeTokensPerSec <= 0 {
		decision.Reason = "MTP target-verify and warm-decode throughput evidence are required"
		return decision
	}
	if evidence.MTPProposedTokens <= 0 || evidence.MTPTargetVerifyCalls <= 0 || evidence.MTPDraftCalls <= 0 {
		decision.Reason = "MTP proposed-token, target-verify, and draft-call counters are required"
		return decision
	}
	if evidence.MTPAcceptedTokens < 0 || evidence.MTPRejectedTokens < 0 || evidence.MTPAcceptedTokens+evidence.MTPRejectedTokens != evidence.MTPProposedTokens {
		decision.Reason = "MTP accepted/rejected counters must account for every proposed token"
		return decision
	}
	if evidence.MTPAcceptedTokens == 0 {
		decision.Reason = "MTP accepted draft tokens are required before promotion"
		return decision
	}
	if evidence.TargetOnlyRestoreDuration <= 0 || evidence.MTPRestoreDuration <= 0 ||
		evidence.TargetOnlyPeakMemoryBytes == 0 || evidence.MTPPeakMemoryBytes == 0 ||
		evidence.TargetOnlyEnergyJoules <= 0 || evidence.MTPEnergyJoules <= 0 ||
		evidence.EstimatedPowerWatts <= 0 {
		decision.Reason = "MTP restore, memory, and energy evidence are required"
		return decision
	}
	if evidence.TargetOnlyActivePlusCacheMemoryBytes == 0 || evidence.MTPActivePlusCacheMemoryBytes == 0 {
		decision.Reason = "MTP active+cache memory evidence is required"
		return decision
	}
	if decision.WallSpeedup <= 1 || decision.VisibleSpeedup <= 1 {
		decision.Reason = "MTP must be faster than target-only on retained wall time and visible throughput"
		return decision
	}
	if decision.EnergySavings <= 0 {
		decision.Reason = "MTP must not increase estimated energy before promotion"
		return decision
	}
	if !productionMTPHasLoadPolicyEvidence(evidence) {
		decision.Reason = "MTP load policy evidence is required"
		return decision
	}
	if evidence.TargetOnlyInputOutputTokensPerSec <= 0 || evidence.MTPInputOutputTokensPerSec <= 0 {
		decision.Reason = "MTP input+output throughput evidence is required"
		return decision
	}
	decision.EnableByDefault = true
	decision.Reason = "MTP retained workflow is faster than target-only with greedy parity"
	return decision
}

func durationSpeedup(baseline, candidate time.Duration) float64 {
	if baseline <= 0 || candidate <= 0 {
		return 0
	}
	return float64(baseline) / float64(candidate)
}

func ratioSpeedup(candidate, baseline float64) float64 {
	if baseline <= 0 || candidate <= 0 {
		return 0
	}
	return candidate / baseline
}

func productionMTPHasLoadPolicyEvidence(evidence ProductionMTPPromotionEvidence) bool {
	return evidence.SameLoadPolicy &&
		evidence.TargetOnlyCachePolicy != "" &&
		evidence.TargetOnlyCachePolicy == evidence.MTPCachePolicy &&
		evidence.TargetOnlyCacheMode != "" &&
		evidence.TargetOnlyCacheMode == evidence.MTPCacheMode &&
		evidence.TargetOnlyContextLength > 0 &&
		evidence.TargetOnlyContextLength == evidence.MTPContextLength
}

func defaultProductionMTPDraftTokenSweeps() []int {
	return []int{1, 2, 4}
}

func requiredProductionMTPDraftTokenSweeps(policy ProductionMTPPolicy) []int {
	if len(policy.RequiredDraftTokenSweeps) == 0 {
		return defaultProductionMTPDraftTokenSweeps()
	}
	return policy.RequiredDraftTokenSweeps
}

func missingProductionMTPDraftTokenSweeps(required, observed []int) []int {
	seen := make(map[int]bool, len(observed))
	for _, value := range observed {
		if value > 0 {
			seen[value] = true
		}
	}
	missing := make([]int, 0, len(required))
	for _, value := range required {
		if value > 0 && !seen[value] {
			missing = append(missing, value)
		}
	}
	return missing
}
