// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/profile"
)

func TestProductionLane_DefaultGemma4E2B_Good(t *testing.T) {
	lane := DefaultProductionLane()

	if lane.ModelID != "mlx-community/gemma-4-e2b-it-6bit" {
		t.Fatalf("ModelID = %q, want Gemma 4 E2B q6 default", lane.ModelID)
	}
	if lane.Architecture != "gemma4_text" || lane.ChatTemplate != "gemma4" || lane.QuantBits != 6 {
		t.Fatalf("lane identity = %+v, want Gemma 4 text q6 with Gemma chat template", lane)
	}
	if ProductionLaneProductDefaultQuantBits != 6 || ProductionLaneQualityQuantBits != 8 || ProductionLaneConstrainedQuantBits != 4 {
		t.Fatalf("quant constants = default:%d quality:%d constrained:%d, want 6/8/4", ProductionLaneProductDefaultQuantBits, ProductionLaneQualityQuantBits, ProductionLaneConstrainedQuantBits)
	}
	if lane.ContextLength != 4096 || lane.MaxTokens != 128 || lane.Runs != 3 {
		t.Fatalf("profile shape = context:%d tokens:%d runs:%d, want GOAL.md target shape", lane.ContextLength, lane.MaxTokens, lane.Runs)
	}
	if ProductionLaneLongContextLength != 32768 || ProductionLaneHyperLongContextLength != 131072 || ProductionLaneLongFormMaxTokens != 8192 || ProductionLaneLongContextPrefillChunkSize != 512 || ProductionLaneLongContextPromptChunkBytes != 4096 || ProductionLanePagedKVPageSize != 2048 || ProductionLaneRetainedKVCacheDType != "fp16" {
		t.Fatalf("long context shape = context:%d hyper:%d tokens:%d prefill:%d prompt:%d page:%d dtype:%s, want retained-state defaults", ProductionLaneLongContextLength, ProductionLaneHyperLongContextLength, ProductionLaneLongFormMaxTokens, ProductionLaneLongContextPrefillChunkSize, ProductionLaneLongContextPromptChunkBytes, ProductionLanePagedKVPageSize, ProductionLaneRetainedKVCacheDType)
	}
	if lane.IncludeOutput || !lane.TraceTokenPhases {
		t.Fatalf("profile reporting = include_output:%v trace:%v, want hidden output plus token phase trace", lane.IncludeOutput, lane.TraceTokenPhases)
	}
	if lane.Prompt != DefaultNewSessionText || !core.Contains(lane.Prompt, "Lemma") {
		t.Fatalf("Prompt = %q, want Lemma new-session default", lane.Prompt)
	}
}

func TestProductionLane_DefaultProductionQuantizationPolicy_Good(t *testing.T) {
	policy := DefaultProductionQuantizationPolicy()

	if policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID || policy.AssistantModelID != OfficialGemma4E2BAssistantLock().ModelID || policy.ArchivedBaseline != ProductionLaneArchivedBaselineModelID {
		t.Fatalf("policy identity = %+v, want official target+assistant plus archived q4 baseline", policy)
	}
	if policy.DefaultBits != 6 || policy.QualityBits != 8 || policy.ConstrainedBits != 4 {
		t.Fatalf("policy bits = default:%d quality:%d constrained:%d, want 6/8/4", policy.DefaultBits, policy.QualityBits, policy.ConstrainedBits)
	}
	if policy.ActiveParameterEstimate != ProductionLaneActiveParameterEstimate || !core.Contains(policy.DecodeThroughputEstimate, "memory bandwidth") {
		t.Fatalf("throughput estimate = params:%d formula:%q, want active-weight-read model", policy.ActiveParameterEstimate, policy.DecodeThroughputEstimate)
	}
	for _, metric := range []string{
		"load_duration",
		"peak_memory_bytes",
		"retained_restore_duration",
		"raw_decode_tokens_per_sec",
		"active_weight_read_bytes_per_token",
		"memory_bandwidth_bytes_per_sec",
		"long_output_quality_flags",
		"step_down_working_set_bytes",
		"context_length",
	} {
		if !stringSliceContains(policy.RequiredBenchmarkMetrics, metric) {
			t.Fatalf("RequiredBenchmarkMetrics = %v, missing %q", policy.RequiredBenchmarkMetrics, metric)
		}
	}
	if len(policy.Tiers) != 3 {
		t.Fatalf("tiers = %+v, want quality/default/constrained", policy.Tiers)
	}
	if policy.Tiers[0].Bits != 8 || policy.Tiers[0].ModelID != "mlx-community/gemma-4-e2b-it-8bit" || !policy.Tiers[0].QualityFirst || policy.Tiers[0].StepDownToBits != 6 {
		t.Fatalf("quality tier = %+v, want q8 quality-first", policy.Tiers[0])
	}
	if policy.Tiers[0].ActiveWeightReadBytesPerToken != 2300000000 {
		t.Fatalf("quality tier active read = %d, want q8 active-weight-read estimate", policy.Tiers[0].ActiveWeightReadBytesPerToken)
	}
	if policy.Tiers[1].Bits != 6 || policy.Tiers[1].ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !policy.Tiers[1].ProductDefault || policy.Tiers[1].StepDownToBits != 4 {
		t.Fatalf("default tier = %+v, want q6 product default", policy.Tiers[1])
	}
	if policy.Tiers[1].ActiveWeightReadBytesPerToken != 1725000000 {
		t.Fatalf("default tier active read = %d, want q6 active-weight-read estimate", policy.Tiers[1].ActiveWeightReadBytesPerToken)
	}
	if policy.Tiers[2].Bits != 4 || policy.Tiers[2].ModelID != "mlx-community/gemma-4-e2b-it-4bit" || !policy.Tiers[2].ConstrainedOnly || !policy.Tiers[2].ArchivedControl {
		t.Fatalf("constrained tier = %+v, want q4 constrained archived control", policy.Tiers[2])
	}
	if policy.Tiers[2].ActiveWeightReadBytesPerToken != 1150000000 {
		t.Fatalf("constrained tier active read = %d, want q4 active-weight-read estimate", policy.Tiers[2].ActiveWeightReadBytesPerToken)
	}
}

func TestProductionLane_DefaultQuantizationPackLocks_Good(t *testing.T) {
	locks := DefaultProductionQuantizationPackLocks()
	if len(locks) != 3 {
		t.Fatalf("DefaultProductionQuantizationPackLocks() = %d locks, want q8 quality plus q6 default plus q4 constrained fallback", len(locks))
	}
	byBits := map[int]ProductionQuantizationPackLock{}
	for _, lock := range locks {
		byBits[lock.QuantBits] = lock
		if lock.BaseModelID != OfficialGemma4E2BTargetLock().ModelID || lock.SourceCheckedAt != "2026-05-31" {
			t.Fatalf("lock provenance = %+v, want official Google E2B source checked on 2026-05-31", lock)
		}
		if lock.BaseRevision != OfficialGemma4E2BTargetLock().Revision || lock.ConversionCommand == "" || lock.AccuracySmoke == "" {
			t.Fatalf("lock conversion record = %+v, want official base revision, conversion command, and accuracy-smoke status", lock)
		}
		if lock.Licence != "apache-2.0" || lock.LicenceURL != "https://ai.google.dev/gemma/docs/gemma_4_license" {
			t.Fatalf("lock licence = %+v, want Apache-2.0 Gemma 4 licence metadata", lock)
		}
		if lock.ConfigSHA256 == "" || lock.TokenizerSHA256 == "" || lock.TokenizerConfigSHA256 == "" || lock.SafetensorsIndexSHA256 == "" {
			t.Fatalf("lock hashes incomplete: %+v", lock)
		}
		if !lock.SafetensorsIndexPresent || len(lock.WeightFiles) == 0 {
			t.Fatalf("lock safetensors = present:%v files:%d, want indexed MLX quant pack", lock.SafetensorsIndexPresent, len(lock.WeightFiles))
		}
	}

	q8 := byBits[ProductionLaneQualityQuantBits]
	if q8.ModelID != "mlx-community/gemma-4-e2b-it-8bit" || q8.Revision != "48ef0737faea4e72556670e49da0ba421027a545" {
		t.Fatalf("q8 lock identity = %+v", q8)
	}
	if len(q8.WeightFiles) != 2 || q8.WeightFiles[0].Name != "model-00001-of-00002.safetensors" || q8.WeightFiles[1].Name != "model-00002-of-00002.safetensors" {
		t.Fatalf("q8 weights = %+v, want two locked shards", q8.WeightFiles)
	}

	q6 := byBits[ProductionLaneProductDefaultQuantBits]
	if q6.ModelID != ProductionLaneModelID || q6.Revision != "40d43b05f94ee798c0e40fe19fcd9ef49928486b" {
		t.Fatalf("q6 lock identity = %+v", q6)
	}
	if len(q6.WeightFiles) != 1 || q6.WeightFiles[0].Name != "model.safetensors" {
		t.Fatalf("q6 weights = %+v, want one locked safetensors file", q6.WeightFiles)
	}

	q4 := byBits[ProductionLaneConstrainedQuantBits]
	if q4.Name != "constrained" || q4.ModelID != ProductionLaneArchivedBaselineModelID || q4.Revision != "99d9a53ff828d365a8ecae538e45f80a08d612cd" {
		t.Fatalf("q4 lock identity = %+v", q4)
	}
	if q4.QuantGroup != 64 || q4.QuantMode != "affine" {
		t.Fatalf("q4 quantisation = group:%d mode:%q, want affine g64", q4.QuantGroup, q4.QuantMode)
	}
	if len(q4.WeightFiles) != 1 || q4.WeightFiles[0].Name != "model.safetensors" || q4.WeightFiles[0].Bytes != 3581101896 {
		t.Fatalf("q4 weights = %+v, want one locked safetensors fallback file", q4.WeightFiles)
	}
}

func TestProductionLane_SelectProductionQuantizationTier_Good(t *testing.T) {
	wide := memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 90 * memory.GiB}
	choice := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
	})
	if choice.Tier.Bits != 6 || choice.Tier.ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !choice.Fits {
		t.Fatalf("default wide choice = %+v, want fitting q6", choice)
	}

	quality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if quality.Tier.Bits != 8 || quality.Tier.ModelID != "mlx-community/gemma-4-e2b-it-8bit" || !quality.Fits {
		t.Fatalf("quality wide choice = %+v, want fitting q8", quality)
	}

	constrained := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
		ContextLength: ProductionLaneLongContextLength,
	})
	if constrained.Tier.Bits != 4 || constrained.Tier.ModelID != "mlx-community/gemma-4-e2b-it-4bit" || !constrained.Fits {
		t.Fatalf("constrained long-context choice = %+v, want fitting q4 fallback", constrained)
	}

	forced := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:              wide,
		ContextLength:       ProductionLaneContextLength,
		ConstrainedFallback: true,
	})
	if forced.Tier.Bits != 4 || !forced.Tier.ConstrainedOnly {
		t.Fatalf("forced constrained choice = %+v, want q4 fallback", forced)
	}
}

func TestProductionLane_ArchitectureProfileNative_Good(t *testing.T) {
	lane := DefaultProductionLane()
	prof, ok := profile.LookupArchitectureProfile(lane.Architecture)

	if !ok {
		t.Fatalf("profile.LookupArchitectureProfile(%q) = false", lane.Architecture)
	}
	if !prof.NativeRuntime || !prof.Generation || !prof.Chat {
		t.Fatalf("architecture profile = %+v, want native chat/generation runtime", prof)
	}
	if prof.ChatTemplate != lane.ChatTemplate {
		t.Fatalf("ChatTemplate = %q, want lane template %q", prof.ChatTemplate, lane.ChatTemplate)
	}
}

func TestProductionLane_DefaultGemma4FastRuntimeGates_Good(t *testing.T) {
	gates := DefaultGemma4FastRuntimeGates()
	seen := map[string]bool{}
	for _, gate := range gates {
		seen[gate] = true
	}

	for _, want := range []string{
		Gemma4FastRuntimeGateExpertIDMatVec,
		Gemma4FastRuntimeGateExpertIDFused,
		Gemma4FastRuntimeGateSortedExpertPrefill,
		Gemma4FastRuntimeGateNativeMLPMatVec,
		Gemma4FastRuntimeGateNativeLinearMatVec,
		Gemma4FastRuntimeGateNativeRouterMatVec,
		Gemma4FastRuntimeGateNativeRouterTopK,
		Gemma4FastRuntimeGateDirectGreedyToken,
		Gemma4FastRuntimeGateGenerationStream,
		Gemma4FastRuntimeGateFixedGemma4SharedMask,
		Gemma4FastRuntimeGatePagedDecodeFastConcat,
	} {
		if !seen[want] {
			t.Fatalf("DefaultGemma4FastRuntimeGates() = %v, missing %s", gates, want)
		}
	}
	for _, rejected := range []string{
		"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		Gemma4FastRuntimeGateNativePagedAttention,
		Gemma4FastRuntimeGateFixedGemma4Cache,
		Gemma4FastRuntimeGateFixedGemma4Sliding,
		Gemma4FastRuntimeGateNativeFixedSliding,
		Gemma4FastRuntimeGateAsyncDecodePrefetch,
	} {
		if seen[rejected] {
			t.Fatalf("DefaultGemma4FastRuntimeGates() = %v, should exclude rejected gate %s", gates, rejected)
		}
	}
}

func TestProductionLane_DefaultMTPPolicy_OptInUntilRetainedBenchmarkWin_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	if policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID || policy.AssistantModelID != OfficialGemma4E2BAssistantLock().ModelID {
		t.Fatalf("policy identity = %+v, want official target+assistant IDs", policy)
	}
	if policy.EnabledByDefault {
		t.Fatalf("EnabledByDefault = true, want MTP opt-in until retained benchmark promotion")
	}
	if policy.DefaultDraftTokens != 2 || policy.MinimumRetainedTurns != 10 {
		t.Fatalf("policy defaults = draft:%d turns:%d, want draft=2 and retained 10-turn evidence", policy.DefaultDraftTokens, policy.MinimumRetainedTurns)
	}
	if !intSliceEqual(policy.RequiredDraftTokenSweeps, []int{1, 2, 4}) {
		t.Fatalf("RequiredDraftTokenSweeps = %v, want 1/2/4 sweep evidence", policy.RequiredDraftTokenSweeps)
	}
	if !policy.RequiresGreedyParity || !policy.RequiresRetainedWorkflow || policy.RequiresSideBySideBenchmark == false {
		t.Fatalf("policy requirements = %+v, want side-by-side retained greedy-parity benchmark", policy)
	}
	for _, metric := range []string{
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
		"mtp_observed_draft_token_sweeps",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"quality_flags",
		"assistant_architecture",
		"assistant_ordered_embeddings",
		"assistant_centroids",
		"assistant_centroid_intermediate_top_k",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsSlowerOrUnproven_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               95,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      11 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   110 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   4096,
		TargetOnlyActivePlusCacheMemoryBytes: 3072,
		MTPActivePlusCacheMemoryBytes:        3072,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      1000,
		EstimatedPowerWatts:                  100,
		MTPTargetTokensPerSec:                90,
		MTPWarmDecodeTokensPerSec:            94,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP rejected when slower than target-only", decision)
	}
	if !core.Contains(decision.Reason, "faster") {
		t.Fatalf("decision reason = %q, want faster-than-target-only failure", decision.Reason)
	}

	unproven := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              false,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        120,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     118,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if unproven.EnableByDefault || !core.Contains(unproven.Reason, "retained") {
		t.Fatalf("unproven decision = %+v, want retained-workflow gate", unproven)
	}

	missingOperationalEvidence := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        125,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     123,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPAcceptedTokens:             30,
		MTPRejectedTokens:             10,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if missingOperationalEvidence.EnableByDefault || !core.Contains(missingOperationalEvidence.Reason, "restore, memory, and energy") {
		t.Fatalf("missing operational evidence decision = %+v, want restore/memory/energy gate", missingOperationalEvidence)
	}

	missingActiveCacheEvidence := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        125,
		MTPTargetTokensPerSec:         110,
		MTPWarmDecodeTokensPerSec:     123,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		TargetOnlyRestoreDuration:     100 * time.Millisecond,
		MTPRestoreDuration:            80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:     4096,
		MTPPeakMemoryBytes:            3584,
		TargetOnlyEnergyJoules:        1000,
		MTPEnergyJoules:               760,
		EstimatedPowerWatts:           100,
		SpeculativeDraftModelPath:     OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:        2,
		MTPDraftTokenSchedule:         []int{2, 2},
		MTPObservedDraftTokenSweeps:   []int{1, 2, 4},
		MTPProposedTokens:             40,
		MTPAcceptedTokens:             30,
		MTPRejectedTokens:             10,
		MTPTargetVerifyCalls:          20,
		MTPDraftCalls:                 20,
	})
	if missingActiveCacheEvidence.EnableByDefault || !core.Contains(missingActiveCacheEvidence.Reason, "active+cache") {
		t.Fatalf("missing active+cache decision = %+v, want active+cache memory gate", missingActiveCacheEvidence)
	}

	missingDraftIdentity := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingDraftIdentity.EnableByDefault || !core.Contains(missingDraftIdentity.Reason, "draft model") {
		t.Fatalf("missing draft identity decision = %+v, want draft model/schedule gate", missingDraftIdentity)
	}

	missingDraftSweep := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{2},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingDraftSweep.EnableByDefault || !core.Contains(missingDraftSweep.Reason, "draft-token sweep") {
		t.Fatalf("missing draft-token sweep decision = %+v, want required 1/2/4 sweep gate", missingDraftSweep)
	}

	missingThroughputBreakdown := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingThroughputBreakdown.EnableByDefault || !core.Contains(missingThroughputBreakdown.Reason, "target-verify and warm-decode") {
		t.Fatalf("missing throughput breakdown decision = %+v, want target-verify/warm-decode gate", missingThroughputBreakdown)
	}

	missingAcceptanceAccounting := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if missingAcceptanceAccounting.EnableByDefault || !core.Contains(missingAcceptanceAccounting.Reason, "accepted/rejected") {
		t.Fatalf("missing acceptance accounting decision = %+v, want accepted/rejected counter gate", missingAcceptanceAccounting)
	}

	missingDraftCalls := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
	})
	if missingDraftCalls.EnableByDefault || !core.Contains(missingDraftCalls.Reason, "draft-call") {
		t.Fatalf("missing draft-call decision = %+v, want draft-call counter gate", missingDraftCalls)
	}

	noAcceptedDraftTokens := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPRejectedTokens:                    40,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})
	if noAcceptedDraftTokens.EnableByDefault || !core.Contains(noAcceptedDraftTokens.Reason, "accepted draft tokens") {
		t.Fatalf("zero accepted draft decision = %+v, want accepted-token gate", noAcceptedDraftTokens)
	}
}

func TestProductionLane_EvaluateMTPPromotion_AcceptsFasterGreedyParityEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
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
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		AssistantArchitecture:                OfficialGemma4E2BAssistantLock().ModelType,
		AssistantOrderedEmbeddings:           true,
		AssistantCentroids:                   2048,
		AssistantCentroidIntermediateTopK:    32,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP promotion when retained wall and visible speed both win", decision)
	}
	if decision.WallSpeedup <= 1 || decision.VisibleSpeedup <= 1 {
		t.Fatalf("speedups = wall:%f visible:%f, want both > 1", decision.WallSpeedup, decision.VisibleSpeedup)
	}
	if decision.RestoreSpeedup <= 1 || decision.EnergySavings <= 0 {
		t.Fatalf("operational ratios = restore:%f energy:%f, want restore speedup and energy savings recorded", decision.RestoreSpeedup, decision.EnergySavings)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingAssistantLayoutEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		TargetOnlyInputOutputTokensPerSec:    33000,
		MTPInputOutputTokensPerSec:           41000,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
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
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "ordered-embedding evidence") {
		t.Fatalf("decision = %+v, want official assistant ordered-embedding evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingInputOutputEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
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
		TargetOnlyCacheMode:                  "paged",
		MTPCacheMode:                         "paged",
		TargetOnlyContextLength:              ProductionLaneLongContextLength,
		MTPContextLength:                     ProductionLaneLongContextLength,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "input+output") {
		t.Fatalf("decision = %+v, want input+output throughput evidence gate", decision)
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsMissingLoadPolicyEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     true,
		Turns:                                10,
		GreedyOutputMatches:                  true,
		TargetOnlyVisibleTokensPerSec:        100,
		MTPVisibleTokensPerSec:               125,
		MTPTargetTokensPerSec:                110,
		MTPWarmDecodeTokensPerSec:            123,
		TargetOnlyWallDuration:               10 * time.Second,
		MTPWallDuration:                      8 * time.Second,
		TargetOnlyRestoreDuration:            100 * time.Millisecond,
		MTPRestoreDuration:                   80 * time.Millisecond,
		TargetOnlyPeakMemoryBytes:            4096,
		MTPPeakMemoryBytes:                   3584,
		TargetOnlyActivePlusCacheMemoryBytes: 2560,
		MTPActivePlusCacheMemoryBytes:        2304,
		TargetOnlyEnergyJoules:               1000,
		MTPEnergyJoules:                      760,
		EstimatedPowerWatts:                  100,
		SpeculativeDraftModelPath:            OfficialGemma4E2BAssistantLock().ModelID,
		SpeculativeDraftTokens:               2,
		MTPDraftTokenSchedule:                []int{2, 2},
		MTPObservedDraftTokenSweeps:          []int{1, 2, 4},
		MTPProposedTokens:                    40,
		MTPAcceptedTokens:                    30,
		MTPRejectedTokens:                    10,
		MTPTargetVerifyCalls:                 20,
		MTPDraftCalls:                        20,
	})

	if decision.EnableByDefault || !core.Contains(decision.Reason, "load policy") {
		t.Fatalf("decision = %+v, want load-policy evidence gate", decision)
	}
}

func TestProductionLane_DefaultTurboQuantPolicy_ResearchOptIn_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	if policy.CacheMode != memory.KVCacheModeTurboQuant || policy.TargetModelID != OfficialGemma4E2BTargetLock().ModelID {
		t.Fatalf("policy identity = %+v, want official target plus turboquant cache mode", policy)
	}
	if policy.EnabledByDefault {
		t.Fatalf("EnabledByDefault = true, want TurboQuant opt-in until retained workflow validation")
	}
	if policy.TargetEffectiveBitsMilli != 3500 {
		t.Fatalf("TargetEffectiveBitsMilli = %d, want 3500 for 3.5 bits/channel research target", policy.TargetEffectiveBitsMilli)
	}
	if !policy.RequiresExplicitOptIn || !policy.RequiresRetainedWorkflow || !policy.RequiresQualityParity ||
		!policy.RequiresSideBySideBenchmark || !policy.RequiresNormalContextValidation || !policy.RequiresStressContextValidation {
		t.Fatalf("policy requirements = %+v, want explicit retained-workflow quality-gated research mode", policy)
	}
	for _, mode := range []memory.KVCacheMode{
		memory.KVCacheModeFP16,
		memory.KVCacheModePaged,
		memory.KVCacheModeQ8,
		memory.KVCacheModeKQ8VQ4,
	} {
		if !kvCacheModeSliceContains(policy.CompareAgainstCacheModes, mode) {
			t.Fatalf("CompareAgainstCacheModes = %v, missing %q", policy.CompareAgainstCacheModes, mode)
		}
	}
	for _, metric := range []string{
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
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsIncompleteValidation_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:             true,
		Turns:                        ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:               true,
		BaselineCacheMode:            memory.KVCacheModePaged,
		CandidateCacheMode:           memory.KVCacheModeTurboQuant,
		ComparedCacheModes:           policy.CompareAgainstCacheModes,
		NormalContextValidated:       true,
		StressContextValidated:       false,
		BaselineWallDuration:         10 * time.Second,
		CandidateWallDuration:        8 * time.Second,
		BaselinePeakMemoryBytes:      10 * memory.GiB,
		CandidatePeakMemoryBytes:     7 * memory.GiB,
		BaselineEnergyJoules:         1000,
		CandidateEnergyJoules:        800,
		EstimatedPowerWatts:          100,
		BaselineRestoreDuration:      100 * time.Millisecond,
		CandidateRestoreDuration:     80 * time.Millisecond,
		BaselineVisibleTokensPerSec:  80,
		CandidateVisibleTokensPerSec: 80,
	})

	if decision.ProductionCandidate {
		t.Fatalf("decision = %+v, want rejection until 100k stress lane is validated", decision)
	}
	if !core.Contains(decision.Reason, "stress") {
		t.Fatalf("decision reason = %q, want stress-context validation failure", decision.Reason)
	}

	missingBaselineMode := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:             true,
		Turns:                        ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:               true,
		CandidateCacheMode:           memory.KVCacheModeTurboQuant,
		ComparedCacheModes:           policy.CompareAgainstCacheModes,
		NormalContextValidated:       true,
		StressContextValidated:       true,
		BaselineWallDuration:         10 * time.Second,
		CandidateWallDuration:        8 * time.Second,
		BaselinePeakMemoryBytes:      10 * memory.GiB,
		CandidatePeakMemoryBytes:     7 * memory.GiB,
		BaselineEnergyJoules:         1000,
		CandidateEnergyJoules:        800,
		EstimatedPowerWatts:          100,
		BaselineRestoreDuration:      100 * time.Millisecond,
		CandidateRestoreDuration:     80 * time.Millisecond,
		BaselineVisibleTokensPerSec:  80,
		CandidateVisibleTokensPerSec: 80,
	})
	if missingBaselineMode.ProductionCandidate || !core.Contains(missingBaselineMode.Reason, "baseline cache mode") {
		t.Fatalf("missing baseline mode decision = %+v, want baseline cache mode gate", missingBaselineMode)
	}

	missingVisibleThroughput := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
	})
	if missingVisibleThroughput.ProductionCandidate || !core.Contains(missingVisibleThroughput.Reason, "visible throughput") {
		t.Fatalf("missing visible throughput decision = %+v, want visible-throughput gate", missingVisibleThroughput)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_AllowsMeasuredCandidate_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
		BaselineInputOutputTokensPerSec:     33000,
		CandidateInputOutputTokensPerSec:    36000,
	})

	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v, want TurboQuant production candidate after full retained validation", decision)
	}
	if decision.EnableByDefault {
		t.Fatalf("EnableByDefault = true, want TurboQuant still explicit/non-default after candidate promotion")
	}
	if decision.WallSpeedup <= 1 || decision.MemorySavingsRatio <= 0 || decision.EnergySavingsRatio <= 0 {
		t.Fatalf("decision metrics = %+v, want wall, memory, and energy savings recorded", decision)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsNoActiveCacheMemoryWin_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  5 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 6 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
		BaselineInputOutputTokensPerSec:     33000,
		CandidateInputOutputTokensPerSec:    36000,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "active+cache memory savings") {
		t.Fatalf("decision = %+v, want active+cache memory-savings gate", decision)
	}
	if decision.MemorySavingsRatio != 0 {
		t.Fatalf("memory savings ratio = %f, want no active+cache savings recorded", decision.MemorySavingsRatio)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsMissingInputOutputEvidence_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "full",
		CandidateCachePolicy:                "full",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "input+output") {
		t.Fatalf("decision = %+v, want input+output throughput evidence gate", decision)
	}
}

func TestProductionLane_EvaluateTurboQuantPromotion_RejectsMissingLoadPolicyEvidence_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	decision := EvaluateProductionTurboQuantPromotion(policy, ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   memory.KVCacheModePaged,
		CandidateCacheMode:                  memory.KVCacheModeTurboQuant,
		ComparedCacheModes:                  policy.CompareAgainstCacheModes,
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               8 * time.Second,
		BaselinePeakMemoryBytes:             10 * memory.GiB,
		CandidatePeakMemoryBytes:            7 * memory.GiB,
		BaselineActivePlusCacheMemoryBytes:  8 * memory.GiB,
		CandidateActivePlusCacheMemoryBytes: 5 * memory.GiB,
		BaselineEnergyJoules:                1000,
		CandidateEnergyJoules:               800,
		EstimatedPowerWatts:                 100,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            80 * time.Millisecond,
		BaselineVisibleTokensPerSec:         80,
		CandidateVisibleTokensPerSec:        80,
	})

	if decision.ProductionCandidate || !core.Contains(decision.Reason, "load policy") {
		t.Fatalf("decision = %+v, want load-policy evidence gate", decision)
	}
}

func kvCacheModeSliceContains(values []memory.KVCacheMode, needle memory.KVCacheMode) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func intSliceEqual(values, want []int) bool {
	if len(values) != len(want) {
		return false
	}
	for i, value := range values {
		if value != want[i] {
			return false
		}
	}
	return true
}
