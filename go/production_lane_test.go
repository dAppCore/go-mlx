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
	if len(policy.Tiers) != 3 {
		t.Fatalf("tiers = %+v, want quality/default/constrained", policy.Tiers)
	}
	if policy.Tiers[0].Bits != 8 || policy.Tiers[0].ModelID != "mlx-community/gemma-4-e2b-it-8bit" || !policy.Tiers[0].QualityFirst {
		t.Fatalf("quality tier = %+v, want q8 quality-first", policy.Tiers[0])
	}
	if policy.Tiers[1].Bits != 6 || policy.Tiers[1].ModelID != "mlx-community/gemma-4-e2b-it-6bit" || !policy.Tiers[1].ProductDefault {
		t.Fatalf("default tier = %+v, want q6 product default", policy.Tiers[1])
	}
	if policy.Tiers[2].Bits != 4 || policy.Tiers[2].ModelID != "mlx-community/gemma-4-e2b-it-4bit" || !policy.Tiers[2].ConstrainedOnly || !policy.Tiers[2].ArchivedControl {
		t.Fatalf("constrained tier = %+v, want q4 constrained archived control", policy.Tiers[2])
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
		Gemma4FastRuntimeGateAsyncDecodePrefetch,
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
		Gemma4FastRuntimeGateFixedGemma4SharedMask,
		Gemma4FastRuntimeGateFixedGemma4Sliding,
		Gemma4FastRuntimeGateNativeFixedSliding,
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
	if !policy.RequiresGreedyParity || !policy.RequiresRetainedWorkflow || policy.RequiresSideBySideBenchmark == false {
		t.Fatalf("policy requirements = %+v, want side-by-side retained greedy-parity benchmark", policy)
	}
	for _, metric := range []string{
		"target_only_visible_tokens_per_sec",
		"mtp_visible_tokens_per_sec",
		"target_only_wall_duration",
		"mtp_wall_duration",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"quality_flags",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionLane_EvaluateMTPPromotion_RejectsSlowerOrUnproven_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        95,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               11 * time.Second,
		MTPProposedTokens:             40,
		MTPTargetVerifyCalls:          20,
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
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		MTPProposedTokens:             40,
		MTPTargetVerifyCalls:          20,
	})
	if unproven.EnableByDefault || !core.Contains(unproven.Reason, "retained") {
		t.Fatalf("unproven decision = %+v, want retained-workflow gate", unproven)
	}
}

func TestProductionLane_EvaluateMTPPromotion_AcceptsFasterGreedyParityEvidence_Good(t *testing.T) {
	policy := DefaultProductionMTPPolicy()

	decision := EvaluateProductionMTPPromotion(policy, ProductionMTPPromotionEvidence{
		RetainedWorkflow:              true,
		Turns:                         10,
		GreedyOutputMatches:           true,
		TargetOnlyVisibleTokensPerSec: 100,
		MTPVisibleTokensPerSec:        125,
		TargetOnlyWallDuration:        10 * time.Second,
		MTPWallDuration:               8 * time.Second,
		MTPProposedTokens:             40,
		MTPAcceptedTokens:             30,
		MTPRejectedTokens:             10,
		MTPTargetVerifyCalls:          20,
	})

	if !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want MTP promotion when retained wall and visible speed both win", decision)
	}
	if decision.WallSpeedup <= 1 || decision.VisibleSpeedup <= 1 {
		t.Fatalf("speedups = wall:%f visible:%f, want both > 1", decision.WallSpeedup, decision.VisibleSpeedup)
	}
}
