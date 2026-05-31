// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/profile"
)

func TestProductionLane_DefaultGemma4E2B_Good(t *testing.T) {
	lane := DefaultProductionLane()

	if lane.ModelID != "mlx-community/gemma-4-e2b-it-4bit" {
		t.Fatalf("ModelID = %q, want Gemma 4 E2B q4", lane.ModelID)
	}
	if lane.Architecture != "gemma4_text" || lane.ChatTemplate != "gemma4" || lane.QuantBits != 4 {
		t.Fatalf("lane identity = %+v, want Gemma 4 text q4 with Gemma chat template", lane)
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

	if policy.TargetModelID != "google/gemma-4-E2B-it" || policy.ArchivedBaseline != ProductionLaneModelID {
		t.Fatalf("policy identity = %+v, want official target plus archived q4 baseline", policy)
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
