// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
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
	if lane.ContextLength != 4096 || lane.MaxTokens != 128 || lane.Runs != 3 {
		t.Fatalf("profile shape = context:%d tokens:%d runs:%d, want GOAL.md target shape", lane.ContextLength, lane.MaxTokens, lane.Runs)
	}
	if ProductionLaneLongContextLength != 32768 || ProductionLaneHyperLongContextLength != 131072 || ProductionLaneLongFormMaxTokens != 8192 || ProductionLaneLongContextPrefillChunkSize != 512 || ProductionLaneLongContextPromptChunkBytes != 4096 || ProductionLanePagedKVPageSize != 1024 || ProductionLaneRetainedKVCacheDType != "fp16" {
		t.Fatalf("long context shape = context:%d hyper:%d tokens:%d prefill:%d prompt:%d page:%d dtype:%s, want retained-state defaults", ProductionLaneLongContextLength, ProductionLaneHyperLongContextLength, ProductionLaneLongFormMaxTokens, ProductionLaneLongContextPrefillChunkSize, ProductionLaneLongContextPromptChunkBytes, ProductionLanePagedKVPageSize, ProductionLaneRetainedKVCacheDType)
	}
	if lane.IncludeOutput || !lane.TraceTokenPhases {
		t.Fatalf("profile reporting = include_output:%v trace:%v, want hidden output plus token phase trace", lane.IncludeOutput, lane.TraceTokenPhases)
	}
	if !core.Contains(lane.Prompt, "retained model state") {
		t.Fatalf("Prompt = %q, want retained-state production prompt", lane.Prompt)
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

func TestProductionLane_LongContextGemma4FastRuntimeGates_Good(t *testing.T) {
	gates := LongContextGemma4FastRuntimeGates()
	if len(gates) != 0 {
		t.Fatalf("LongContextGemma4FastRuntimeGates() = %v, want no fixed-cache context gates", gates)
	}
}

func TestProductionLane_Gemma4FastRuntimeGatesForContext_HyperLongStaysPaged_Good(t *testing.T) {
	gates := Gemma4FastRuntimeGatesForContext(ProductionLaneHyperLongContextLength)
	seen := map[string]bool{}
	for _, gate := range gates {
		seen[gate] = true
	}
	for _, want := range []string{
		Gemma4FastRuntimeGateGenerationStream,
		Gemma4FastRuntimeGateAsyncDecodePrefetch,
		Gemma4FastRuntimeGateExpertIDMatVec,
		Gemma4FastRuntimeGatePagedDecodeFastConcat,
	} {
		if !seen[want] {
			t.Fatalf("Gemma4FastRuntimeGatesForContext() = %v, missing %s for hyper-long context", gates, want)
		}
	}
	for _, rejected := range []string{
		Gemma4FastRuntimeGateFixedGemma4Cache,
		Gemma4FastRuntimeGateFixedGemma4SharedMask,
		Gemma4FastRuntimeGateFixedGemma4Sliding,
		Gemma4FastRuntimeGateNativeFixedSliding,
	} {
		if seen[rejected] {
			t.Fatalf("Gemma4FastRuntimeGatesForContext() = %v, should exclude fixed-cache gate %s", gates, rejected)
		}
	}
}

func TestProductionLane_Gemma4FastRuntimeGatesForContext_LongContextStaysPaged_Good(t *testing.T) {
	gates := Gemma4FastRuntimeGatesForContext(ProductionLaneLongContextLength)
	seen := map[string]bool{}
	for _, gate := range gates {
		seen[gate] = true
	}
	for _, want := range []string{
		Gemma4FastRuntimeGateGenerationStream,
		Gemma4FastRuntimeGateAsyncDecodePrefetch,
	} {
		if !seen[want] {
			t.Fatalf("Gemma4FastRuntimeGatesForContext() = %v, missing %s for long context", gates, want)
		}
	}
	for _, rejected := range []string{
		Gemma4FastRuntimeGateFixedGemma4Cache,
		Gemma4FastRuntimeGateFixedGemma4SharedMask,
		Gemma4FastRuntimeGateFixedGemma4Sliding,
	} {
		if seen[rejected] {
			t.Fatalf("Gemma4FastRuntimeGatesForContext() = %v, should exclude fixed-cache gate %s", gates, rejected)
		}
	}
}
