// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
)

func TestInferenceContract_MetalAdapterImplementsSharedInterfaces_Good(t *testing.T) {
	target := "metaladapter TokenizerModel AdapterModel ProbeableModel BenchableModel Evaluator SFTTrainer CapabilityReporter"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.TokenizerModel = (*metaladapter)(nil)
	var _ inference.AdapterModel = (*metaladapter)(nil)
	var _ inference.ProbeableModel = (*metaladapter)(nil)
	var _ inference.BenchableModel = (*metaladapter)(nil)
	var _ inference.Evaluator = (*metaladapter)(nil)
	var _ inference.SFTTrainer = (*metaladapter)(nil)
	var _ inference.CapabilityReporter = (*metaladapter)(nil)
}

func TestInferenceContract_MetalBackendImplementsFitPlanner_Good(t *testing.T) {
	target := "metalbackend ModelFitPlanner CapabilityReporter"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.ModelFitPlanner = (*metalbackend)(nil)
	var _ inference.CapabilityReporter = (*metalbackend)(nil)
	var _ inference.RuntimeMemoryLimiter = (*metalbackend)(nil)
}

func TestInferenceContract_MetalBackendRuntimeMemoryLimits_UglyZero(t *testing.T) {
	got := (&metalbackend{}).SetRuntimeMemoryLimits(inference.RuntimeMemoryLimits{})

	if got != (inference.RuntimeMemoryLimits{}) {
		t.Fatalf("SetRuntimeMemoryLimits zero = %+v, want zero response", got)
	}
}

func TestInferenceContract_MetalBackendCapabilities_Good(t *testing.T) {
	report := (&metalbackend{}).Capabilities()

	if report.Runtime.Backend != "metal" || !report.Runtime.NativeRuntime {
		t.Fatalf("runtime = %+v, want native metal", report.Runtime)
	}
	if !report.Supports(inference.CapabilityModelLoad) || !report.Supports(inference.CapabilityMemoryPlanning) {
		t.Fatalf("capabilities = %+v, want load and memory planning", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityLoRATraining) || !report.Supports(inference.CapabilityGRPO) {
		t.Fatalf("capabilities = %+v, want training features", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityProbeEvents) || !report.Supports(inference.CapabilityAttentionProbe) {
		t.Fatalf("capabilities = %+v, want probe features", report.CapabilityIDs())
	}
	if len(report.Architectures) == 0 || len(report.Quantizations) == 0 || len(report.CacheModes) == 0 {
		t.Fatalf("report = %+v, want architecture/quant/cache metadata", report)
	}
}

func TestInferenceContract_MetalAdapterCapabilities_UglyNilModel(t *testing.T) {
	report := (&metaladapter{}).Capabilities()

	if report.Available {
		t.Fatalf("Available = true, want false for nil loaded model")
	}
	if !report.Supports(inference.CapabilityGenerate) || !report.Supports(inference.CapabilityLoRAInference) {
		t.Fatalf("capabilities = %+v, want model feature surface even before load", report.CapabilityIDs())
	}
	if report.Adapter.Path != "" {
		t.Fatalf("adapter = %+v, want empty adapter identity", report.Adapter)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Good(t *testing.T) {
	report, err := (&metalbackend{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "qwen3",
		QuantBits:     4,
		ContextLength: 32768,
		NumLayers:     28,
		HiddenSize:    2048,
	}, 16*MemoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.ArchitectureOK || !report.QuantizationOK {
		t.Fatalf("PlanModelFit report = %+v, want supported qwen3/q4", report)
	}
	if report.MemoryPlan.ContextLength == 0 || report.MemoryPlan.CacheMode == "" {
		t.Fatalf("MemoryPlan = %+v, want context/cache recommendation", report.MemoryPlan)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Bad(t *testing.T) {
	report, err := (&metalbackend{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture: "unknown-transformer",
		QuantBits:    16,
	}, 8*MemoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || report.ArchitectureOK || report.QuantizationOK {
		t.Fatalf("PlanModelFit report = %+v, want unsupported architecture and quantization", report)
	}
}

func TestInferenceContract_MetalBackendPlanModelFit_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	report, err := (&metalbackend{}).PlanModelFit(ctx, inference.ModelIdentity{Architecture: "qwen3"}, 0)

	if err == nil {
		t.Fatalf("PlanModelFit cancelled error = nil, report=%+v", report)
	}
}

func TestInferenceContract_MetalAdapterSetProbeSink_Good(t *testing.T) {
	adapter := &metaladapter{}
	var got inference.ProbeEvent
	adapter.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		got = event
	}))

	toMetalInferenceProbeSink(adapter.probeSink).EmitProbe(metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Token: &metal.ProbeToken{ID: 7, Text: "ok", PromptTokens: 3, GeneratedTokens: 1},
	})

	if got.Kind != inference.ProbeEventToken || got.Token == nil || got.Token.Text != "ok" {
		t.Fatalf("probe event = %+v, want token event", got)
	}
}

func TestInferenceContract_ToInferenceProbeEvent_Ugly(t *testing.T) {
	got := toInferenceProbeEvent(metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Logits: &metal.ProbeLogits{
			VocabSize: 11,
			MinLogit:  -1.5,
			MaxLogit:  2.5,
			MeanLogit: 0.25,
			Top:       []metal.ProbeLogit{{TokenID: 4, Logit: 2.5}},
		},
	})

	if got.Logits == nil || got.Logits.VocabularySize != 11 || got.Logits.Top[0].ID != 4 {
		t.Fatalf("logits event = %+v, want compact logits", got)
	}
}
