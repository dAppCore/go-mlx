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
	target := "metaladapter TokenizerModel AdapterModel ProbeableModel BenchableModel Evaluator SFTTrainer"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.TokenizerModel = (*metaladapter)(nil)
	var _ inference.AdapterModel = (*metaladapter)(nil)
	var _ inference.ProbeableModel = (*metaladapter)(nil)
	var _ inference.BenchableModel = (*metaladapter)(nil)
	var _ inference.Evaluator = (*metaladapter)(nil)
	var _ inference.SFTTrainer = (*metaladapter)(nil)
}

func TestInferenceContract_MetalBackendImplementsFitPlanner_Good(t *testing.T) {
	target := "metalbackend ModelFitPlanner"
	if target == "" {
		t.Fatalf("missing coverage target for %s", t.Name())
	}
	var _ inference.ModelFitPlanner = (*metalbackend)(nil)
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
