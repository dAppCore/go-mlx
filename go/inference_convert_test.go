// SPDX-Licence-Identifier: EUPL-1.2

// Unit tests for the pure metal→inference converters in inference_convert.go.
// These functions translate the native probe / config envelopes into the
// dappco.re/go/inference contract types; they are pure value-mapping, so the
// tests construct synthetic native structs and assert the mapped fields —
// no model load.

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// toInferenceProbeEvent maps the native event envelope. The mandatory
// header (Kind/Phase/Step) plus the optional pointer sections each map
// independently; Good drives every optional section at once so the per-
// field copies are all exercised, Bad confirms a header-only event leaves
// the optional sections nil.
func TestInferenceConvert_ToInferenceProbeEvent_GoodAllSectionsMapped(t *testing.T) {
	event := metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Step:  7,
		Meta:  map[string]string{"k": "v"},
		Token: &metal.ProbeToken{ID: 11, Text: "hi", PromptTokens: 3, GeneratedTokens: 4},
		Logits: &metal.ProbeLogits{
			VocabSize: 32000,
			MinLogit:  -9,
			MaxLogit:  12,
			MeanLogit: 0.5,
			Top:       []metal.ProbeLogit{{TokenID: 11, Logit: 12}},
		},
		Entropy:       &metal.ProbeEntropy{Value: 3.5, Unit: "nats"},
		SelectedHeads: &metal.ProbeHeadSelection{Layer: 2, Heads: []int{0, 3}},
		LayerCoherence: &metal.ProbeLayerCoherence{
			Layer:          2,
			KeyCoherence:   0.8,
			ValueCoherence: 0.6,
			CrossAlignment: 0.4,
			KVCoupling:     0.9,
			HeadEntropy:    0.2,
			PhaseLock:      0.7,
		},
		RouterDecision: &metal.ProbeRouterDecision{Layer: 2, ExpertIDs: []int{1, 5}, Weights: []float32{0.7, 0.3}},
		Residual:       &metal.ProbeResidualSummary{Layer: 2, Mean: 0.1, RMS: 0.2, L2Norm: 0.3},
		Cache:          &metal.ProbeCachePressure{PromptTokens: 3, GeneratedTokens: 4, CacheTokens: 7, Utilization: 0.5},
		Memory:         &metal.ProbeMemoryPressure{ActiveBytes: 100, PeakBytes: 200},
		Training:       &metal.ProbeTraining{Epoch: 1, Step: 9, Loss: 0.42, LearningRate: 0.001},
	}

	out := toInferenceProbeEvent(event)

	if string(out.Kind) != string(metal.ProbeEventToken) || string(out.Phase) != string(metal.ProbePhaseDecode) || out.Step != 7 {
		t.Fatalf("header = (%v,%v,%d), want (token,decode,7)", out.Kind, out.Phase, out.Step)
	}
	if out.Labels["k"] != "v" {
		t.Errorf("Labels = %v, want k=v", out.Labels)
	}
	if out.Token == nil || out.Token.ID != 11 || out.Token.Text != "hi" {
		t.Errorf("Token = %+v, want id 11 text hi", out.Token)
	}
	if out.Logits == nil || out.Logits.VocabularySize != 32000 || len(out.Logits.Top) != 1 {
		t.Errorf("Logits = %+v", out.Logits)
	}
	if out.Entropy == nil || out.Entropy.Value != 3.5 || out.Entropy.Unit != "nats" {
		t.Errorf("Entropy = %+v", out.Entropy)
	}
	if out.SelectedHeads == nil || out.SelectedHeads.Layer != 2 || len(out.SelectedHeads.Heads) != 2 {
		t.Errorf("SelectedHeads = %+v", out.SelectedHeads)
	}
	if out.LayerCoherence == nil || out.LayerCoherence.KVCoupling != 0.9 || out.LayerCoherence.PhaseLock != 0.7 {
		t.Errorf("LayerCoherence = %+v", out.LayerCoherence)
	}
	if out.RouterDecision == nil || len(out.RouterDecision.ExpertIDs) != 2 || len(out.RouterDecision.ExpertProbs) != 2 {
		t.Errorf("RouterDecision = %+v", out.RouterDecision)
	}
	if out.Residual == nil || out.Residual.Norm != 0.3 {
		t.Errorf("Residual = %+v", out.Residual)
	}
	if out.Cache == nil || out.Cache.CachedTokens != 7 || out.Cache.HitRate != 0.5 {
		t.Errorf("Cache = %+v", out.Cache)
	}
	if out.Memory == nil || out.Memory.ActiveBytes != 100 || out.Memory.PeakBytes != 200 {
		t.Errorf("Memory = %+v", out.Memory)
	}
	if out.Training == nil || out.Training.Loss != 0.42 || out.Training.LearningRate != 0.001 {
		t.Errorf("Training = %+v", out.Training)
	}
}

func TestInferenceConvert_ToInferenceProbeEvent_BadHeaderOnlyLeavesSectionsNil(t *testing.T) {
	out := toInferenceProbeEvent(metal.ProbeEvent{Kind: metal.ProbeEventToken, Step: 2})

	if out.Step != 2 {
		t.Fatalf("Step = %d, want 2", out.Step)
	}
	switch {
	case out.Token != nil:
		t.Error("Token = non-nil, want nil")
	case out.Logits != nil:
		t.Error("Logits = non-nil, want nil")
	case out.Entropy != nil:
		t.Error("Entropy = non-nil, want nil")
	case out.SelectedHeads != nil:
		t.Error("SelectedHeads = non-nil, want nil")
	case out.LayerCoherence != nil:
		t.Error("LayerCoherence = non-nil, want nil")
	case out.RouterDecision != nil:
		t.Error("RouterDecision = non-nil, want nil")
	case out.Residual != nil:
		t.Error("Residual = non-nil, want nil")
	case out.Cache != nil:
		t.Error("Cache = non-nil, want nil")
	case out.Memory != nil:
		t.Error("Memory = non-nil, want nil")
	case out.Training != nil:
		t.Error("Training = non-nil, want nil")
	}
}

// sftDType maps the bfloat16 toggle to the SFT weight dtype: true selects
// bf16, false leaves the zero dtype so the trainer keeps the source
// precision.
func TestInferenceConvert_SFTDType_GoodBFloat16(t *testing.T) {
	if got := sftDType(true); got != DTypeBFloat16 {
		t.Errorf("sftDType(true) = %v, want DTypeBFloat16", got)
	}
}

func TestInferenceConvert_SFTDType_BadDefaultIsZeroDType(t *testing.T) {
	if got := sftDType(false); got != 0 {
		t.Errorf("sftDType(false) = %v, want 0", got)
	}
}

// inferenceGenerateConfigToMetal copies the request-level sampling knobs
// into the native config. MinP is mapped only when the linked inference
// contract exposes the field (a cached reflect probe), so the assertion
// stays on the always-present fields.
func TestInferenceConvert_GenerateConfigToMetal_GoodCopiesSamplingKnobs(t *testing.T) {
	think := true
	cfg := inference.GenerateConfig{
		MaxTokens:           128,
		Temperature:         0.7,
		TopK:                40,
		TopP:                0.95,
		MinP:                0.05,
		Seed:                99,
		SeedSet:             true,
		StopTokens:          []int32{2},
		SuppressTokens:      []int32{7, 8},
		MinTokensBeforeStop: 3,
		RepeatPenalty:       1.1,
		EnableThinking:      &think,
		ThinkingBudget:      64,
	}

	out := inferenceGenerateConfigToMetal(cfg)

	if out.MaxTokens != 128 || out.Temperature != 0.7 || out.TopK != 40 || out.TopP != 0.95 || out.MinP != 0.05 {
		t.Fatalf("core knobs = %+v", out)
	}
	if out.Seed != 99 || !out.SeedSet || out.RepeatPenalty != 1.1 || len(out.StopTokens) != 1 || len(out.SuppressTokens) != 2 || out.MinTokensBeforeStop != 3 || out.ThinkingBudget != 64 {
		t.Fatalf("aux knobs = %+v", out)
	}
	if out.EnableThinking == nil || !*out.EnableThinking {
		t.Errorf("EnableThinking = %v, want true", out.EnableThinking)
	}
}
