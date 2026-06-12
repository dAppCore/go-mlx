// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestSpineProbeConversion_AllFields_Good(t *testing.T) {
	meta := map[string]string{"scope": "unit"}
	logitMeta := map[string]string{"logits": "kept"}
	got := toProbeEvent(metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Step:  6,
		Meta:  meta,
		Token: &metal.ProbeToken{ID: 1, Text: "tok", PromptTokens: 2, GeneratedTokens: 3},
		Logits: &metal.ProbeLogits{
			Shape:      []int32{1, 2},
			VocabSize:  16,
			MaxTokenID: 4,
			MaxLogit:   1.5,
			MinTokenID: 5,
			MinLogit:   -1.5,
			MeanLogit:  0.25,
			Top:        []metal.ProbeLogit{{TokenID: 4, Logit: 1.5, Probability: 0.7}},
			Values:     []float32{0.1, 0.2},
			Meta:       logitMeta,
		},
		Entropy:        &metal.ProbeEntropy{Value: 0.4, Unit: "nats"},
		SelectedHeads:  &metal.ProbeHeadSelection{Layer: 2, Heads: []int{1, 3}, Scores: []float64{0.5, 0.6}},
		LayerCoherence: &metal.ProbeLayerCoherence{Layer: 3, KeyCoherence: 0.1, ValueCoherence: 0.2, CrossAlignment: 0.3, KVCoupling: 0.4, HeadEntropy: 0.5, PhaseLock: 0.6},
		RouterDecision: &metal.ProbeRouterDecision{Layer: 4, TokenID: 7, ExpertIDs: []int{8, 9}, Weights: []float32{0.25, 0.75}, Temperature: 0.8},
		Residual:       &metal.ProbeResidualSummary{Layer: 5, Mean: 0.1, Variance: 0.2, RMS: 0.3, L2Norm: 0.4, MaxAbs: 0.5},
		Cache:          &metal.ProbeCachePressure{PromptTokens: 10, GeneratedTokens: 2, LayerCount: 6, CacheTokens: 12, ProcessedTokens: 14, MaxCacheTokens: 20, Utilization: 0.6, Rotating: true},
		Memory:         &metal.ProbeMemoryPressure{ActiveBytes: 100, PeakBytes: 200, CacheBytes: 50},
		Training:       &metal.ProbeTraining{Step: 6, Epoch: 1, Loss: 0.9, LearningRate: 0.01, GradNorm: 0.3},
	})
	if got.Token == nil || got.Logits == nil || got.SelectedHeads == nil || got.RouterDecision == nil || got.Training == nil {
		t.Fatalf("probe event = %+v, want all nested payloads", got)
	}
	if got.Meta["scope"] != "unit" || got.Logits.Top[0].TokenID != 4 || got.Cache == nil || !got.Cache.Rotating {
		t.Fatalf("probe event = %+v, want cloned meta/logits/cache", got)
	}
	got.Meta["scope"] = "changed"
	got.Logits.Meta["logits"] = "changed"
	if meta["scope"] != "unit" || logitMeta["logits"] != "kept" {
		t.Fatal("probe conversion leaked metadata map mutation")
	}
	if toProbeLogits(nil) != nil || cloneProbeMeta(nil) != nil {
		t.Fatal("empty probe helpers should return nil")
	}
}

func TestSpinePromptChunksToString_Good(t *testing.T) {
	chunks := func(yield func(string) bool) {
		for _, s := range []string{"a", "b", "c"} {
			if !yield(s) {
				return
			}
		}
	}
	if PromptChunksToString(chunks) != "abc" || PromptChunksToString(nil) != "" {
		t.Fatal("PromptChunksToString returned unexpected string")
	}
}
