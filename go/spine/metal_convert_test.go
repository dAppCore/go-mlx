// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
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

// --- ToMetalGenerateConfig ---

func TestMetalConvert_ToMetalGenerateConfig_Good(t *testing.T) {
	// Field-name divergence is the only non-trivial thing to assert: the
	// spine GenerationClearCache{,Interval} names map onto the metal
	// ClearCache / ClearCacheInterval names. A blind copy would drop
	// these; assert the rename plus representative scalars + a slice.
	cfg := GenerateConfig{
		MaxTokens:                    128,
		Temperature:                  0.7,
		TopK:                         40,
		StopTokens:                   []int32{2},
		GenerationClearCache:         true,
		GenerationClearCacheInterval: 16,
		ThinkingBudget:               64,
	}
	got := ToMetalGenerateConfig(cfg)
	if got.MaxTokens != 128 || got.Temperature != 0.7 || got.TopK != 40 {
		t.Fatalf("scalars = %+v, want MaxTokens=128 Temp=0.7 TopK=40", got)
	}
	if !got.ClearCache || got.ClearCacheInterval != 16 {
		t.Fatalf("ClearCache=%v interval=%d, want true/16 (field rename)", got.ClearCache, got.ClearCacheInterval)
	}
	if got.ThinkingBudget != 64 {
		t.Fatalf("ThinkingBudget = %d, want 64", got.ThinkingBudget)
	}
	if len(got.StopTokens) != 1 || got.StopTokens[0] != 2 {
		t.Fatalf("StopTokens = %v, want [2]", got.StopTokens)
	}
}

func TestMetalConvert_ToMetalGenerateConfig_Bad(t *testing.T) {
	// Zero config in → zero config out, and crucially a nil ProbeSink in
	// must NOT manufacture a non-nil metal sink (the steady-state path).
	got := ToMetalGenerateConfig(GenerateConfig{})
	if got.ProbeSink != nil {
		t.Fatal("zero GenerateConfig produced a non-nil metal ProbeSink")
	}
	if got.MaxTokens != 0 || got.Temperature != 0 {
		t.Fatalf("zero config = %+v, want all-zero scalars", got)
	}
}

func TestMetalConvert_ToMetalGenerateConfig_Ugly(t *testing.T) {
	// A non-nil ProbeSink must survive into the metal config as a non-nil
	// adapter (the trace path), distinct from the nil steady-state.
	got := ToMetalGenerateConfig(GenerateConfig{ProbeSink: spyProbeSink{}})
	if got.ProbeSink == nil {
		t.Fatal("non-nil ProbeSink dropped during ToMetalGenerateConfig")
	}
}

// --- ToMetalProbeSink + EmitProbe (exercised through the adapter) ---

// spyProbeSink records the converted probe.Event it receives so the test
// can assert the metal→probe EmitProbe conversion ran end to end.
type spyProbeSink struct {
	got *probe.Event
}

func (s spyProbeSink) EmitProbe(e probe.Event) {
	if s.got != nil {
		*s.got = e
	}
}

func TestMetalConvert_ToMetalProbeSink_Good(t *testing.T) {
	// Drive a metal.ProbeEvent through the adapter and assert the spy
	// received the CONVERTED probe.Event — this covers ToMetalProbeSink
	// (the wrap) and metalProbeSinkAdapter.EmitProbe (the per-event
	// conversion + forward) with real behaviour.
	var seen probe.Event
	sink := ToMetalProbeSink(spyProbeSink{got: &seen})
	if sink == nil {
		t.Fatal("ToMetalProbeSink(non-nil) returned nil")
	}
	sink.EmitProbe(metal.ProbeEvent{
		Kind:  metal.ProbeEventToken,
		Phase: metal.ProbePhaseDecode,
		Step:  9,
		Token: &metal.ProbeToken{ID: 7, Text: "tok"},
	})
	if seen.Step != 9 || seen.Token == nil || seen.Token.ID != 7 || seen.Token.Text != "tok" {
		t.Fatalf("forwarded event = %+v, want Step=9 Token{ID:7,Text:tok}", seen)
	}
}

func TestMetalConvert_ToMetalProbeSink_Bad(t *testing.T) {
	// Nil sink in → nil sink out: the steady-state Generate call carries
	// no sink and must not allocate an adapter.
	if ToMetalProbeSink(nil) != nil {
		t.Fatal("ToMetalProbeSink(nil) != nil")
	}
}
