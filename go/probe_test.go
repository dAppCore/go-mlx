// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

func TestProbeRecorder_RecordsDefensiveCopies_Good(t *testing.T) {
	recorder := NewProbeRecorder()
	event := ProbeEvent{
		Kind:  ProbeEventLogits,
		Phase: ProbePhaseDecode,
		Step:  3,
		Token: &ProbeToken{
			ID:              7,
			Text:            "answer",
			PromptTokens:    11,
			GeneratedTokens: 2,
		},
		Logits: &ProbeLogits{
			Shape:      []int32{1, 4},
			VocabSize:  4,
			MaxTokenID: 7,
			MaxLogit:   4.5,
			Top:        []ProbeLogit{{TokenID: 7, Logit: 4.5, Probability: 0.75}},
		},
		Cache: &ProbeCachePressure{
			LayerCount:      2,
			CacheTokens:     16,
			ProcessedTokens: 18,
		},
		Meta: map[string]string{"source": "test"},
	}

	recorder.EmitProbe(event)
	event.Token.Text = "mutated"
	event.Logits.Shape[0] = 99
	event.Logits.Top[0].Logit = -1
	event.Meta["source"] = "mutated"

	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("Events() len = %d, want 1", len(events))
	}
	if events[0].Token.Text != "answer" {
		t.Fatalf("recorded token text = %q, want answer", events[0].Token.Text)
	}
	if events[0].Logits.Shape[0] != 1 {
		t.Fatalf("recorded logits shape = %v, want [1 4]", events[0].Logits.Shape)
	}
	if events[0].Logits.Top[0].Logit != 4.5 {
		t.Fatalf("recorded top logit = %f, want 4.5", events[0].Logits.Top[0].Logit)
	}
	if events[0].Meta["source"] != "test" {
		t.Fatalf("recorded meta source = %q, want test", events[0].Meta["source"])
	}

	events[0].Logits.Top[0].TokenID = 99
	again := recorder.Events()
	if again[0].Logits.Top[0].TokenID != 7 {
		t.Fatalf("Events() returned aliased top logits: %+v", again[0].Logits.Top)
	}
}

func TestProbeSinkFunc_Good(t *testing.T) {
	called := false
	ProbeSinkFunc(func(event ProbeEvent) {
		called = event.Kind == ProbeEventMemoryPressure
	}).EmitProbe(ProbeEvent{Kind: ProbeEventMemoryPressure})

	if !called {
		t.Fatal("ProbeSinkFunc did not emit event")
	}
}

func TestProbeSinkFunc_Nil_Bad(t *testing.T) {
	var sink ProbeSinkFunc

	sink.EmitProbe(ProbeEvent{Kind: ProbeEventToken})
}

func TestProbeBus_Fanout_Good(t *testing.T) {
	first := NewProbeRecorder()
	second := NewProbeRecorder()
	bus := NewProbeBus(first)
	bus.Add(second)

	bus.EmitProbe(ProbeEvent{
		Kind:  ProbeEventTraining,
		Phase: ProbePhaseTraining,
		Training: &ProbeTraining{
			Step: 13,
			Loss: 0.125,
		},
	})

	if got := len(first.Events()); got != 1 {
		t.Fatalf("first recorder events = %d, want 1", got)
	}
	events := second.Events()
	if len(events) != 1 {
		t.Fatalf("second recorder events = %d, want 1", len(events))
	}
	if events[0].Training == nil || events[0].Training.Step != 13 || events[0].Training.Loss != 0.125 {
		t.Fatalf("training event = %+v", events[0])
	}
}

func TestProbeBus_FanoutDefensiveCopy_Ugly(t *testing.T) {
	recorder := NewProbeRecorder()
	bus := NewProbeBus(
		ProbeSinkFunc(func(event ProbeEvent) {
			event.Training.Loss = 9
		}),
		recorder,
	)

	bus.EmitProbe(ProbeEvent{
		Kind:     ProbeEventTraining,
		Phase:    ProbePhaseTraining,
		Training: &ProbeTraining{Step: 1, Loss: 0.5},
	})

	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("events len = %d, want 1", len(events))
	}
	if events[0].Training == nil || events[0].Training.Loss != 0.5 {
		t.Fatalf("fanout leaked mutation into recorder: %+v", events[0])
	}
}

func TestProbeOptionsAndClonePayloads_Ugly(t *testing.T) {
	var cfg GenerateConfig
	WithProbeCallback(nil)(&cfg)
	if cfg.ProbeSink != nil {
		t.Fatalf("nil callback configured sink: %+v", cfg.ProbeSink)
	}
	called := false
	WithProbeCallback(func(event ProbeEvent) {
		called = event.Kind == ProbeEventRouterDecision
	})(&cfg)
	cfg.ProbeSink.EmitProbe(ProbeEvent{Kind: ProbeEventRouterDecision})
	if !called {
		t.Fatal("probe callback was not invoked")
	}

	event := cloneProbeEvent(ProbeEvent{
		Kind:           ProbeEventSelectedHeads,
		SelectedHeads:  &ProbeHeadSelection{Heads: []int{1, 2}, Scores: []float64{0.25, 0.75}},
		LayerCoherence: &ProbeLayerCoherence{Layer: 2, KeyCoherence: 0.5},
		RouterDecision: &ProbeRouterDecision{ExpertIDs: []int{3}, Weights: []float32{0.9}},
		ExpertResidency: &ProbeExpertResidency{
			Action:    ExpertResidencyActionPageIn,
			ExpertIDs: []int{5},
		},
		Residual: &ProbeResidualSummary{Layer: 1, RMS: 0.2},
		Memory:   &ProbeMemoryPressure{ActiveBytes: 10},
	})
	event.SelectedHeads.Heads[0] = 9
	event.RouterDecision.ExpertIDs[0] = 8
	event.ExpertResidency.ExpertIDs[0] = 7
	if event.LayerCoherence.Layer != 2 || event.Residual.RMS != 0.2 || event.Memory.ActiveBytes != 10 {
		t.Fatalf("cloned scalar payloads = %+v", event)
	}
}
