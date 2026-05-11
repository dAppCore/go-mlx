// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/probe"
)

// These tests cover the mlx-root probe.go shim. The canonical
// algorithmic coverage lives in go-mlx/go/probe/probe_test.go; here we
// verify the alias surface + the mlx-specific GenerateOption helpers.

func TestProbeAliases_PointAtProbePackage_Good(t *testing.T) {
	// Type aliases are identical types in Go's type system, so this
	// assignment compiles only if the alias is wired through.
	var event ProbeEvent = probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 7}}
	if event.Kind != ProbeEventToken {
		t.Fatalf("Kind = %q, want %q", event.Kind, ProbeEventToken)
	}
	if event.Token.ID != 7 {
		t.Fatalf("Token.ID = %d, want 7", event.Token.ID)
	}
}

func TestProbeEventConstants_PreservedAtMlxRoot_Good(t *testing.T) {
	cases := []struct {
		got, want ProbeEventKind
	}{
		{ProbeEventToken, "token"},
		{ProbeEventLogits, "logits"},
		{ProbeEventEntropy, "entropy"},
		{ProbeEventSelectedHeads, "selected_heads"},
		{ProbeEventLayerCoherence, "layer_coherence"},
		{ProbeEventRouterDecision, "router_decision"},
		{ProbeEventExpertResidency, "expert_residency"},
		{ProbeEventResidual, "residual_summary"},
		{ProbeEventCachePressure, "cache_pressure"},
		{ProbeEventMemoryPressure, "memory_pressure"},
		{ProbeEventTraining, "training"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Fatalf("constant = %q, want %q", c.got, c.want)
		}
	}
}

func TestProbePhaseConstants_PreservedAtMlxRoot_Good(t *testing.T) {
	if ProbePhasePrefill != "prefill" || ProbePhaseDecode != "decode" || ProbePhaseTraining != "training" {
		t.Fatalf("phase constants drifted: %q %q %q", ProbePhasePrefill, ProbePhaseDecode, ProbePhaseTraining)
	}
}

func TestExpertResidencyAction_AliasIdentity_Good(t *testing.T) {
	// Cross-package equality between the mlx-root alias and the canonical
	// probe-package constant — proves the alias wires the same type.
	if ExpertResidencyActionPageIn != probe.ExpertResidencyActionPageIn {
		t.Fatal("ExpertResidencyAction alias drifted from probe package")
	}
}

func TestNewProbeBusAndRecorder_Wiring_Good(t *testing.T) {
	rec := NewProbeRecorder()
	bus := NewProbeBus(rec)
	bus.EmitProbe(ProbeEvent{Kind: ProbeEventToken, Token: &ProbeToken{ID: 1}})
	events := rec.Events()
	if len(events) != 1 || events[0].Kind != ProbeEventToken || events[0].Token.ID != 1 {
		t.Fatalf("events = %+v", events)
	}
}

func TestWithProbeSink_SetsConfigField_Good(t *testing.T) {
	rec := NewProbeRecorder()
	var cfg GenerateConfig
	WithProbeSink(rec)(&cfg)
	if cfg.ProbeSink == nil {
		t.Fatal("ProbeSink not set by WithProbeSink")
	}
	cfg.ProbeSink.EmitProbe(ProbeEvent{Kind: ProbeEventToken})
	if len(rec.Events()) != 1 {
		t.Fatal("ProbeSink not wired to recorder")
	}
}

func TestWithProbeCallback_NilIsNoOp_Ugly(t *testing.T) {
	var cfg GenerateConfig
	WithProbeCallback(nil)(&cfg)
	if cfg.ProbeSink != nil {
		t.Fatal("WithProbeCallback(nil) installed a sink")
	}
}

func TestWithProbeCallback_DispatchesEvent_Good(t *testing.T) {
	var got ProbeEvent
	var cfg GenerateConfig
	WithProbeCallback(func(e ProbeEvent) { got = e })(&cfg)
	if cfg.ProbeSink == nil {
		t.Fatal("WithProbeCallback(non-nil) did not install sink")
	}
	cfg.ProbeSink.EmitProbe(ProbeEvent{Kind: ProbeEventLogits, Step: 4})
	if got.Kind != ProbeEventLogits || got.Step != 4 {
		t.Fatalf("got = %+v", got)
	}
}

func TestProbeSinkFunc_AdaptsClosure_Good(t *testing.T) {
	called := false
	var sink ProbeSink = ProbeSinkFunc(func(_ ProbeEvent) { called = true })
	sink.EmitProbe(ProbeEvent{Kind: ProbeEventToken})
	if !called {
		t.Fatal("ProbeSinkFunc did not dispatch")
	}
}
