// SPDX-Licence-Identifier: EUPL-1.2

package probe

import (
	"sync"
	"testing"
)

func TestRecorder_RecordsDefensiveCopies_Good(t *testing.T) {
	recorder := NewRecorder()
	event := Event{
		Kind:  KindLogits,
		Phase: PhaseDecode,
		Step:  3,
		Token: &Token{
			ID: 7, Text: "answer", PromptTokens: 11, GeneratedTokens: 2,
		},
		Logits: &Logits{
			Shape: []int32{1, 4}, VocabSize: 4,
			MaxTokenID: 7, MaxLogit: 4.5,
			Top: []Logit{{TokenID: 7, Logit: 4.5, Probability: 0.75}},
		},
		Cache: &CachePressure{
			LayerCount: 2, CacheTokens: 16, ProcessedTokens: 18,
		},
		Meta: map[string]string{"prompt_id": "abc"},
	}
	recorder.EmitProbe(event)
	// Mutate caller-side payloads — should not surface in recorded copy.
	event.Token.Text = "mutated"
	event.Logits.Top[0].Probability = 0.0
	event.Cache.ProcessedTokens = 99
	event.Meta["prompt_id"] = "changed"
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("Events() len = %d, want 1", len(events))
	}
	got := events[0]
	if got.Token.Text != "answer" {
		t.Fatalf("Token.Text = %q, want answer (defensive copy)", got.Token.Text)
	}
	if got.Logits.Top[0].Probability != 0.75 {
		t.Fatalf("Logits.Top probability = %v, want 0.75 (defensive copy)", got.Logits.Top[0].Probability)
	}
	if got.Cache.ProcessedTokens != 18 {
		t.Fatalf("Cache.ProcessedTokens = %d, want 18 (defensive copy)", got.Cache.ProcessedTokens)
	}
	if got.Meta["prompt_id"] != "abc" {
		t.Fatalf("Meta[prompt_id] = %q, want abc (defensive copy)", got.Meta["prompt_id"])
	}
}

func TestRecorder_NilReceiver_Ugly(t *testing.T) {
	var r *Recorder
	r.EmitProbe(Event{}) // must not panic
	if got := r.Events(); got != nil {
		t.Fatalf("nil Recorder.Events() = %v, want nil", got)
	}
}

func TestBus_FansOutToAllSinks_Good(t *testing.T) {
	rec1 := NewRecorder()
	rec2 := NewRecorder()
	bus := NewBus(rec1, rec2)
	bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	if len(rec1.Events()) != 1 || len(rec2.Events()) != 1 {
		t.Fatalf("fanout = rec1:%d rec2:%d, want 1 each", len(rec1.Events()), len(rec2.Events()))
	}
}

// TestBus_OwnedSink_EventsAreDeepClonedOnRead verifies the
// owned-sink path: the Bus skips on-emit cloning, but Recorder.Events()
// returns deep-cloned events so consumers can never alias storage.
// Even if the underlying recorder storage shares pointers with the
// bus-delivered event (per the relaxed owned-sink contract), the
// snapshot returned by Events() is fully detached.
func TestBus_OwnedSink_EventsAreDeepClonedOnRead_Good(t *testing.T) {
	rec := NewRecorder()
	bus := NewBus(rec)
	bus.EmitProbe(Event{
		Kind:  KindToken,
		Token: &Token{ID: 7, Text: "answer"},
		Meta:  map[string]string{"k": "v"},
	})
	first := rec.Events()
	second := rec.Events()
	if len(first) != 1 || len(second) != 1 {
		t.Fatalf("events len first=%d second=%d, want 1 each", len(first), len(second))
	}
	if first[0].Token == second[0].Token {
		t.Fatal("Events() returned aliased Token pointers across calls")
	}
	// Mutating first[] snapshot must not affect second[] snapshot.
	first[0].Token.ID = 99
	first[0].Meta["k"] = "mutated"
	if second[0].Token.ID != 7 {
		t.Fatalf("second snapshot Token.ID = %d, want 7 (snapshots aliased)", second[0].Token.ID)
	}
	if second[0].Meta["k"] != "v" {
		t.Fatalf("second snapshot Meta[k] = %q, want v (snapshots aliased)", second[0].Meta["k"])
	}
}

func TestBus_AddNilIgnored_Ugly(t *testing.T) {
	bus := NewBus()
	bus.Add(nil) // must not panic; no sink added
	rec := NewRecorder()
	bus.Add(rec)
	bus.EmitProbe(Event{Kind: KindToken})
	if len(rec.Events()) != 1 {
		t.Fatalf("rec.Events() len = %d, want 1", len(rec.Events()))
	}
}

func TestBus_NilReceiver_Ugly(t *testing.T) {
	var b *Bus
	b.Add(NewRecorder()) // must not panic
	b.EmitProbe(Event{}) // must not panic
}

func TestSinkFunc_NilFuncIsSilent_Ugly(t *testing.T) {
	var f SinkFunc
	f.EmitProbe(Event{Kind: KindToken}) // must not panic
}

func TestSinkFunc_DispatchesToWrappedFunc_Good(t *testing.T) {
	var got Event
	f := SinkFunc(func(e Event) { got = e })
	f.EmitProbe(Event{Kind: KindRouterDecision, RouterDecision: &RouterDecision{Layer: 2}})
	if got.Kind != KindRouterDecision || got.RouterDecision == nil || got.RouterDecision.Layer != 2 {
		t.Fatalf("got = %+v", got)
	}
}

func TestBus_ConcurrentSafe_Good(t *testing.T) {
	bus := NewBus()
	rec := NewRecorder()
	bus.Add(rec)
	var wg sync.WaitGroup
	for range 100 {
		wg.Go(func() {
			bus.EmitProbe(Event{Kind: KindToken})
		})
	}
	wg.Wait()
	if got := len(rec.Events()); got != 100 {
		t.Fatalf("concurrent emit count = %d, want 100", got)
	}
}

func TestCloneEvent_DefensiveCopiesAllPayloads_Good(t *testing.T) {
	src := Event{
		Kind: KindLogits, Step: 1,
		Token:           &Token{ID: 1, Text: "x"},
		Logits:          &Logits{Shape: []int32{1, 2}, Top: []Logit{{TokenID: 1}}, Values: []float32{0.1}, Meta: map[string]string{"k": "v"}},
		SelectedHeads:   &HeadSelection{Heads: []int{0, 1}, Scores: []float64{0.5}},
		RouterDecision:  &RouterDecision{ExpertIDs: []int{0, 1}, Weights: []float32{0.5, 0.5}},
		ExpertResidency: &ExpertResidency{Action: ExpertResidencyActionPageIn, ExpertIDs: []int{0}},
		Meta:            map[string]string{"prompt": "p"},
	}
	out := CloneEvent(src)
	// Mutate originals.
	src.Token.Text = "mutated"
	src.Logits.Shape[0] = 99
	src.Logits.Top[0].TokenID = 99
	src.Logits.Values[0] = 9
	src.Logits.Meta["k"] = "z"
	src.SelectedHeads.Heads[0] = 99
	src.SelectedHeads.Scores[0] = 99
	src.RouterDecision.ExpertIDs[0] = 99
	src.RouterDecision.Weights[0] = 99
	src.ExpertResidency.ExpertIDs[0] = 99
	src.Meta["prompt"] = "mutated"
	if out.Token.Text != "x" {
		t.Fatal("CloneEvent shared Token")
	}
	if out.Logits.Shape[0] != 1 || out.Logits.Top[0].TokenID != 1 || out.Logits.Values[0] != 0.1 || out.Logits.Meta["k"] != "v" {
		t.Fatalf("CloneEvent shared Logits internals: %+v", out.Logits)
	}
	if out.SelectedHeads.Heads[0] != 0 || out.SelectedHeads.Scores[0] != 0.5 {
		t.Fatalf("CloneEvent shared SelectedHeads: %+v", out.SelectedHeads)
	}
	if out.RouterDecision.ExpertIDs[0] != 0 || out.RouterDecision.Weights[0] != 0.5 {
		t.Fatalf("CloneEvent shared RouterDecision: %+v", out.RouterDecision)
	}
	if out.ExpertResidency.ExpertIDs[0] != 0 {
		t.Fatalf("CloneEvent shared ExpertResidency: %+v", out.ExpertResidency)
	}
	if out.Meta["prompt"] != "p" {
		t.Fatalf("CloneEvent shared Meta: %+v", out.Meta)
	}
}

func TestCloneEvent_NilPayloadsPreserved_Ugly(t *testing.T) {
	src := Event{Kind: KindToken, Step: 1}
	out := CloneEvent(src)
	if out.Kind != KindToken || out.Step != 1 {
		t.Fatalf("CloneEvent lost scalar fields: %+v", out)
	}
	if out.Token != nil || out.Logits != nil || out.Entropy != nil {
		t.Fatalf("CloneEvent created phantom payload pointers: %+v", out)
	}
}

func TestExpertResidencyAction_ConstantsAreStrings_Good(t *testing.T) {
	cases := []struct {
		got, want ExpertResidencyAction
	}{
		{ExpertResidencyActionStartup, "startup"},
		{ExpertResidencyActionPageIn, "page_in"},
		{ExpertResidencyActionEvict, "evict"},
		{ExpertResidencyActionHit, "hit"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Fatalf("constant = %q, want %q", c.got, c.want)
		}
	}
}

func TestKindAndPhase_StringValues_Good(t *testing.T) {
	if KindToken != "token" || KindTraining != "training" || PhasePrefill != "prefill" {
		t.Fatal("constants do not have expected string values")
	}
}
