// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Extra coverage for conversation_continuity.go: the construction guards, the
// stats snapshot, the order-list maintenance, and the request-knob translation.
// The live turn paths (Chat → acquire → Prefill → GenerateStream → finishTurn)
// drive a real session and are exercised by the model_eval-gated live suite;
// here we cover everything reachable without a session.

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
)

func TestNewConversationContinuity_NilModel_Bad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	if _, err := NewConversationContinuity(nil, ConversationContinuityOptions{Store: store}); err == nil {
		t.Fatal("NewConversationContinuity(nil model) err = nil, want model-nil")
	}
}

func TestNewConversationContinuity_NilStore_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if _, err := NewConversationContinuity(model, ConversationContinuityOptions{}); err == nil {
		t.Fatal("NewConversationContinuity(nil store) err = nil, want store-nil")
	}
}

func TestNewConversationContinuity_DefaultsApplied_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	model := &Model{model: &fakeNativeModel{}}
	cc, err := NewConversationContinuity(model, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity error = %v", err)
	}
	if cc.max != 4 {
		t.Fatalf("default MaxResident = %d, want 4", cc.max)
	}
	if cc.prefix != "mlx://conversation/" {
		t.Fatalf("default EntryPrefix = %q, want mlx://conversation/", cc.prefix)
	}
	// A fresh manager reports zeroed stats.
	if (cc.Stats() != ContinuityStats{}) {
		t.Fatalf("fresh Stats = %+v, want zero", cc.Stats())
	}
}

func TestNewConversationContinuity_ExplicitOptions_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	model := &Model{model: &fakeNativeModel{}}
	cc, err := NewConversationContinuity(model, ConversationContinuityOptions{
		Store: store, MaxResident: 9, EntryPrefix: "x://conv/",
	})
	if err != nil {
		t.Fatalf("NewConversationContinuity error = %v", err)
	}
	if cc.max != 9 || cc.prefix != "x://conv/" {
		t.Fatalf("explicit options = max %d prefix %q, want 9/x://conv/", cc.max, cc.prefix)
	}
}

func TestConversationContinuity_RemoveOrderLocked_Good(t *testing.T) {
	cc := &ConversationContinuity{order: []string{"a", "b", "c"}}
	cc.removeOrderLocked("b")
	if len(cc.order) != 2 || cc.order[0] != "a" || cc.order[1] != "c" {
		t.Fatalf("removeOrderLocked(b) = %v, want [a c]", cc.order)
	}
	// Removing an absent key is a no-op.
	cc.removeOrderLocked("zzz")
	if len(cc.order) != 2 {
		t.Fatalf("removeOrderLocked(absent) changed order: %v", cc.order)
	}
}

func TestConversationContinuity_RootGenerateOptions_Good(t *testing.T) {
	cfg := inference.GenerateConfig{
		MaxTokens: 64, Temperature: 0.7, TopK: 40, TopP: 0.9,
		StopTokens: []int32{1, 2}, RepeatPenalty: 1.1, ThinkingBudget: 32,
	}
	applied := DefaultGenerateConfig()
	for _, opt := range rootGenerateOptions(cfg) {
		opt(&applied)
	}
	if applied.MaxTokens != 64 || applied.Temperature != 0.7 {
		t.Fatalf("rootGenerateOptions base = maxTokens %d temp %v", applied.MaxTokens, applied.Temperature)
	}
	if applied.TopK != 40 || applied.TopP != 0.9 || applied.RepeatPenalty != 1.1 {
		t.Fatalf("rootGenerateOptions sampler = topK %d topP %v rp %v", applied.TopK, applied.TopP, applied.RepeatPenalty)
	}
	if len(applied.StopTokens) != 2 || applied.ThinkingBudget != 32 {
		t.Fatalf("rootGenerateOptions stops/budget = %v / %d", applied.StopTokens, applied.ThinkingBudget)
	}
}

func TestConversationContinuity_RootGenerateOptions_OmitsUnset_Ugly(t *testing.T) {
	// RepeatPenalty == 1 is the no-op identity and must NOT emit an option;
	// zero topK/topP/stops likewise stay unset.
	cfg := inference.GenerateConfig{Temperature: 0.5, RepeatPenalty: 1}
	applied := GenerateConfig{}
	for _, opt := range rootGenerateOptions(cfg) {
		opt(&applied)
	}
	if applied.Temperature != 0.5 {
		t.Fatalf("Temperature = %v, want 0.5", applied.Temperature)
	}
	if applied.TopK != 0 || applied.TopP != 0 || applied.RepeatPenalty != 0 || len(applied.StopTokens) != 0 {
		t.Fatalf("unset knobs leaked: topK %d topP %v rp %v stops %v", applied.TopK, applied.TopP, applied.RepeatPenalty, applied.StopTokens)
	}
}

func TestEnableConversationContinuity_NotMetalAdapter_Bad(t *testing.T) {
	// A TextModel that is not the metal adapter is rejected.
	if _, err := EnableConversationContinuity(notAMetalAdapter{}, ConversationContinuityOptions{Store: memvid.NewInMemoryStore(nil)}); err == nil {
		t.Fatal("EnableConversationContinuity(non-adapter) err = nil, want adapter-type error")
	}
}

// notAMetalAdapter is an inference.TextModel that is not *metaladapter.
type notAMetalAdapter struct{ inference.TextModel }
