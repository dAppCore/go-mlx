// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/agent"
)

// TestModel_WakeForkLiveModel drives the Model-level agent-memory lifecycle
// aliases on the live model: a session is prefilled and slept to an in-memory
// store, then Model.Wake (the WakeAgentMemory alias) and Model.ForkState (the
// backend-neutral fork contract) both restore from that durable bundle. These
// two entry points are 0% without a real model — every other path nil-guards.
func TestModel_WakeForkLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)
	ctx := context.Background()

	// Prefill a source session and sleep it to a store.
	src, err := m.NewSession()
	if err != nil {
		t.Fatalf("NewSession(src): %v", err)
	}
	defer src.Close()
	if err := src.Prefill("The archivist's name is Wren and the vault key is brass."); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	store := state.NewInMemoryStore(nil)
	sleep, err := src.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI: "mlx://test/wakefork",
		Title:    "wakefork",
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory: %v", err)
	}

	// Model.Wake — the lifecycle alias that creates a session and restores the
	// durable prefix in one call.
	woken, wakeReport, err := m.Wake(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	})
	if err != nil {
		t.Fatalf("Wake: %v", err)
	}
	defer woken.Close()
	if wakeReport == nil || wakeReport.PrefixTokens == 0 {
		t.Errorf("Wake report missing prefix tokens: %+v", wakeReport)
	}
	t.Logf("Wake: strategy=%q prefix=%d blocks=%d", wakeReport.RestoreStrategy, wakeReport.PrefixTokens, wakeReport.BlocksRead)

	// Model.ForkState — the backend-neutral fork contract over the same bundle.
	forked, forkResult, err := m.ForkState(ctx, inference.AgentMemoryWakeRequest{
		Store:    store,
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	})
	if err != nil {
		t.Fatalf("ForkState: %v", err)
	}
	if forked == nil {
		t.Fatal("ForkState returned nil session")
	}
	// The concrete fork is a *ModelSession; close it to release its KV.
	if closer, ok := forked.(interface{ Close() error }); ok {
		defer closer.Close()
	}
	if forkResult == nil {
		t.Errorf("ForkState returned nil result")
	}
	t.Logf("ForkState restored a session from %s", sleep.EntryURI)
}

// TestModel_ForkStateBadStore drives the ForkState type-assert guard: a store
// that is not a state.Store returns the fork-needs-store sentinel. No model
// work — the guard fires before any restore.
func TestModel_ForkStateBadStore(t *testing.T) {
	m := requireLiveE2BModel(t)
	_, _, err := m.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store: "not-a-store",
	})
	if err == nil {
		t.Fatal("ForkState with a non-store Store should error")
	}
}

// TestConversationContinuity_LiveChat drives the conversation-continuity loop on
// the live model under metal_runtime: Chat (the no-prompt-replay turn router),
// acquire (conversation match/create), and finishTurn (sleep-back) — all 0%
// without a real model and otherwise only proven under the model_eval suite.
func TestConversationContinuity_LiveChat(t *testing.T) {
	m := requireLiveE2BModel(t)
	store := state.NewInMemoryStore(nil)
	cc, err := NewConversationContinuity(m, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity: %v", err)
	}
	ctx := context.Background()
	off := false

	drive := func(label string, messages []inference.Message) string {
		t.Helper()
		seq, ok := cc.Chat(ctx, messages,
			inference.WithMaxTokens(32), inference.WithTemperature(0), inference.WithEnableThinking(&off))
		if !ok {
			t.Fatalf("%s: continuity declined the turn", label)
		}
		reply := core.NewBuilder()
		for token := range seq {
			reply.WriteString(token.Text)
		}
		t.Logf("%s -> %q", label, reply.String())
		return reply.String()
	}

	// Turn 1 — a fresh conversation, acquire creates it, finishTurn sleeps it.
	turn1 := []inference.Message{{Role: "user", Content: "Remember the codeword is mistral. Acknowledge briefly."}}
	reply1 := drive("turn 1 (fresh)", turn1)
	if reply1 == "" {
		t.Fatal("turn 1 generated nothing")
	}

	// Turn 2 — the same conversation continues RAM-resident; acquire matches the
	// existing prefix instead of creating, finishTurn sleeps again. The prefix
	// match keys on the prior turns verbatim, so turn 1's ACTUAL reply must be
	// fed back as the assistant message.
	turn2 := append(append([]inference.Message{}, turn1...),
		inference.Message{Role: "assistant", Content: reply1},
		inference.Message{Role: "user", Content: "What was the codeword? One word."})
	drive("turn 2 (resident)", turn2)

	stats := cc.Stats()
	if stats.FreshConversations != 1 {
		t.Errorf("FreshConversations=%d, want 1", stats.FreshConversations)
	}
	if stats.ResidentTurns != 1 {
		t.Errorf("ResidentTurns=%d, want 1", stats.ResidentTurns)
	}
	if stats.Sleeps != 2 {
		t.Errorf("Sleeps=%d, want 2 (one per turn)", stats.Sleeps)
	}
}
