// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestConversationContinuity_LiveModel proves the no-prompt-replay loop on a
// real model across all three turn paths: fresh prefill, RAM-resident
// continuation, and store wake on a fresh manager (the serve-restart case).
// Recall of turn-one facts in later turns proves the state carried — the
// model never re-reads its prior text.
//
//	go test -tags model_eval -run TestConversationContinuity_LiveModel -count=1 dappco.re/go/mlx
func TestConversationContinuity_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	store := state.NewInMemoryStore(nil)
	continuity, err := NewConversationContinuity(m, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity: %v", err)
	}
	ctx := context.Background()
	off := false

	turn := func(label string, cc *ConversationContinuity, messages []inference.Message) string {
		t.Helper()
		seq, ok := cc.Chat(ctx, messages,
			inference.WithMaxTokens(48), inference.WithEnableThinking(&off))
		if !ok {
			t.Fatalf("%s: continuity declined", label)
		}
		reply := core.NewBuilder()
		for token := range seq {
			reply.WriteString(token.Text)
		}
		t.Logf("%s -> %q", label, reply.String())
		return reply.String()
	}

	// Turn 1 — fresh conversation, facts planted.
	turn1 := []inference.Message{{Role: "user", Content: "The lighthouse keeper is called Snider and his lamp burns teal. Acknowledge in one short sentence."}}
	reply1 := turn(`turn 1 (fresh)`, continuity, turn1)
	if reply1 == "" {
		t.Fatalf("turn 1 generated nothing")
	}

	// Turn 2 — RAM-resident continuation; recall proves the state carried.
	turn2 := append(append([]inference.Message{}, turn1...),
		inference.Message{Role: "assistant", Content: reply1},
		inference.Message{Role: "user", Content: "What is the keeper's name and the lamp colour? Answer in one short sentence."})
	reply2 := turn(`turn 2 (resident)`, continuity, turn2)
	if !core.Contains(reply2, "Snider") || !core.Contains(reply2, "teal") {
		t.Errorf("turn 2 did not recall the facts: %q", reply2)
	}

	stats := continuity.Stats()
	if stats.FreshConversations != 1 || stats.ResidentTurns != 1 || stats.StoreWakes != 0 {
		t.Errorf("manager paths = %+v, want fresh=1 resident=1 wakes=0", stats)
	}
	if stats.Sleeps != 2 {
		t.Errorf("sleeps = %d, want 2 (one per turn)", stats.Sleeps)
	}

	// Turn 3 — a FRESH manager over the SAME store: the serve-restart case.
	// The conversation must wake from durable state, not re-prefill.
	restarted, err := NewConversationContinuity(m, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity(restarted): %v", err)
	}
	turn3 := append(append([]inference.Message{}, turn2...),
		inference.Message{Role: "assistant", Content: reply2},
		inference.Message{Role: "user", Content: "Once more: name and colour, three words."})
	reply3 := turn(`turn 3 (store wake)`, restarted, turn3)
	if !core.Contains(reply3, "Snider") || !core.Contains(reply3, "teal") {
		t.Errorf("turn 3 did not recall across the restart: %q", reply3)
	}
	restartStats := restarted.Stats()
	if restartStats.StoreWakes != 1 || restartStats.FreshConversations != 0 {
		t.Errorf("restarted manager paths = %+v, want wakes=1 fresh=0", restartStats)
	}
}
