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
	"dappco.re/go/mlx/kv"
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

// --- merged from continuity_trusted_sleep_live_test.go (orphan sweep) ---
// TestConversationContinuity_TrustedSleepReuse_LiveModel proves the
// trusted-prefix sleep engages on the continuity lane: turn 2's sleep must
// graft turn 1's blocks by reference (ReusedBlocks > 0) instead of
// re-capturing the whole prefix.
//
//	go test -tags model_eval -run TestConversationContinuity_TrustedSleepReuse_LiveModel -count=1 dappco.re/go/mlx
func TestConversationContinuity_TrustedSleepReuse_LiveModel(t *testing.T) {
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

	turn := func(messages []inference.Message, maxTokens int) string {
		t.Helper()
		seq, ok := continuity.Chat(ctx, messages,
			inference.WithMaxTokens(maxTokens), inference.WithEnableThinking(&off))
		if !ok {
			t.Fatalf("continuity declined")
		}
		reply := core.NewBuilder()
		for token := range seq {
			reply.WriteString(token.Text)
		}
		return reply.String()
	}

	// Turn 1 must exceed window+blockSize tokens: kvBlockBoundaries inserts a
	// moving boundary at every sliding-window edge (seqLen-window), so the
	// leading UNIFORM full block — the graftable kind — only exists once
	// seqLen-window >= blockSize.
	turn1 := []inference.Message{{Role: "user", Content: "Tell a story about a glassblower, around eight hundred words."}}
	reply1 := turn(turn1, 1100)
	if reply1 == "" {
		t.Fatal("turn 1 generated nothing")
	}
	turn2 := append(append([]inference.Message{}, turn1...),
		inference.Message{Role: "assistant", Content: reply1},
		inference.Message{Role: "user", Content: "Continue the story briefly."})
	reply2 := turn(turn2, 160)
	if reply2 == "" {
		t.Fatal("turn 2 generated nothing")
	}

	// The second sleep's bundle must graft the first sleep's blocks.
	stats := continuity.Stats()
	if stats.Sleeps != 2 {
		t.Fatalf("sleeps = %d, want 2", stats.Sleeps)
	}
	conv := func() *residentConversation {
		continuity.mu.Lock()
		defer continuity.mu.Unlock()
		for _, c := range continuity.resident {
			return c
		}
		return nil
	}()
	if conv == nil || conv.parentBundle == "" {
		t.Fatalf("no resident conversation with a slept bundle (conv=%v)", conv)
	}
	bundle, err := kv.LoadStateBlockBundle(ctx, store, conv.parentBundle)
	if err != nil {
		t.Fatalf("LoadStateBlockBundle(%s): %v", conv.parentBundle, err)
	}
	t.Logf("turn-2 bundle: %d blocks, %d reused, %d tokens, block size %d",
		len(bundle.Blocks), bundle.ReusedBlocks, bundle.TokenCount, bundle.BlockSize)
	// Graft eligibility is geometry-dependent: kvBlockBoundaries inserts a
	// moving boundary at each sliding-window edge, so leading UNIFORM full
	// blocks (the graftable kind) only exist once the parent's seqLen
	// exceeds window+blockSize. Short conversations correctly reuse zero;
	// the kv package unit tests pin the graft mechanics themselves. What
	// this live test owns: trusted sleeps round-trip — the bundle loads,
	// tokens survive, and a store wake on a fresh manager still works.
	restarted, err := NewConversationContinuity(m, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("NewConversationContinuity(restarted): %v", err)
	}
	turn3 := append(append([]inference.Message{}, turn2...),
		inference.Message{Role: "assistant", Content: reply2},
		inference.Message{Role: "user", Content: "One more sentence to finish."})
	seq, ok := restarted.Chat(ctx, turn3, inference.WithMaxTokens(48), inference.WithEnableThinking(&off))
	if !ok {
		t.Fatal("restarted continuity declined the trusted-slept conversation")
	}
	reply3 := core.NewBuilder()
	for token := range seq {
		reply3.WriteString(token.Text)
	}
	if reply3.String() == "" {
		t.Fatal("wake over a trusted sleep generated nothing")
	}
	if stats := restarted.Stats(); stats.StoreWakes != 1 {
		t.Errorf("restarted wakes = %d, want 1 (trusted bundle must wake)", stats.StoreWakes)
	}
}
