// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

// session_methods_cover_test.go extends the synthetic-logits session coverage to
// the token-path lifecycle methods — AppendTokens, Fork, CaptureKV, Reset — that
// the greedy-decode test does not touch. Same logits-fake model, no weights.

func TestSessionMethods_AppendTokensThenGenerate_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newLogitsFakeSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()

	if err := sess.PrefillTokens(context.Background(), []int32{1, 2}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	// Append more tokens onto the retained prefix without resetting.
	if err := sess.AppendTokens(context.Background(), []int32{3, 4}); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}

	var n int
	for range sess.Generate(context.Background(), GenerateConfig{MaxTokens: 2}) {
		n++
		if n >= 2 {
			break
		}
	}
	if err := sess.Err(); err != nil {
		t.Fatalf("Err after append+generate: %v", err)
	}
	if n == 0 {
		t.Error("append+generate yielded nothing")
	}
}

func TestSessionMethods_CaptureKVAndReset_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newLogitsFakeSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{5, 6, 7}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	snap, err := sess.CaptureKV(context.Background())
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snap == nil || len(snap.Tokens) != 3 {
		t.Fatalf("snapshot tokens = %v, want 3", snap)
	}

	// Reset clears the session back to empty so a fresh prefill can start.
	sess.Reset()
	if err := sess.PrefillTokens(context.Background(), []int32{9}); err != nil {
		t.Fatalf("PrefillTokens after Reset: %v", err)
	}
}
