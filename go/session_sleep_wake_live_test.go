// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestSessionSleepWakeRoundTrip_LiveModel pins the wake->append->generate
// seam on a real model with three greedy arms that must agree byte-for-byte:
//
//	A one-shot:   Prefill(story+cont)                          -> Generate
//	B append:     Prefill(story) + AppendPrompt(cont)          -> Generate
//	C round-trip: Prefill(story) + Sleep + Wake + Append(cont) -> Generate
//
// B diverging from A means the append seam is broken independent of any
// state machinery; C diverging from B isolates the sleep/wake restore path.
//
//	go test -tags model_eval -run TestSessionSleepWakeRoundTrip -count=1 dappco.re/go/mlx
func TestSessionSleepWakeRoundTrip_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	const story = "Story: The lighthouse keeper was called Snider, and his lamp burned a strange teal colour. Every night he polished the brass."
	const cont = " The keeper's name was"
	ctx := context.Background()

	gen := func(label string, s *ModelSession) string {
		t.Helper()
		text, err := s.Generate(WithMaxTokens(8), WithTemperature(0))
		if err != nil {
			t.Fatalf("%s: Generate: %v", label, err)
		}
		t.Logf("%s -> %q", label, text)
		return text
	}

	// Arm A — one-shot prefill, the known-good shape.
	oneShot, err := m.NewSession()
	if err != nil {
		t.Fatalf("A: NewSession: %v", err)
	}
	defer oneShot.Close()
	if err := oneShot.Prefill(story + cont); err != nil {
		t.Fatalf("A: Prefill: %v", err)
	}
	want := gen("A one-shot", oneShot)
	if want == "" {
		t.Fatalf("A one-shot generated nothing — baseline broken, cannot attribute")
	}
	if !core.Contains(want, "Snider") {
		t.Logf("A one-shot did not name the keeper (%q) — continuing, arms must still agree", want)
	}

	// Arm B — append seam, no state machinery.
	appended, err := m.NewSession()
	if err != nil {
		t.Fatalf("B: NewSession: %v", err)
	}
	defer appended.Close()
	if err := appended.Prefill(story); err != nil {
		t.Fatalf("B: Prefill: %v", err)
	}
	if err := appended.AppendPrompt(cont); err != nil {
		t.Fatalf("B: AppendPrompt: %v", err)
	}
	gotB := gen("B append", appended)

	// Arm C — full sleep/wake round-trip through an in-memory store.
	src, err := m.NewSession()
	if err != nil {
		t.Fatalf("C: NewSession: %v", err)
	}
	defer src.Close()
	if err := src.Prefill(story); err != nil {
		t.Fatalf("C: Prefill: %v", err)
	}
	store := state.NewInMemoryStore(nil)
	sleep, err := src.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://test/roundtrip", Title: "roundtrip"})
	if err != nil {
		t.Fatalf("C: Sleep: %v", err)
	}
	woken, err := m.NewSession()
	if err != nil {
		t.Fatalf("C: NewSession(wake): %v", err)
	}
	defer woken.Close()
	if _, err := woken.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI, LoadOptions: kv.LoadOptions{RawKVOnly: true}}); err != nil {
		t.Fatalf("C: Wake: %v", err)
	}
	if err := woken.AppendPrompt(cont); err != nil {
		t.Fatalf("C: AppendPrompt: %v", err)
	}
	gotC := gen("C round-trip", woken)

	// Arm D — direct snapshot capture/restore, no store and no block codec.
	// D agreeing with B while C diverges pins the block-streaming codec; D
	// diverging too pins CaptureKV/RestoreKV itself.
	srcD, err := m.NewSession()
	if err != nil {
		t.Fatalf("D: NewSession: %v", err)
	}
	defer srcD.Close()
	if err := srcD.Prefill(story); err != nil {
		t.Fatalf("D: Prefill: %v", err)
	}
	snapshot, err := srcD.CaptureKV()
	if err != nil {
		t.Fatalf("D: CaptureKV: %v", err)
	}
	restored, err := m.NewSessionFromKV(snapshot)
	if err != nil {
		t.Fatalf("D: NewSessionFromKV: %v", err)
	}
	defer restored.Close()
	if err := restored.AppendPrompt(cont); err != nil {
		t.Fatalf("D: AppendPrompt: %v", err)
	}
	gotD := gen("D snapshot", restored)

	if gotB != want {
		t.Errorf("append seam diverged from one-shot:\n  A %q\n  B %q", want, gotB)
	}
	if gotC != gotB {
		t.Errorf("sleep/wake round-trip diverged from append:\n  B %q\n  C %q", gotB, gotC)
	}
	if gotD != gotB {
		t.Errorf("direct snapshot round-trip diverged from append:\n  B %q\n  D %q", gotB, gotD)
	}
}
