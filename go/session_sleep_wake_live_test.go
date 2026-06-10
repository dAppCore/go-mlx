// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"slices"
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
		if p := m.Metrics().CacheProfile; p != nil {
			t.Logf("%s caches: full=%d rot=%d fixed=%d paged=%d quant=%d unknown=%d local=%d global=%d localTok=%d/%d globalTok=%d/%d procTok=%d leak=%v",
				label, p.FullCaches, p.RotatingCaches, p.FixedCaches, p.PagedCaches, p.QuantizedCaches, p.UnknownCaches,
				p.LocalCaches, p.GlobalCaches, p.MaxLocalTokens, p.MaxLocalCapacity, p.MaxGlobalTokens, p.MaxGlobalCapacity,
				p.MaxProcessedTokens, p.LocalWindowLeaked)
		}
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

	// Arm E — tensor fidelity: the woken session's caches must capture
	// byte-identically to the source session's. Any differing layer names the
	// corrupt tensor before generation obscures it.
	srcSnap, err := src.CaptureKV()
	if err != nil {
		t.Fatalf("E: CaptureKV(src): %v", err)
	}
	wokeSnap, err := woken.CaptureKV()
	if err != nil {
		t.Fatalf("E: CaptureKV(woken): %v", err)
	}
	diffKVSnapshots(t, srcSnap, wokeSnap)

	// Arm F — codec field drift: reconstruct the FULL snapshot from the stored
	// blocks (the same decode the wake feeds on) and diff it against the source
	// capture PRE-restore — capture normalises fields post-restore, so this is
	// the only probe that can see SeqLen/offset/shape drift in the codec output.
	if idx, idxErr := agent.LoadStateIndex(ctx, store, sleep.IndexURI); idxErr != nil {
		t.Errorf("F: LoadStateIndex: %v", idxErr)
	} else if decoded, _, loadErr := agent.LoadPrefixFromStateIndex(ctx, store, idx, sleep.EntryURI, kv.LoadOptions{RawKVOnly: true}); loadErr != nil {
		t.Errorf("F: LoadPrefixFromStateIndex: %v", loadErr)
	} else {
		t.Logf("F: codec snapshot: seqlen=%d offset=%d layers=%d heads=%d headdim=%d logits=%d tokens=%d",
			decoded.SeqLen, decoded.TokenOffset, len(decoded.Layers), decoded.NumHeads, decoded.HeadDim, len(decoded.Logits), len(decoded.Tokens))
		t.Logf("F: src   snapshot: seqlen=%d offset=%d layers=%d heads=%d headdim=%d logits=%d tokens=%d",
			srcSnap.SeqLen, srcSnap.TokenOffset, len(srcSnap.Layers), srcSnap.NumHeads, srcSnap.HeadDim, len(srcSnap.Logits), len(srcSnap.Tokens))

		// Arm G — the codec's own decoded snapshot through D's PROVEN restore
		// lane. G matching B clears the codec content entirely and pins the bug
		// inside restoreKVBlocksLocked's per-block assembly; G failing means the
		// codec content is subtly wrong despite matching fields.
		viaSnapshot, gErr := m.NewSessionFromKV(decoded)
		if gErr != nil {
			t.Errorf("G: NewSessionFromKV(decoded): %v", gErr)
		} else {
			defer viaSnapshot.Close()
			if err := viaSnapshot.AppendPrompt(cont); err != nil {
				t.Errorf("G: AppendPrompt: %v", err)
			} else if gotG := gen("G codec->snapshot-lane", viaSnapshot); gotG != want {
				t.Errorf("G codec-content through the proven lane diverged:\n  A %q\n  G %q", want, gotG)
			}
		}
	}

	// Split C: generate straight off the woken state (first token samples the
	// RESTORED logits — the one field no other probe compares), with the source
	// session generating directly as its control.
	gotC0 := gen("C0 src direct", src)
	gotC1 := gen("C1 wake direct", woken)
	if gotC1 != gotC0 {
		t.Errorf("wake-direct diverged from src-direct (restored logits suspect):\n  C0 %q\n  C1 %q", gotC0, gotC1)
	}

	// Fresh wake for the append lane, untouched by the direct generation above.
	woken2, err := m.NewSession()
	if err != nil {
		t.Fatalf("C2: NewSession: %v", err)
	}
	defer woken2.Close()
	if _, err := woken2.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI, LoadOptions: kv.LoadOptions{RawKVOnly: true}}); err != nil {
		t.Fatalf("C2: Wake: %v", err)
	}
	if err := woken2.AppendPrompt(cont); err != nil {
		t.Fatalf("C2: AppendPrompt: %v", err)
	}
	gotC := gen("C2 wake+append", woken2)

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

// diffKVSnapshots reports, per layer, every field where the woken capture
// differs from the source capture — naming the corrupt tensors directly.
func diffKVSnapshots(t *testing.T, want, got *kv.Snapshot) {
	t.Helper()
	if want == nil || got == nil {
		t.Errorf("E: nil snapshot: src=%v woken=%v", want == nil, got == nil)
		return
	}
	if len(want.Tokens) != len(got.Tokens) {
		t.Errorf("E: tokens: src %d woken %d", len(want.Tokens), len(got.Tokens))
	}
	if want.TokenOffset != got.TokenOffset {
		t.Errorf("E: token offset: src %d woken %d", want.TokenOffset, got.TokenOffset)
	}
	if len(want.Layers) != len(got.Layers) {
		t.Errorf("E: layer count: src %d woken %d", len(want.Layers), len(got.Layers))
		return
	}
	bad := 0
	for i := range want.Layers {
		w, g := &want.Layers[i], &got.Layers[i]
		var fields []string
		if w.CacheMode != g.CacheMode {
			fields = append(fields, core.Sprintf("mode %q->%q", w.CacheMode, g.CacheMode))
		}
		if w.KeyDType != g.KeyDType {
			fields = append(fields, core.Sprintf("kdtype %q->%q", w.KeyDType, g.KeyDType))
		}
		if w.ValueDType != g.ValueDType {
			fields = append(fields, core.Sprintf("vdtype %q->%q", w.ValueDType, g.ValueDType))
		}
		if !slices.Equal(w.KeyShape, g.KeyShape) {
			fields = append(fields, core.Sprintf("kshape %v->%v", w.KeyShape, g.KeyShape))
		}
		if !slices.Equal(w.ValueShape, g.ValueShape) {
			fields = append(fields, core.Sprintf("vshape %v->%v", w.ValueShape, g.ValueShape))
		}
		if d := firstByteDiff(w.KeyBytes, g.KeyBytes); d >= 0 {
			fields = append(fields, core.Sprintf("kbytes len %d->%d first-diff @%d", len(w.KeyBytes), len(g.KeyBytes), d))
		}
		if d := firstByteDiff(w.ValueBytes, g.ValueBytes); d >= 0 {
			fields = append(fields, core.Sprintf("vbytes len %d->%d first-diff @%d", len(w.ValueBytes), len(g.ValueBytes), d))
		}
		if len(w.TurboQuantPayloads) != len(g.TurboQuantPayloads) {
			fields = append(fields, core.Sprintf("turbo payloads %d->%d", len(w.TurboQuantPayloads), len(g.TurboQuantPayloads)))
		}
		if len(w.Heads) != len(g.Heads) {
			fields = append(fields, core.Sprintf("heads %d->%d", len(w.Heads), len(g.Heads)))
		} else {
			for h := range w.Heads {
				if d := firstFloatDiff(w.Heads[h].Key, g.Heads[h].Key); d >= 0 {
					fields = append(fields, core.Sprintf("head %d key len %d->%d first-diff @%d (%g vs %g)",
						h, len(w.Heads[h].Key), len(g.Heads[h].Key), d, floatAt(w.Heads[h].Key, d), floatAt(g.Heads[h].Key, d)))
					break
				}
				if d := firstFloatDiff(w.Heads[h].Value, g.Heads[h].Value); d >= 0 {
					fields = append(fields, core.Sprintf("head %d value len %d->%d first-diff @%d (%g vs %g)",
						h, len(w.Heads[h].Value), len(g.Heads[h].Value), d, floatAt(w.Heads[h].Value, d), floatAt(g.Heads[h].Value, d)))
					break
				}
			}
		}
		if len(fields) > 0 {
			bad++
			if bad <= 6 {
				t.Errorf("E: layer %d (cache %d, mode %s): %v", w.Layer, w.CacheIndex, w.CacheMode, fields)
			}
		}
	}
	if bad > 0 {
		t.Errorf("E: kv fidelity: %d/%d layers differ", bad, len(want.Layers))
	} else {
		t.Logf("E: kv fidelity: all %d layers byte-identical", len(want.Layers))
	}
}

// firstByteDiff returns the first index where a and b differ, or -1 when
// byte-identical (length difference counts as a diff at min length).
func firstByteDiff(a, b []byte) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// firstFloatDiff returns the first index where a and b differ, or -1 when
// identical (length difference counts as a diff at min length).
func firstFloatDiff(a, b []float32) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func floatAt(s []float32, i int) float32 {
	if i >= 0 && i < len(s) {
		return s[i]
	}
	return 0
}
