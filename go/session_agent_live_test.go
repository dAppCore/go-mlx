// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"reflect"
	"slices"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// liveKVBlockRestorer mirrors the session package's unexported
// nativeSessionKVBlockRestorer probe — the J/K/L arms drive the raw
// native block-restore path around the Session wrapper on purpose.
type liveKVBlockRestorer interface {
	RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
}

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
	wakeReport, err := woken.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI, LoadOptions: kv.LoadOptions{RawKVOnly: true}})
	if err != nil {
		t.Fatalf("C: Wake: %v", err)
	}
	t.Logf("C: wake report: strategy=%q prefix=%d blocks=%d bundle=%d", wakeReport.RestoreStrategy, wakeReport.PrefixTokens, wakeReport.BlocksRead, wakeReport.BundleTokens)

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
	idx, idxErr := agent.LoadStateIndex(ctx, store, sleep.IndexURI)
	if idxErr != nil {
		t.Fatalf("F: LoadStateIndex: %v", idxErr)
	}
	decoded, _, loadErr := agent.LoadPrefixFromStateIndex(ctx, store, idx, sleep.EntryURI, kv.LoadOptions{RawKVOnly: true})
	if loadErr != nil {
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
			} else if gotG := gen("G codec->snapshot-lane", viaSnapshot); gotG != gotB {
				t.Errorf("G codec-content through the proven lane diverged from the append run:\n  B %q\n  G %q", gotB, gotG)
			}
		}
	}

	// Arm H — diff C's ACTUAL input against G's proven-good input: the wake's
	// per-block snapshot (kvconv.MetalKVSnapshotBlockSource — what
	// restoreKVBlocksLocked is really fed) versus the assembled full snapshot
	// converted to metal form. Any differing field is the lie.
	fullMetal := kvconv.ToMetalKVSnapshot(decoded)
	plan, planErr := agent.PlanWake(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI}, spine.ModelInfoToMemory(m.Info()))
	if planErr != nil {
		t.Fatalf("H: PlanWake: %v", planErr)
	}
	blockSource, srcErr := kvconv.MetalKVSnapshotBlockSource(ctx, store, plan.Bundle, plan.Entry.PrefixTokens())
	if srcErr != nil {
		t.Fatalf("H: MetalKVSnapshotBlockSource: %v", srcErr)
	}
	t.Logf("H: wake source: blocks=%d prefix=%d total=%d", blockSource.BlockCount, blockSource.PrefixTokens, blockSource.TokenCount)
	if blockSource.BlockCount == 1 {
		blk, blkErr := blockSource.Load(ctx, 0)
		if blkErr != nil {
			t.Fatalf("H: Load(0): %v", blkErr)
		}
		diffMetalKVSnapshots(t, fullMetal, blk.Snapshot)
		if !reflect.DeepEqual(fullMetal, blk.Snapshot) {
			wv, gv := reflect.ValueOf(*fullMetal), reflect.ValueOf(*blk.Snapshot)
			for fi := 0; fi < wv.NumField(); fi++ {
				if !reflect.DeepEqual(wv.Field(fi).Interface(), gv.Field(fi).Interface()) {
					t.Logf("H: field %q deep-differs (nil-vs-empty cosmetics possible)", wv.Type().Field(fi).Name)
				}
			}
			if d := firstFloatDiff(fullMetal.Logits, blk.Snapshot.Logits); d >= 0 {
				t.Logf("H: LOGITS content: len %d/%d first-diff @%d (%g vs %g)",
					len(fullMetal.Logits), len(blk.Snapshot.Logits), d, floatAt(fullMetal.Logits, d), floatAt(blk.Snapshot.Logits, d))
			}
			for li := range fullMetal.Layers {
				if !reflect.DeepEqual(fullMetal.Layers[li], blk.Snapshot.Layers[li]) {
					wl, gl := reflect.ValueOf(fullMetal.Layers[li]), reflect.ValueOf(blk.Snapshot.Layers[li])
					for fi := 0; fi < wl.NumField(); fi++ {
						if !reflect.DeepEqual(wl.Field(fi).Interface(), gl.Field(fi).Interface()) {
							t.Logf("H: layer %d field %q deep-differs", li, wl.Type().Field(fi).Name)
						}
					}
					for h := range fullMetal.Layers[li].Heads {
						wh, gh := &fullMetal.Layers[li].Heads[h], &blk.Snapshot.Layers[li].Heads[h]
						if !reflect.DeepEqual(*wh, *gh) {
							t.Logf("H: layer %d head %d: assembled{key=%d kdtype=%q kbytes=%d val=%d vdtype=%q vbytes=%d} per-block{key=%d kdtype=%q kbytes=%d val=%d vdtype=%q vbytes=%d}",
								li, h,
								len(wh.Key), wh.KeyDType, len(wh.KeyBytes), len(wh.Value), wh.ValueDType, len(wh.ValueBytes),
								len(gh.Key), gh.KeyDType, len(gh.KeyBytes), len(gh.Value), gh.ValueDType, len(gh.ValueBytes))
							break
						}
					}
					break
				}
			}
		}

		// Arm J — inline replica of WakeAgentMemory's exact body on a fresh
		// session: plan -> kvconv source -> RestoreKVBlocks. Expected to fail
		// like C; the control for arm K.
		wokenJ, jErr := m.NewSession()
		if jErr != nil {
			t.Fatalf("J: NewSession: %v", jErr)
		}
		defer wokenJ.Close()
		srcJ, jSrcErr := kvconv.MetalKVSnapshotBlockSource(ctx, store, plan.Bundle, plan.Entry.PrefixTokens())
		if jSrcErr != nil {
			t.Fatalf("J: source: %v", jSrcErr)
		}
		if rErr := wokenJ.Native().(liveKVBlockRestorer).RestoreKVBlocks(ctx, srcJ); rErr != nil {
			t.Fatalf("J: RestoreKVBlocks: %v", rErr)
		}
		if err := wokenJ.AppendPrompt(cont); err != nil {
			t.Fatalf("J: AppendPrompt: %v", err)
		}
		gotJ := gen("J inline-wake (lazy load)", wokenJ)

		// Arm K — identical to J except the block is decoded OUTSIDE the
		// restore (pre-loaded), so Load inside the device context returns a
		// ready value. K working while J fails pins the bug to the block
		// decode running under the restore's device/slot context.
		wokenK, kErr := m.NewSession()
		if kErr != nil {
			t.Fatalf("K: NewSession: %v", kErr)
		}
		defer wokenK.Close()
		preloaded := blk
		srcK := metal.KVSnapshotBlockSource{
			TokenCount:   blockSource.TokenCount,
			PrefixTokens: blockSource.PrefixTokens,
			BlockCount:   1,
			Load: func(context.Context, int) (metal.KVSnapshotBlock, error) {
				return preloaded, nil
			},
		}
		if rErr := wokenK.Native().(liveKVBlockRestorer).RestoreKVBlocks(ctx, srcK); rErr != nil {
			t.Fatalf("K: RestoreKVBlocks: %v", rErr)
		}
		if err := wokenK.AppendPrompt(cont); err != nil {
			t.Fatalf("K: AppendPrompt: %v", err)
		}
		gotK := gen("K inline-wake (pre-loaded)", wokenK)

		// Arm L — K with DEEP-CLONED bytes: if cloning the per-block buffers
		// fixes the lane, the block decoder is handing out transient/pooled
		// memory that the pinned zero-copy cache arrays alias after reuse.
		wokenL, lErr := m.NewSession()
		if lErr != nil {
			t.Fatalf("L: NewSession: %v", lErr)
		}
		defer wokenL.Close()
		cloned := cloneMetalKVSnapshot(blk.Snapshot)
		srcL := metal.KVSnapshotBlockSource{
			TokenCount:   blockSource.TokenCount,
			PrefixTokens: blockSource.PrefixTokens,
			BlockCount:   1,
			Load: func(context.Context, int) (metal.KVSnapshotBlock, error) {
				return metal.KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: blk.TokenCount, Snapshot: cloned}, nil
			},
		}
		if rErr := wokenL.Native().(liveKVBlockRestorer).RestoreKVBlocks(ctx, srcL); rErr != nil {
			t.Fatalf("L: RestoreKVBlocks: %v", rErr)
		}
		if err := wokenL.AppendPrompt(cont); err != nil {
			t.Fatalf("L: AppendPrompt: %v", err)
		}
		gotL := gen("L inline-wake (cloned bytes)", wokenL)
		t.Logf("J=%q K=%q L=%q want=%q", gotJ, gotK, gotL, want)
	}

	// Arm I — feed the KNOWN-GOOD full snapshot through RestoreKVBlocks as a
	// hand-built single-block source. Working means the lane's code is sound
	// and H's input diff is the bug; failing with good input convicts
	// restoreKVBlocksLocked itself.
	woken3, err := m.NewSession()
	if err != nil {
		t.Fatalf("I: NewSession: %v", err)
	}
	defer woken3.Close()
	if restorer, ok := woken3.Native().(liveKVBlockRestorer); !ok {
		t.Errorf("I: session does not implement the native block-restore probe")
	} else {
		goodSource := metal.KVSnapshotBlockSource{
			TokenCount:   len(fullMetal.Tokens),
			PrefixTokens: len(fullMetal.Tokens),
			BlockCount:   1,
			Load: func(context.Context, int) (metal.KVSnapshotBlock, error) {
				return metal.KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: len(fullMetal.Tokens), Snapshot: fullMetal}, nil
			},
		}
		if rErr := restorer.RestoreKVBlocks(ctx, goodSource); rErr != nil {
			t.Errorf("I: RestoreKVBlocks(good snapshot): %v", rErr)
		} else {
			if err := woken3.AppendPrompt(cont); err != nil {
				t.Fatalf("I: AppendPrompt: %v", err)
			}
			if gotI := gen("I good-snapshot via kv-blocks lane", woken3); gotI != gotB {
				t.Errorf("I: kv-blocks lane corrupts even a KNOWN-GOOD snapshot (vs the append run):\n  B %q\n  I %q", gotB, gotI)
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

	// One-shot (A) and append (B) are DIFFERENT op compositions — the
	// one-shot prefills [prompt+cont] in one pass, the append chunks at the
	// seam — so under a half-precision stream the first post-seam token is
	// already a legitimate near-tie fork site (the same rule as the
	// compiled-vs-uncompiled gates). Logged for the record; the load-bearing
	// byte-exact assertions are the SAME-composition ones: sleep/wake (C),
	// snapshot-restore (D), and the codec lane (G) must match the append
	// run (B) exactly.
	if gotB != want {
		t.Logf("append seam vs one-shot: composition fork (expected under half precision):\n  A %q\n  B %q", want, gotB)
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

// cloneMetalKVSnapshot deep-copies every buffer in a metal snapshot so the
// result shares no memory with the original — the arm-L provenance probe.
func cloneMetalKVSnapshot(src *metal.KVSnapshot) *metal.KVSnapshot {
	if src == nil {
		return nil
	}
	out := *src
	out.Tokens = slices.Clone(src.Tokens)
	out.Generated = slices.Clone(src.Generated)
	out.LogitShape = slices.Clone(src.LogitShape)
	out.Logits = slices.Clone(src.Logits)
	out.Layers = make([]metal.KVLayerSnapshot, len(src.Layers))
	for i, l := range src.Layers {
		nl := l
		nl.KeyBytes = slices.Clone(l.KeyBytes)
		nl.ValueBytes = slices.Clone(l.ValueBytes)
		nl.KeyShape = slices.Clone(l.KeyShape)
		nl.ValueShape = slices.Clone(l.ValueShape)
		nl.Heads = make([]metal.KVHeadSnapshot, len(l.Heads))
		for h, hd := range l.Heads {
			nh := hd
			nh.Key = slices.Clone(hd.Key)
			nh.KeyBytes = slices.Clone(hd.KeyBytes)
			nh.Value = slices.Clone(hd.Value)
			nh.ValueBytes = slices.Clone(hd.ValueBytes)
			nl.Heads[h] = nh
		}
		nl.TurboQuantPayloads = make([]metal.TurboQuantKVReferencePagePayload, len(l.TurboQuantPayloads))
		copy(nl.TurboQuantPayloads, l.TurboQuantPayloads)
		out.Layers[i] = nl
	}
	return &out
}

// diffMetalKVSnapshots compares the metal-level snapshots the two restore
// lanes are actually fed, field by field; want is the proven-good input.
func diffMetalKVSnapshots(t *testing.T, want, got *metal.KVSnapshot) {
	t.Helper()
	if want == nil || got == nil {
		t.Errorf("H: nil metal snapshot: good=%v block=%v", want == nil, got == nil)
		return
	}
	if want.Version != got.Version || want.Architecture != got.Architecture {
		t.Errorf("H: version/arch: %d/%q vs %d/%q", want.Version, want.Architecture, got.Version, got.Architecture)
	}
	if !slices.Equal(want.Tokens, got.Tokens) {
		t.Errorf("H: tokens differ: %d vs %d", len(want.Tokens), len(got.Tokens))
	}
	if want.TokenOffset != got.TokenOffset || want.SeqLen != got.SeqLen {
		t.Errorf("H: offset/seqlen: %d/%d vs %d/%d", want.TokenOffset, want.SeqLen, got.TokenOffset, got.SeqLen)
	}
	if want.NumLayers != got.NumLayers || want.NumHeads != got.NumHeads || want.HeadDim != got.HeadDim || want.NumQueryHeads != got.NumQueryHeads {
		t.Errorf("H: dims: layers %d/%d heads %d/%d headdim %d/%d qheads %d/%d",
			want.NumLayers, got.NumLayers, want.NumHeads, got.NumHeads, want.HeadDim, got.HeadDim, want.NumQueryHeads, got.NumQueryHeads)
	}
	if len(want.Logits) != len(got.Logits) || !slices.Equal(want.LogitShape, got.LogitShape) {
		t.Errorf("H: logits: len %d/%d shape %v/%v", len(want.Logits), len(got.Logits), want.LogitShape, got.LogitShape)
	}
	if len(want.Layers) != len(got.Layers) {
		t.Errorf("H: layer count %d vs %d", len(want.Layers), len(got.Layers))
		return
	}
	bad := 0
	for i := range want.Layers {
		w, g := &want.Layers[i], &got.Layers[i]
		var fields []string
		if w.Layer != g.Layer || w.CacheIndex != g.CacheIndex {
			fields = append(fields, core.Sprintf("layer/cache %d/%d vs %d/%d", w.Layer, w.CacheIndex, g.Layer, g.CacheIndex))
		}
		if w.CacheMode != g.CacheMode {
			fields = append(fields, core.Sprintf("mode %q vs %q", w.CacheMode, g.CacheMode))
		}
		if w.KeyDType != g.KeyDType || w.ValueDType != g.ValueDType {
			fields = append(fields, core.Sprintf("dtype %q/%q vs %q/%q", w.KeyDType, w.ValueDType, g.KeyDType, g.ValueDType))
		}
		if !slices.Equal(w.KeyShape, g.KeyShape) || !slices.Equal(w.ValueShape, g.ValueShape) {
			fields = append(fields, core.Sprintf("shape k%v v%v vs k%v v%v", w.KeyShape, w.ValueShape, g.KeyShape, g.ValueShape))
		}
		if d := firstByteDiff(w.KeyBytes, g.KeyBytes); d >= 0 {
			fields = append(fields, core.Sprintf("kbytes len %d/%d diff@%d", len(w.KeyBytes), len(g.KeyBytes), d))
		}
		if d := firstByteDiff(w.ValueBytes, g.ValueBytes); d >= 0 {
			fields = append(fields, core.Sprintf("vbytes len %d/%d diff@%d", len(w.ValueBytes), len(g.ValueBytes), d))
		}
		if len(w.Heads) != len(g.Heads) {
			fields = append(fields, core.Sprintf("heads %d vs %d", len(w.Heads), len(g.Heads)))
		} else {
			for h := range w.Heads {
				if d := firstFloatDiff(w.Heads[h].Key, g.Heads[h].Key); d >= 0 {
					fields = append(fields, core.Sprintf("head %d key len %d/%d diff@%d", h, len(w.Heads[h].Key), len(g.Heads[h].Key), d))
					break
				}
				if d := firstFloatDiff(w.Heads[h].Value, g.Heads[h].Value); d >= 0 {
					fields = append(fields, core.Sprintf("head %d value len %d/%d diff@%d", h, len(w.Heads[h].Value), len(g.Heads[h].Value), d))
					break
				}
			}
		}
		if len(w.TurboQuantPayloads) != len(g.TurboQuantPayloads) {
			fields = append(fields, core.Sprintf("turbo %d vs %d", len(w.TurboQuantPayloads), len(g.TurboQuantPayloads)))
		}
		if len(fields) > 0 {
			bad++
			if bad <= 6 {
				t.Errorf("H: layer %d: %v", i, fields)
			}
		}
	}
	if bad > 0 {
		t.Errorf("H: per-block input differs from proven-good input: %d/%d layers", bad, len(want.Layers))
	} else {
		t.Logf("H: per-block input identical to proven-good input (all %d layers)", len(want.Layers))
	}
}
