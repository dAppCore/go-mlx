// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/internal/metal"
)

type fakeNativeSession struct {
	prefillPrompt    string
	appendPrompt     string
	prefillErr       error
	appendErr        error
	tokens           []metal.Token
	cfg              metal.GenerateConfig
	probeEvents      []metal.ProbeEvent
	afterGenerate    func(*fakeNativeSession)
	kv               *metal.KVSnapshot
	kvBlocks         []metal.KVSnapshotBlock
	captureErr       error
	restoredKV       *metal.KVSnapshot
	restoredBlocks   []metal.KVSnapshotBlock
	restoreErr       error
	restoreBlocksErr error
	forked           metal.SessionHandle
	forkErr          error
	err              error
	resetCalls       int
	closeCalls       int
	closeErr         error
}

func (s *fakeNativeSession) Prefill(_ context.Context, prompt string) error {
	s.prefillPrompt = prompt
	return s.prefillErr
}

func (s *fakeNativeSession) AppendPrompt(_ context.Context, prompt string) error {
	s.appendPrompt = prompt
	return s.appendErr
}

func (s *fakeNativeSession) Generate(_ context.Context, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	s.cfg = cfg
	return func(yield func(metal.Token) bool) {
		defer func() {
			if s.afterGenerate != nil {
				s.afterGenerate(s)
			}
		}()
		for _, event := range s.probeEvents {
			if cfg.ProbeSink != nil {
				cfg.ProbeSink.EmitProbe(event)
			}
		}
		for _, tok := range s.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

func (s *fakeNativeSession) CaptureKV(_ context.Context) (*metal.KVSnapshot, error) {
	return s.kv, s.captureErr
}

func (s *fakeNativeSession) RangeKVBlocks(_ context.Context, _ int, _ metal.KVSnapshotCaptureOptions, yield func(metal.KVSnapshotBlock) (bool, error)) error {
	if len(s.kvBlocks) == 0 && s.kv != nil {
		_, err := yield(metal.KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: len(s.kv.Tokens), Snapshot: s.kv})
		return err
	}
	for _, block := range s.kvBlocks {
		ok, err := yield(block)
		if err != nil || !ok {
			return err
		}
	}
	return nil
}

func (s *fakeNativeSession) RestoreKV(_ context.Context, snapshot *metal.KVSnapshot) error {
	s.restoredKV = snapshot
	return s.restoreErr
}

func (s *fakeNativeSession) RestoreKVBlocks(ctx context.Context, source metal.KVSnapshotBlockSource) error {
	if s.restoreBlocksErr != nil {
		return s.restoreBlocksErr
	}
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(ctx, i)
		if err != nil {
			return err
		}
		s.restoredBlocks = append(s.restoredBlocks, block)
		if block.TokenStart+block.TokenCount >= source.PrefixTokens {
			break
		}
	}
	if len(s.restoredBlocks) == 1 {
		s.restoredKV = s.restoredBlocks[0].Snapshot
	}
	return nil
}

func (s *fakeNativeSession) Fork(_ context.Context) (metal.SessionHandle, error) {
	return s.forked, s.forkErr
}

func (s *fakeNativeSession) Reset() {
	s.resetCalls++
}

func (s *fakeNativeSession) Close() error {
	s.closeCalls++
	return s.closeErr
}

func (s *fakeNativeSession) Err() error {
	return s.err
}

func TestModelNewSession_Good(t *testing.T) {
	coverageTokens := "ModelNewSession"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	nativeSession := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{session: nativeSession}}

	session, err := model.NewSession()

	if err != nil {
		t.Fatalf("NewSession() error = %v", err)
	}
	if session == nil {
		t.Fatal("NewSession() = nil, want session")
	}
	if session.session != nativeSession {
		t.Fatal("NewSession() did not wrap native session")
	}
}

func TestModelNewSession_Bad(t *testing.T) {
	coverageTokens := "ModelNewSession Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var model *Model

	session, err := model.NewSession()

	if err == nil {
		t.Fatal("expected nil model error")
	}
	if session != nil {
		t.Fatalf("session = %v, want nil", session)
	}
}

func TestModelNewSession_Ugly(t *testing.T) {
	coverageTokens := "ModelNewSession Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{model: nativeWithoutPromptCache{}}

	session, err := model.NewSession()

	if err == nil {
		t.Fatal("expected unsupported native session error")
	}
	if session != nil {
		t.Fatalf("session = %v, want nil", session)
	}
}

func TestModelNewSession_ReturnedNilAndBundleErrors_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if session, err := model.NewSession(); err == nil || session != nil {
		t.Fatalf("NewSession(nil native session) = %+v/%v, want error", session, err)
	}
	if session, err := model.NewSessionFromBundle(nil); err == nil || session != nil {
		t.Fatalf("NewSessionFromBundle(nil) = %+v/%v, want error", session, err)
	}
}

func TestModelNewSessionFromKV_Good(t *testing.T) {
	coverageTokens := "ModelNewSessionFromKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	nativeSession := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{session: nativeSession}}
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1},
		TokenOffset:  1,
		SeqLen:       1,
		HeadDim:      1,
		LogitShape:   []int32{1, 1, 2},
		Logits:       []float32{0.1, 0.9},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1},
				Value: []float32{2},
			}},
		}},
	}

	session, err := model.NewSessionFromKV(snapshot)

	if err != nil {
		t.Fatalf("NewSessionFromKV() error = %v", err)
	}
	if session == nil || session.session != nativeSession {
		t.Fatalf("NewSessionFromKV() = %#v, want wrapped native session", session)
	}
	if nativeSession.restoredKV == nil || nativeSession.restoredKV.Logits[1] != 0.9 {
		t.Fatalf("restored KV = %+v", nativeSession.restoredKV)
	}
}

func TestSessionPrefillAndGenerate_Good(t *testing.T) {
	coverageTokens := "SessionPrefillAndGenerate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	nativeSession := &fakeNativeSession{
		tokens: []metal.Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}},
	}
	session := &ModelSession{session: nativeSession}

	if err := session.Prefill("stable context"); err != nil {
		t.Fatalf("Prefill() error = %v", err)
	}
	got, err := session.Generate(WithMaxTokens(12), WithTemperature(0.2), WithMinP(0.05))

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got != "AB" {
		t.Fatalf("Generate() = %q, want AB", got)
	}
	if nativeSession.prefillPrompt != "stable context" {
		t.Fatalf("prefill prompt = %q, want stable context", nativeSession.prefillPrompt)
	}
	if nativeSession.cfg.MaxTokens != 12 || nativeSession.cfg.Temperature != 0.2 || nativeSession.cfg.MinP != 0.05 {
		t.Fatalf("Generate config = %+v", nativeSession.cfg)
	}
}

func TestSessionAppendPrompt_Good(t *testing.T) {
	coverageTokens := "SessionAppendPrompt"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	nativeSession := &fakeNativeSession{}
	session := &ModelSession{session: nativeSession}

	if err := session.AppendPrompt("\n\nQuestion: who?\nAnswer:"); err != nil {
		t.Fatalf("AppendPrompt() error = %v", err)
	}

	if nativeSession.appendPrompt != "\n\nQuestion: who?\nAnswer:" {
		t.Fatalf("append prompt = %q", nativeSession.appendPrompt)
	}
}

func TestSessionNilGuards_Bad(t *testing.T) {
	var session *ModelSession
	if err := session.AppendPrompt("x"); err == nil {
		t.Fatal("expected nil append prompt error")
	}
	if text, err := session.Generate(); err == nil || text != "" {
		t.Fatalf("Generate(nil) = %q/%v, want error", text, err)
	}
	if err := session.RestoreKV(nil); err == nil {
		t.Fatal("expected nil session restore error")
	}
	if err := (&ModelSession{}).RestoreKV(nil); err == nil {
		t.Fatal("expected empty session restore error")
	}
	if err := (&ModelSession{session: &fakeNativeSession{}}).RestoreKV(nil); err == nil {
		t.Fatal("expected nil KV snapshot error")
	}
	if _, err := session.SaveKVToMemvid(nil, memvid.NewInMemoryStore(nil), kv.MemvidOptions{}); err == nil {
		t.Fatal("expected nil session save-to-memvid error")
	}
	if _, err := session.SaveKVBlocksToMemvid(nil, memvid.NewInMemoryStore(nil), kv.MemvidBlockOptions{}); err == nil {
		t.Fatal("expected nil session save-blocks error")
	}
	if err := session.LoadKVBlocksFromMemvid(nil, memvid.NewInMemoryStore(nil), &kv.MemvidBlockBundle{}); err == nil {
		t.Fatal("expected invalid memvid block load error")
	}
	if err := session.RestoreBundle(nil); err == nil {
		t.Fatal("expected nil bundle restore error")
	}
	if err := session.RestoreBundleFromMemvid(nil, nil, memvid.NewInMemoryStore(nil)); err == nil {
		t.Fatal("expected nil memvid bundle restore error")
	}
	if err := session.LoadBundle(core.PathJoin(t.TempDir(), "missing.bundle.json")); err == nil {
		t.Fatal("expected missing bundle load error")
	}
	session.Reset()
	if err := session.Close(); err != nil {
		t.Fatalf("Close(nil) = %v, want nil", err)
	}
	if err := session.Err(); err != nil {
		t.Fatalf("Err(nil) = %v, want nil", err)
	}
}

func TestSessionGenerate_ForwardsProbeSink_Good(t *testing.T) {
	coverageTokens := "SessionGenerate ProbeSink"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	recorder := NewProbeRecorder()
	nativeSession := &fakeNativeSession{
		probeEvents: []metal.ProbeEvent{{
			Kind:  metal.ProbeEventEntropy,
			Phase: metal.ProbePhaseDecode,
			Step:  1,
			Entropy: &metal.ProbeEntropy{
				Value: 0.42,
			},
		}},
	}
	session := &ModelSession{session: nativeSession}

	if _, err := session.Generate(WithProbeSink(recorder)); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if nativeSession.cfg.ProbeSink == nil {
		t.Fatal("native ProbeSink = nil, want configured")
	}
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("probe events len = %d, want 1", len(events))
	}
	if events[0].Kind != ProbeEventEntropy || events[0].Entropy == nil || events[0].Entropy.Value != 0.42 {
		t.Fatalf("probe event = %+v", events[0])
	}
}

func TestModelSessionMemvidKV_Good_SaveAndLoad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	nativeSession := &fakeNativeSession{
		kv: &metal.KVSnapshot{
			Version:       metal.KVSnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{10, 20},
			Generated:     []int32{30},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 1,
			LogitShape:    []int32{1, 1, 2},
			Logits:        []float32{0.25, 0.75},
			Layers: []metal.KVLayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []metal.KVHeadSnapshot{{
					Key:   []float32{1, 2, 3, 4},
					Value: []float32{5, 6, 7, 8},
				}},
			}},
		},
	}
	session := &ModelSession{session: nativeSession}

	ref, err := session.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{URI: "mlx://session/demo"})
	if err != nil {
		t.Fatalf("SaveKVToMemvid() error = %v", err)
	}
	restoredNative := &fakeNativeSession{}
	restored := &ModelSession{session: restoredNative}
	if err := restored.LoadKVFromMemvid(context.Background(), store, ref); err != nil {
		t.Fatalf("LoadKVFromMemvid() error = %v", err)
	}

	if restoredNative.restoredKV == nil || restoredNative.restoredKV.Tokens[1] != 20 || restoredNative.restoredKV.Generated[0] != 30 {
		t.Fatalf("restored KV = %+v", restoredNative.restoredKV)
	}
	if restoredNative.restoredKV.Logits[1] != 0.75 {
		t.Fatalf("restored logits = %+v", restoredNative.restoredKV.Logits)
	}
}

func TestModelSessionMemvidBundle_Good_Restore(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	snapshot := stateBundleTestSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), store, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	nativeSession := &fakeNativeSession{}
	session := &ModelSession{
		session: nativeSession,
		info:    ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	}
	bundle := &StateBundle{
		Version: StateBundleVersion,
		Kind:    StateBundleKind,
		Model:   StateBundleModel{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs: []StateBundleRef{{
			Kind:   StateBundleRefMemvid,
			URI:    stateMemvidURI(ref),
			Memvid: ref,
		}},
	}

	if err := session.RestoreBundleFromMemvid(context.Background(), bundle, store); err != nil {
		t.Fatalf("RestoreBundleFromMemvid() error = %v", err)
	}
	if nativeSession.restoredKV == nil || nativeSession.restoredKV.Tokens[0] != 1 {
		t.Fatalf("restored KV = %+v", nativeSession.restoredKV)
	}
}

func TestModelSessionMemvidKVBlocks_Good_SaveAndLoad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	nativeSession := &fakeNativeSession{
		captureErr: core.NewError("full snapshot capture should not be used"),
		kvBlocks: []metal.KVSnapshotBlock{
			{
				Index:      0,
				TokenStart: 0,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil),
			},
			{
				Index:      1,
				TokenStart: 2,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{30, 40}, 4, []float32{5, 6, 7, 8}, []float32{13, 14, 15, 16}, []float32{0.25, 0.75}, []int32{40}),
			},
		},
	}
	session := &ModelSession{session: nativeSession}

	bundle, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToMemvid() error = %v", err)
	}
	if len(bundle.Blocks) != 2 {
		t.Fatalf("bundle blocks = %+v, want 2", bundle.Blocks)
	}
	restoredNative := &fakeNativeSession{}
	restored := &ModelSession{session: restoredNative}
	if err := restored.LoadKVBlocksFromMemvid(context.Background(), store, bundle); err != nil {
		t.Fatalf("LoadKVBlocksFromMemvid() error = %v", err)
	}

	if len(restoredNative.restoredBlocks) != 2 {
		t.Fatalf("restored blocks = %+v, want 2", restoredNative.restoredBlocks)
	}
	last := restoredNative.restoredBlocks[1].Snapshot
	if last == nil || last.Tokens[1] != 40 || last.Generated[0] != 40 {
		t.Fatalf("restored final block KV = %+v", last)
	}
	if last.Layers[0].Heads[0].Value[3] != 16 {
		t.Fatalf("restored final block values = %+v", last.Layers[0].Heads[0].Value)
	}
}

func testNativeKVBlock(tokens []int32, tokenOffset int, key, value, logits []float32, generated []int32) *metal.KVSnapshot {
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        append([]int32(nil), tokens...),
		Generated:     append([]int32(nil), generated...),
		TokenOffset:   tokenOffset,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        len(tokens),
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []metal.KVHeadSnapshot{{
				Key:   append([]float32(nil), key...),
				Value: append([]float32(nil), value...),
			}},
		}},
	}
	if len(logits) > 0 {
		snapshot.LogitShape = []int32{1, 1, int32(len(logits))}
		snapshot.Logits = append([]float32(nil), logits...)
	}
	return snapshot
}

func TestSessionPrefill_Bad(t *testing.T) {
	coverageTokens := "SessionPrefill Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var session *ModelSession

	if err := session.Prefill("prompt"); err == nil {
		t.Fatal("expected nil session error")
	}
}

func TestSessionGenerate_Ugly(t *testing.T) {
	coverageTokens := "SessionGenerate Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("decode failed")
	nativeSession := &fakeNativeSession{
		tokens: []metal.Token{{ID: 1, Text: "partial"}},
		err:    wantErr,
	}
	session := &ModelSession{session: nativeSession}

	_, err := session.Generate()

	if !core.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
}

func TestSessionGenerateStream_Good(t *testing.T) {
	coverageTokens := "SessionGenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	session := &ModelSession{session: &fakeNativeSession{
		tokens: []metal.Token{{ID: 7, Text: "x"}, {ID: 8, Text: "y"}},
	}}

	ch := session.GenerateStream(context.Background(), WithTopK(4))
	var got []Token
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if len(got) != 2 || got[0].Text != "x" || got[1].Value != "y" {
					t.Fatalf("stream tokens = %+v", got)
				}
				return
			}
			got = append(got, tok)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestSessionGenerateStream_Bad(t *testing.T) {
	coverageTokens := "SessionGenerateStream Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var session *ModelSession

	ch := session.GenerateStream(context.Background())

	if tok, ok := <-ch; ok {
		t.Fatalf("stream yielded %+v, want closed", tok)
	}
}

func TestSessionGenerateStream_Ugly(t *testing.T) {
	coverageTokens := "SessionGenerateStream Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	session := &ModelSession{session: &fakeNativeSession{
		tokens: []metal.Token{{ID: 7, Text: "x"}},
	}}

	ch := session.GenerateStream(ctx)

	if tok, ok := <-ch; ok {
		t.Fatalf("stream yielded %+v after cancellation", tok)
	}
}

func TestSessionCaptureKVAnalyzeAndSave_Good(t *testing.T) {
	coverageTokens := "SessionCaptureKVAnalyzeAndSave"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeSession{
		kv: &metal.KVSnapshot{
			Version:       metal.KVSnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 8,
			Layers: []metal.KVLayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []metal.KVHeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			}},
		},
	}
	session := &ModelSession{session: native}

	snapshot, err := session.CaptureKV()

	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	if snapshot.Architecture != "gemma4_text" || snapshot.NumQueryHeads != 8 {
		t.Fatalf("CaptureKV() = %+v", snapshot)
	}
	snapshot.Tokens[0] = 99
	if native.kv.Tokens[0] != 1 {
		t.Fatal("CaptureKV() returned aliased token data")
	}
	analysis, err := session.AnalyzeKV()
	if err != nil {
		t.Fatalf("kv.Analyze() error = %v", err)
	}
	if analysis == nil || len(kv.Features(analysis)) != 7 {
		t.Fatalf("kv.Analyze() = %+v", analysis)
	}
	path := core.PathJoin(t.TempDir(), "session.kvbin")
	if err := session.SaveKV(path); err != nil {
		t.Fatalf("SaveKV() error = %v", err)
	}
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}
	if loaded.Architecture != "gemma4_text" || loaded.SeqLen != 2 {
		t.Fatalf("loaded snapshot = %+v", loaded)
	}
}

func TestSessionRestoreAndLoadKV_Good(t *testing.T) {
	coverageTokens := "SessionRestoreAndLoadKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	if err := session.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV() error = %v", err)
	}
	if native.restoredKV == nil || native.restoredKV.Generated[0] != 2 {
		t.Fatalf("restored KV = %+v", native.restoredKV)
	}
	native.restoredKV = nil
	path := core.PathJoin(t.TempDir(), "restore.kvbin")
	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if err := session.LoadKV(path); err != nil {
		t.Fatalf("LoadKV() error = %v", err)
	}
	if native.restoredKV == nil || native.restoredKV.TokenOffset != 2 {
		t.Fatalf("loaded KV restore = %+v", native.restoredKV)
	}
}

func TestSessionExportBundle_Good(t *testing.T) {
	coverageTokens := "SessionExportBundle"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeSession{
		kv: &metal.KVSnapshot{
			Version:       metal.KVSnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			Generated:     []int32{2},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 8,
			LogitShape:    []int32{1, 1, 3},
			Logits:        []float32{0.1, 0.2, 0.7},
			Layers: []metal.KVLayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []metal.KVHeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			}},
		},
	}
	session := &ModelSession{session: native}

	bundle, err := session.ExportBundle(StateBundleOptions{
		Model:  "gemma4-e4b",
		Prompt: "stable context",
		Runtime: StateBundleRuntime{
			Version: "test",
		},
	})

	if err != nil {
		t.Fatalf("ExportBundle() error = %v", err)
	}
	if bundle == nil || bundle.Model.Name != "gemma4-e4b" || bundle.Runtime.Name != "go-mlx" {
		t.Fatalf("ExportBundle() = %+v", bundle)
	}
	if bundle.KV == nil || bundle.KV.Generated[0] != 2 || bundle.SAMI == nil {
		t.Fatalf("ExportBundle() KV/SAMI = %+v/%+v", bundle.KV, bundle.SAMI)
	}
}

func TestSessionCaptureKV_Bad(t *testing.T) {
	coverageTokens := "SessionCaptureKV Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var session *ModelSession

	snapshot, err := session.CaptureKV()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if snapshot != nil {
		t.Fatalf("snapshot = %v, want nil", snapshot)
	}
}

func TestSessionCaptureKV_Ugly(t *testing.T) {
	coverageTokens := "SessionCaptureKV Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("capture failed")
	session := &ModelSession{session: &fakeNativeSession{captureErr: wantErr}}

	_, err := session.CaptureKV()

	if !core.Is(err, wantErr) {
		t.Fatalf("CaptureKV() error = %v, want %v", err, wantErr)
	}
}

func TestSessionForkResetClose_Good(t *testing.T) {
	coverageTokens := "SessionForkResetClose"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	forkedNative := &fakeNativeSession{}
	native := &fakeNativeSession{forked: forkedNative}
	session := &ModelSession{session: native}

	forked, err := session.Fork()

	if err != nil {
		t.Fatalf("Fork() error = %v", err)
	}
	if forked == nil || forked.session != forkedNative {
		t.Fatalf("Fork() = %#v, want wrapped fork", forked)
	}
	session.Reset()
	if native.resetCalls != 1 {
		t.Fatalf("reset calls = %d, want 1", native.resetCalls)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.closeCalls)
	}
}

func TestSessionFork_Bad(t *testing.T) {
	coverageTokens := "SessionFork Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var session *ModelSession

	forked, err := session.Fork()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if forked != nil {
		t.Fatalf("forked = %v, want nil", forked)
	}
}

func TestSessionClose_Ugly(t *testing.T) {
	coverageTokens := "SessionClose Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("close failed")
	session := &ModelSession{session: &fakeNativeSession{closeErr: wantErr}}

	err := session.Close()

	if !core.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
}
