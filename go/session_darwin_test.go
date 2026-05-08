// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

type fakeNativeSession struct {
	prefillPrompt string
	prefillErr    error
	tokens        []metal.Token
	cfg           metal.GenerateConfig
	probeEvents   []metal.ProbeEvent
	kv            *metal.KVSnapshot
	captureErr    error
	restoredKV    *metal.KVSnapshot
	restoreErr    error
	forked        metal.SessionHandle
	forkErr       error
	err           error
	resetCalls    int
	closeCalls    int
	closeErr      error
}

func (s *fakeNativeSession) Prefill(_ context.Context, prompt string) error {
	s.prefillPrompt = prompt
	return s.prefillErr
}

func (s *fakeNativeSession) Generate(_ context.Context, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	s.cfg = cfg
	return func(yield func(metal.Token) bool) {
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

func (s *fakeNativeSession) RestoreKV(_ context.Context, snapshot *metal.KVSnapshot) error {
	s.restoredKV = snapshot
	return s.restoreErr
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

func TestModelNewSessionFromKV_Good(t *testing.T) {
	coverageTokens := "ModelNewSessionFromKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	nativeSession := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{session: nativeSession}}
	snapshot := &KVSnapshot{
		Version:      KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1},
		TokenOffset:  1,
		SeqLen:       1,
		HeadDim:      1,
		LogitShape:   []int32{1, 1, 2},
		Logits:       []float32{0.1, 0.9},
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
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
		t.Fatalf("AnalyzeKV() error = %v", err)
	}
	if analysis == nil || len(KVFeatures(analysis)) != 7 {
		t.Fatalf("AnalyzeKV() = %+v", analysis)
	}
	path := core.PathJoin(t.TempDir(), "session.kvbin")
	if err := session.SaveKV(path); err != nil {
		t.Fatalf("SaveKV() error = %v", err)
	}
	loaded, err := LoadKVSnapshot(path)
	if err != nil {
		t.Fatalf("LoadKVSnapshot() error = %v", err)
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
	snapshot := &KVSnapshot{
		Version:       KVSnapshotVersion,
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
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
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
