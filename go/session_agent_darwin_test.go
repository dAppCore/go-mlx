// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/internal/metal"
)

func TestAgentMemoryWakeSleep_Good(t *testing.T) {
	coverageTokens := "AgentMemoryWakeSleep"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := StateBundleTokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	native := &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()}
	session := &ModelSession{session: native, info: info}

	sleep, err := session.SleepAgentMemory(ctx, store, AgentMemorySleepOptions{
		EntryURI:  "mlx://agent/chapter-1",
		Title:     "Chapter 1",
		Tokenizer: tokenizer,
		BlockOptions: kv.MemvidBlockOptions{
			BlockSize: 1,
		},
		Labels: []string{"chapter"},
		Meta:   map[string]string{"ordinal": "1"},
	})

	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}
	if sleep.EntryURI != "mlx://agent/chapter-1" || sleep.BundleURI != "mlx://agent/chapter-1/bundle" || sleep.IndexURI != "mlx://agent/chapter-1/index" {
		t.Fatalf("sleep URIs = %+v", sleep)
	}
	if sleep.KVEncoding != kv.EncodingNative || sleep.TokenCount != 2 || sleep.BlocksWritten != 1 {
		t.Fatalf("sleep report = %+v, want native two-token single streamed block", sleep)
	}
	if sleep.BundleRef.ChunkID == 0 || sleep.IndexRef.ChunkID == 0 || sleep.IndexHash == "" {
		t.Fatalf("sleep refs/hash = %+v", sleep)
	}
	index, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, sleep.IndexURI)
	if err != nil {
		t.Fatalf("LoadKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	if index.Tokenizer.Hash != "tok-a" || index.Entries[0].Meta["ordinal"] != "1" {
		t.Fatalf("loaded index = %+v", index)
	}

	awakeNative := &fakeNativeSession{
		tokens: []metal.Token{{ID: 10, Text: "Rome"}},
	}
	awake := &ModelSession{session: awakeNative, info: info}
	wake, err := awake.WakeAgentMemory(ctx, store, AgentMemoryWakeOptions{
		IndexURI:    sleep.IndexURI,
		EntryURI:    sleep.EntryURI,
		Tokenizer:   tokenizer,
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	})

	if err != nil {
		t.Fatalf("WakeAgentMemory() error = %v", err)
	}
	if wake.PrefixTokens != 2 || wake.BlocksRead != 1 || wake.BundleTokens != 2 {
		t.Fatalf("wake report = %+v, want one two-token block", wake)
	}
	if awakeNative.restoredKV == nil || len(awakeNative.restoredKV.Tokens) != 2 {
		t.Fatalf("restored KV = %+v", awakeNative.restoredKV)
	}
	text, err := awake.Generate(WithMaxTokens(1))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if text != "Rome" {
		t.Fatalf("Generate() = %q, want Rome", text)
	}

	awakeNative.kv = awakeNative.restoredKV
	afterAppend, err := awake.AppendAndSleep(ctx, "\n\nQuestion: first question?\nAnswer:", store, AgentMemorySleepOptions{
		EntryURI:  "mlx://agent/chapter-1/after-question",
		Title:     "Chapter 1 after question",
		Tokenizer: tokenizer,
	})
	if err != nil {
		t.Fatalf("AppendAndSleep() error = %v", err)
	}
	if awakeNative.appendPrompt == "" || afterAppend.EntryURI != "mlx://agent/chapter-1/after-question" || afterAppend.ParentEntryURI != "mlx://agent/chapter-1" {
		t.Fatalf("append/sleep = %q/%+v", awakeNative.appendPrompt, afterAppend)
	}
	afterAppendIndex, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, afterAppend.IndexURI)
	if err != nil {
		t.Fatalf("LoadKVSnapshotMemvidBundleIndex(after append) error = %v", err)
	}
	if got := afterAppendIndex.Entries[0].Meta["parent_entry_uri"]; got != "mlx://agent/chapter-1" {
		t.Fatalf("after append parent = %q, want chapter-1", got)
	}

	awakeNative.tokens = []metal.Token{{ID: 10, Text: "Rome"}}
	awakeNative.afterGenerate = func(s *fakeNativeSession) {
		s.kv = agentMemoryGeneratedTestMetalSnapshot()
	}
	answer, afterAnswer, err := awake.GenerateAndSleep(ctx, store, AgentMemorySleepOptions{
		EntryURI:  "mlx://agent/chapter-1/after-answer",
		Title:     "Chapter 1 after answer",
		Tokenizer: tokenizer,
	}, WithMaxTokens(1))
	if err != nil {
		t.Fatalf("GenerateAndSleep() error = %v", err)
	}
	if answer != "Rome" || afterAnswer.ParentEntryURI != "mlx://agent/chapter-1/after-question" || afterAnswer.TokenCount != 3 {
		t.Fatalf("answer/sleep = %q/%+v, want Rome child of after-question with three tokens", answer, afterAnswer)
	}
	afterAnswerIndex, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, afterAnswer.IndexURI)
	if err != nil {
		t.Fatalf("LoadKVSnapshotMemvidBundleIndex(after answer) error = %v", err)
	}
	if got := afterAnswerIndex.Entries[0].Meta["parent_entry_uri"]; got != "mlx://agent/chapter-1/after-question" {
		t.Fatalf("after answer parent = %q, want after-question", got)
	}

	forkNative := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{
		session: forkNative,
		info:    metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}}
	forked, forkWake, err := model.ForkFromBundle(ctx, store, AgentMemoryWakeOptions{
		IndexURI:  sleep.IndexURI,
		Tokenizer: tokenizer,
	})
	if err != nil {
		t.Fatalf("ForkFromBundle() error = %v", err)
	}
	defer forked.Close()
	if forkWake.EntryURI != "mlx://agent/chapter-1" || forkNative.restoredKV == nil {
		t.Fatalf("fork wake/restored = %+v/%+v", forkWake, forkNative.restoredKV)
	}
}

func TestAgentMemoryInferenceContract_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := inference.TokenizerIdentity{Hash: "tok-contract", ChatTemplate: "chat"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	source := &ModelSession{session: &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()}, info: info}

	sleep, err := any(source).(inference.AgentMemorySession).SleepState(ctx, inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  "mlx://agent/contract",
		Title:     "contract state",
		Tokenizer: tokenizer,
		BlockSize: 1,
		Encoding:  string(kv.EncodingNative),
		Metadata:  map[string]string{"suite": "inference"},
	})

	if err != nil {
		t.Fatalf("SleepState() error = %v", err)
	}
	if sleep.Entry.URI != "mlx://agent/contract" || sleep.TokenCount != 2 || sleep.BlocksWritten != 1 {
		t.Fatalf("SleepState() = %+v, want contract state with one block", sleep)
	}
	if sleep.Index.URI == "" || sleep.Bundle.URI == "" {
		t.Fatalf("SleepState refs = %+v/%+v, want index and bundle refs", sleep.Index, sleep.Bundle)
	}

	awakeNative := &fakeNativeSession{}
	awake := &ModelSession{session: awakeNative, info: info}
	wake, err := any(awake).(inference.AgentMemorySession).WakeState(ctx, inference.AgentMemoryWakeRequest{
		Store:     store,
		IndexURI:  sleep.Index.URI,
		EntryURI:  sleep.Entry.URI,
		Tokenizer: tokenizer,
	})

	if err != nil {
		t.Fatalf("WakeState() error = %v", err)
	}
	if wake.Entry.URI != sleep.Entry.URI || wake.PrefixTokens != 2 || awakeNative.restoredKV == nil {
		t.Fatalf("WakeState() = %+v restored=%+v, want restored contract state", wake, awakeNative.restoredKV)
	}
}

func TestModelWakeAgentMemory_ClosesOnRestoreError_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	source := &ModelSession{
		session: &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()},
		info:    ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}
	sleep, err := source.SleepAgentMemory(ctx, store, AgentMemorySleepOptions{EntryURI: "mlx://agent/error"})
	if err != nil {
		t.Fatalf("seed SleepAgentMemory() error = %v", err)
	}
	wantErr := core.NewError("restore failed")
	native := &fakeNativeSession{restoreBlocksErr: wantErr}
	model := &Model{model: &fakeNativeModel{
		session: native,
		info:    metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}}

	session, report, err := model.WakeAgentMemory(ctx, store, AgentMemoryWakeOptions{IndexURI: sleep.IndexURI})

	if !core.Is(err, wantErr) {
		t.Fatalf("WakeAgentMemory() error = %v, want %v", err, wantErr)
	}
	if session != nil || report != nil {
		t.Fatalf("WakeAgentMemory() session/report = %+v/%+v, want nils", session, report)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.closeCalls)
	}
}

func TestAgentMemoryWakeSleep_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	var session *ModelSession
	if _, err := session.SleepAgentMemory(ctx, store, AgentMemorySleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil session) error = nil")
	}
	session = &ModelSession{session: &fakeNativeSession{}}
	if _, err := session.SleepAgentMemory(ctx, nil, AgentMemorySleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil store) error = nil")
	}
	if _, err := session.WakeAgentMemory(ctx, store, AgentMemoryWakeOptions{}); err == nil {
		t.Fatal("WakeAgentMemory(missing index) error = nil")
	}

	bundle := kvSnapshotIndexTestBundle()
	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Entries: []KVSnapshotMemvidBundleIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	_, err = session.WakeAgentMemory(ctx, store, AgentMemoryWakeOptions{
		Index:    index,
		EntryURI: "mlx://chapter",
	})
	if err == nil {
		t.Fatal("WakeAgentMemory(missing bundle) error = nil")
	}
}

func agentMemoryTestMetalSnapshot() *metal.KVSnapshot {
	return &metal.KVSnapshot{
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
				Key:        []float32{1, 0, 0, 1},
				KeyDType:   metal.DTypeFloat32,
				KeyBytes:   []byte{0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63},
				Value:      []float32{0, 1, 1, 0},
				ValueDType: metal.DTypeFloat32,
				ValueBytes: []byte{0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 0, 0},
			}},
		}},
	}
}

func agentMemoryGeneratedTestMetalSnapshot() *metal.KVSnapshot {
	return &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 10},
		Generated:     []int32{10},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.7, 0.2, 0.1},
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []metal.KVHeadSnapshot{{
				Key:   []float32{1, 0, 0, 1, 1, 1},
				Value: []float32{0, 1, 1, 0, 1, 1},
			}},
		}},
	}
}
