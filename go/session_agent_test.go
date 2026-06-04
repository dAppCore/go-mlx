// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
)

func TestAgentMemoryWakeSleep_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := mlxbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	native := &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()}
	session := &ModelSession{session: native, info: info}

	sleep, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{
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
	index, err := agent.LoadMemvidIndex(ctx, store, sleep.IndexURI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex() error = %v", err)
	}
	if index.Tokenizer.Hash != "tok-a" || index.Entries[0].Meta["ordinal"] != "1" {
		t.Fatalf("loaded index = %+v", index)
	}

	awakeNative := &fakeNativeSession{
		tokens: []metal.Token{{ID: 10, Text: "Rome"}},
	}
	awake := &ModelSession{session: awakeNative, info: info}
	wake, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
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
	if err := awake.AppendPrompt("\n\nQuestion: Which city was retained by the restored state?\nAnswer:"); err != nil {
		t.Fatalf("AppendPrompt(restored question) error = %v", err)
	}
	if core.Contains(awakeNative.appendPrompt, "Rome") {
		t.Fatalf("restored-state question prompt = %q, want no retained answer text", awakeNative.appendPrompt)
	}
	text, err := awake.Generate(WithMaxTokens(1))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if text != "Rome" {
		t.Fatalf("Generate() = %q, want Rome", text)
	}

	awakeNative.kv = awakeNative.restoredKV
	afterAppend, err := awake.AppendAndSleep(ctx, "\n\nQuestion: first question?\nAnswer:", store, agent.SleepOptions{
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
	afterAppendIndex, err := agent.LoadMemvidIndex(ctx, store, afterAppend.IndexURI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(after append) error = %v", err)
	}
	if got := afterAppendIndex.Entries[0].Meta["parent_entry_uri"]; got != "mlx://agent/chapter-1" {
		t.Fatalf("after append parent = %q, want chapter-1", got)
	}

	awakeNative.tokens = []metal.Token{{ID: 10, Text: "Rome"}}
	awakeNative.afterGenerate = func(s *fakeNativeSession) {
		s.kv = agentMemoryGeneratedTestMetalSnapshot()
	}
	answer, afterAnswer, err := awake.GenerateAndSleep(ctx, store, agent.SleepOptions{
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
	afterAnswerIndex, err := agent.LoadMemvidIndex(ctx, store, afterAnswer.IndexURI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(after answer) error = %v", err)
	}
	if got := afterAnswerIndex.Entries[0].Meta["parent_entry_uri"]; got != "mlx://agent/chapter-1/after-question" {
		t.Fatalf("after answer parent = %q, want after-question", got)
	}

	forkNative := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{
		session: forkNative,
		info:    metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}}
	forked, forkWake, err := model.ForkFromBundle(ctx, store, agent.WakeOptions{
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
		Adapter:   inference.AdapterIdentity{Hash: "adapter-contract", Format: "lora"},
		Runtime:   inference.RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
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
	index, err := agent.LoadMemvidIndex(ctx, store, sleep.Index.URI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(contract) error = %v", err)
	}
	if index.Entries[0].Meta["adapter_hash"] != "adapter-contract" || index.Entries[0].Meta["runtime_backend"] != "metal" || index.Entries[0].Meta["runtime_cache_mode"] != "paged-q8" {
		t.Fatalf("contract metadata = %+v, want adapter/runtime identity", index.Entries[0].Meta)
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

func TestAppendAndSleepAgentMemory_NoReply_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	native := &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()}
	session := &ModelSession{
		session: native,
		info:    ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}

	report, err := session.AppendAndSleepAgentMemory(ctx, "repo observation: tests pass", store, agent.SleepOptions{
		EntryURI: "mlx://agent/no-reply",
		Title:    "No reply observation",
	})

	if err != nil {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v", err)
	}
	if native.appendPrompt != "repo observation: tests pass" {
		t.Fatalf("append prompt = %q, want observation", native.appendPrompt)
	}
	if native.generateCalls != 0 {
		t.Fatalf("Generate calls = %d, want no-reply append/sleep path", native.generateCalls)
	}
	if report.EntryURI != "mlx://agent/no-reply" || report.TokenCount != 2 {
		t.Fatalf("report = %+v, want durable two-token state", report)
	}
}

func TestFoldAgentMemory_CheckpointSummaryTail_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := mlxbundle.Tokenizer{Hash: "tok-fold", ChatTemplateHash: "chat-fold"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	exhaustedNative := &fakeNativeSession{kv: agentMemoryGeneratedTestMetalSnapshot()}
	exhausted := &ModelSession{session: exhaustedNative, info: info}
	foldedNative := &fakeNativeSession{kvBlocks: []metal.KVSnapshotBlock{
		agentMemoryTestMetalBlock(0, 0, 1),
		agentMemoryTestMetalBlock(1, 1, 2),
	}}
	model := &Model{model: &fakeNativeModel{
		session: foldedNative,
		info:    metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}}

	folded, report, err := model.FoldAgentMemory(ctx, exhausted, store, AgentMemoryFoldOptions{
		Summary:           "The previous window found long-context degradation after 60k tokens.",
		RecentTail:        "The operator asked to compact and continue from a folded state.",
		PrefillChunkBytes: 32,
		Checkpoint: agent.SleepOptions{
			EntryURI:  "mlx://agent/exhausted",
			Title:     "exhausted context",
			Tokenizer: tokenizer,
		},
		Folded: agent.SleepOptions{
			EntryURI:  "mlx://agent/folded",
			Title:     "folded context",
			Tokenizer: tokenizer,
			BlockOptions: kv.StateBlockOptions{
				BlockSize: 1,
			},
		},
	})

	if err != nil {
		t.Fatalf("FoldAgentMemory() error = %v", err)
	}
	if folded == nil || folded.session != foldedNative {
		t.Fatalf("folded session = %+v, want fresh model session", folded)
	}
	if report == nil || report.Checkpoint == nil || report.Folded == nil {
		t.Fatalf("fold report = %+v, want checkpoint and folded reports", report)
	}
	if report.Checkpoint.EntryURI != "mlx://agent/exhausted" || report.Folded.EntryURI != "mlx://agent/folded" {
		t.Fatalf("fold URIs = %+v, want exhausted and folded entries", report)
	}
	if report.Folded.BlocksWritten < 2 {
		t.Fatalf("folded blocks written = %d, want multi-block folded State", report.Folded.BlocksWritten)
	}
	if report.Folded.ParentEntryURI != report.Checkpoint.EntryURI {
		t.Fatalf("folded parent = %q, want checkpoint %q", report.Folded.ParentEntryURI, report.Checkpoint.EntryURI)
	}
	prompt := promptChunksToString(func(yield func(string) bool) {
		for _, chunk := range foldedNative.prefillChunks {
			if !yield(chunk) {
				return
			}
		}
	})
	for _, want := range []string{"<summary>", "long-context degradation", "<recent_tail>", "folded state", "full exhausted context"} {
		if !core.Contains(prompt, want) {
			t.Fatalf("folded prefill prompt = %q, want %q", prompt, want)
		}
	}
	if len(foldedNative.prefillChunks) < 2 {
		t.Fatalf("prefill chunks = %v, want chunked folded prefill", foldedNative.prefillChunks)
	}
	index, err := agent.LoadMemvidIndex(ctx, store, report.Folded.IndexURI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(folded) error = %v", err)
	}
	entry := index.Entries[0]
	if entry.Meta["folded_state"] != "true" || entry.Meta["folded_from_entry_uri"] != report.Checkpoint.EntryURI {
		t.Fatalf("folded metadata = %+v, want folded lineage", entry.Meta)
	}
	if !stringSliceContains(entry.Labels, "folded-state") {
		t.Fatalf("folded labels = %+v, want folded-state", entry.Labels)
	}

	continuedNative := &fakeNativeSession{
		tokens: []metal.Token{{ID: 40, Text: "continued"}},
	}
	continued := &ModelSession{session: continuedNative, info: info}
	wake, err := continued.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI:    report.Folded.IndexURI,
		EntryURI:    report.Folded.EntryURI,
		Tokenizer:   tokenizer,
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	})
	if err != nil {
		t.Fatalf("WakeAgentMemory(folded) error = %v", err)
	}
	if wake.EntryURI != report.Folded.EntryURI || wake.PrefixTokens != report.Folded.TokenCount {
		t.Fatalf("folded wake = %+v, want folded entry and token count", wake)
	}
	if wake.RestoreStrategy != "folded-prefill" {
		t.Fatalf("folded wake restore strategy = %q, want folded-prefill", wake.RestoreStrategy)
	}
	if len(continuedNative.prefillTokens) != report.Folded.TokenCount {
		t.Fatalf("folded wake prefill tokens = %d, want %d", len(continuedNative.prefillTokens), report.Folded.TokenCount)
	}
	if continuedNative.restoredKV != nil {
		t.Fatalf("folded wake restored KV = %+v, want compact token prefill path", continuedNative.restoredKV)
	}
	if err := continued.AppendPrompt("Next turn: continue from the folded state."); err != nil {
		t.Fatalf("AppendPrompt(folded continuation) error = %v", err)
	}
	if core.Contains(continuedNative.appendPrompt, "long-context degradation") {
		t.Fatalf("folded continuation prompt = %q, want no replayed summary text", continuedNative.appendPrompt)
	}
	text, err := continued.Generate(WithMaxTokens(1))
	if err != nil {
		t.Fatalf("Generate(folded continuation) error = %v", err)
	}
	if text != "continued" {
		t.Fatalf("Generate(folded continuation) = %q, want continued", text)
	}
}

func TestFoldAgentMemory_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	model := &Model{model: &fakeNativeModel{session: &fakeNativeSession{}}}
	exhausted := &ModelSession{session: &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()}}

	folded, report, err := model.FoldAgentMemory(ctx, exhausted, store, AgentMemoryFoldOptions{})

	if err == nil {
		t.Fatal("FoldAgentMemory(empty summary) error = nil")
	}
	if folded != nil || report != nil {
		t.Fatalf("FoldAgentMemory(empty summary) = %+v/%+v, want nils", folded, report)
	}
}

func TestModelWakeAgentMemory_ClosesOnRestoreError_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	source := &ModelSession{
		session: &fakeNativeSession{kv: agentMemoryTestMetalSnapshot()},
		info:    ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/error"})
	if err != nil {
		t.Fatalf("seed SleepAgentMemory() error = %v", err)
	}
	wantErr := core.NewError("restore failed")
	native := &fakeNativeSession{restoreBlocksErr: wantErr}
	model := &Model{model: &fakeNativeModel{
		session: native,
		info:    metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}}

	session, report, err := model.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI})

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
	if _, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil session) error = nil")
	}
	session = &ModelSession{session: &fakeNativeSession{}}
	if _, err := session.SleepAgentMemory(ctx, nil, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil store) error = nil")
	}
	if _, err := session.WakeAgentMemory(ctx, store, agent.WakeOptions{}); err == nil {
		t.Fatal("WakeAgentMemory(missing index) error = nil")
	}

	bundle := kvSnapshotIndexTestBundle()
	index, err := agent.NewMemvidIndex(bundle, agent.MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: modelInfoToMemory(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}),
		Entries: []agent.MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("agent.NewMemvidIndex() error = %v", err)
	}
	_, err = session.WakeAgentMemory(ctx, store, agent.WakeOptions{
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

func agentMemoryTestMetalBlock(index, tokenStart int, token int32) metal.KVSnapshotBlock {
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{token},
		TokenOffset:   tokenStart + 1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       2,
		NumQueryHeads: 8,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []metal.KVHeadSnapshot{{
				Key:   []float32{float32(token), 0},
				Value: []float32{0, float32(token)},
			}},
		}},
	}
	return metal.KVSnapshotBlock{
		Index:      index,
		TokenStart: tokenStart,
		TokenCount: 1,
		Snapshot:   snapshot,
	}
}

// kvSnapshotIndexTestBundle returns a small KV memvid block bundle for
// mlx-root tests (session_agent_darwin_test.go) that need fixture data.
// Duplicated from agent/index_test.go because Go test packages cannot
// import each other's internal _test.go symbols.
func kvSnapshotIndexTestBundle() *kv.MemvidBlockBundle {
	return &kv.MemvidBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		TokenOffset:  4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.MemvidBlockRef{{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			Memvid:     memvid.ChunkRef{ChunkID: 1},
		}},
	}
}
