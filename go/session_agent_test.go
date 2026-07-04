// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	mlxbundle "dappco.re/go/inference/bundle"
	"dappco.re/go/mlx/internal/sessionfake"
	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/session"
	"dappco.re/go/mlx/spine"
)

func TestAgentMemoryWakeSleep_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := mlxbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	sess := session.New(native, info, nil)

	sleep, err := sess.SleepAgentMemory(ctx, store, agent.SleepOptions{
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

	awakeNative := &sessionfake.Handle{
		Tokens: []metal.Token{{ID: 10, Text: "Rome"}},
	}
	awake := session.New(awakeNative, info, nil)
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
	if awakeNative.RestoredKV == nil || len(awakeNative.RestoredKV.Tokens) != 2 {
		t.Fatalf("restored KV = %+v", awakeNative.RestoredKV)
	}
	if err := awake.AppendPrompt("\n\nQuestion: Which city was retained by the restored state?\nAnswer:"); err != nil {
		t.Fatalf("AppendPrompt(restored question) error = %v", err)
	}
	if core.Contains(awakeNative.AppendPromptSeen, "Rome") {
		t.Fatalf("restored-state question prompt = %q, want no retained answer text", awakeNative.AppendPromptSeen)
	}
	text, err := awake.Generate(WithMaxTokens(1))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if text != "Rome" {
		t.Fatalf("Generate() = %q, want Rome", text)
	}

	awakeNative.KV = awakeNative.RestoredKV
	afterAppend, err := awake.AppendAndSleep(ctx, "\n\nQuestion: first question?\nAnswer:", store, agent.SleepOptions{
		EntryURI:  "mlx://agent/chapter-1/after-question",
		Title:     "Chapter 1 after question",
		Tokenizer: tokenizer,
	})
	if err != nil {
		t.Fatalf("AppendAndSleep() error = %v", err)
	}
	if awakeNative.AppendPromptSeen == "" || afterAppend.EntryURI != "mlx://agent/chapter-1/after-question" || afterAppend.ParentEntryURI != "mlx://agent/chapter-1" {
		t.Fatalf("append/sleep = %q/%+v", awakeNative.AppendPromptSeen, afterAppend)
	}
	afterAppendIndex, err := agent.LoadMemvidIndex(ctx, store, afterAppend.IndexURI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(after append) error = %v", err)
	}
	if got := afterAppendIndex.Entries[0].Meta["parent_entry_uri"]; got != "mlx://agent/chapter-1" {
		t.Fatalf("after append parent = %q, want chapter-1", got)
	}

	awakeNative.Tokens = []metal.Token{{ID: 10, Text: "Rome"}}
	awakeNative.AfterGenerate = func(s *sessionfake.Handle) {
		s.KV = agentMemoryGeneratedTestMetalSnapshot()
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

	forkNative := &sessionfake.Handle{}
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
	if forkWake.EntryURI != "mlx://agent/chapter-1" || forkNative.RestoredKV == nil {
		t.Fatalf("fork wake/restored = %+v/%+v", forkWake, forkNative.RestoredKV)
	}
}

func TestFoldAgentMemory_CheckpointSummaryTail_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := mlxbundle.Tokenizer{Hash: "tok-fold", ChatTemplateHash: "chat-fold"}
	info := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	exhaustedNative := &sessionfake.Handle{KV: agentMemoryGeneratedTestMetalSnapshot()}
	exhausted := session.New(exhaustedNative, info, nil)
	foldedNative := &sessionfake.Handle{KVBlocks: []metal.KVSnapshotBlock{
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
	if !folded.Valid() {
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
	prompt := spine.PromptChunksToString(func(yield func(string) bool) {
		for _, chunk := range foldedNative.PrefillChunksSeen {
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
	if len(foldedNative.PrefillChunksSeen) < 2 {
		t.Fatalf("prefill chunks = %v, want chunked folded prefill", foldedNative.PrefillChunksSeen)
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

	continuedNative := &sessionfake.Handle{
		Tokens: []metal.Token{{ID: 40, Text: "continued"}},
	}
	continued := session.New(continuedNative, info, nil)
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
	if len(continuedNative.PrefillTokensSeen) != report.Folded.TokenCount {
		t.Fatalf("folded wake prefill tokens = %d, want %d", len(continuedNative.PrefillTokensSeen), report.Folded.TokenCount)
	}
	if continuedNative.RestoredKV != nil {
		t.Fatalf("folded wake restored KV = %+v, want compact token prefill path", continuedNative.RestoredKV)
	}
	if err := continued.AppendPrompt("Next turn: continue from the folded state."); err != nil {
		t.Fatalf("AppendPrompt(folded continuation) error = %v", err)
	}
	if core.Contains(continuedNative.AppendPromptSeen, "long-context degradation") {
		t.Fatalf("folded continuation prompt = %q, want no replayed summary text", continuedNative.AppendPromptSeen)
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
	model := &Model{model: &fakeNativeModel{session: &sessionfake.Handle{}}}
	exhausted := session.New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, ModelInfo{}, nil)

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
	source := session.New(
		&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()},
		ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
		nil,
	)
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/error"})
	if err != nil {
		t.Fatalf("seed SleepAgentMemory() error = %v", err)
	}
	wantErr := core.NewError("restore failed")
	native := &sessionfake.Handle{RestoreBlocksErr: wantErr}
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
	if native.CloseCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.CloseCalls)
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
