// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/internal/sessionfake"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/spine"
)

func TestAgentMemoryInferenceContract_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := inference.TokenizerIdentity{Hash: "tok-contract", ChatTemplate: "chat"}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

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

	awakeNative := &sessionfake.Handle{}
	awake := &Session{session: awakeNative, info: info}
	wake, err := any(awake).(inference.AgentMemorySession).WakeState(ctx, inference.AgentMemoryWakeRequest{
		Store:     store,
		IndexURI:  sleep.Index.URI,
		EntryURI:  sleep.Entry.URI,
		Tokenizer: tokenizer,
	})

	if err != nil {
		t.Fatalf("WakeState() error = %v", err)
	}
	if wake.Entry.URI != sleep.Entry.URI || wake.PrefixTokens != 2 || awakeNative.RestoredKV == nil {
		t.Fatalf("WakeState() = %+v restored=%+v, want restored contract state", wake, awakeNative.RestoredKV)
	}
}

func TestAppendAndSleepAgentMemory_NoReply_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{
		session: native,
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}

	report, err := session.AppendAndSleepAgentMemory(ctx, "repo observation: tests pass", store, agent.SleepOptions{
		EntryURI: "mlx://agent/no-reply",
		Title:    "No reply observation",
	})

	if err != nil {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v", err)
	}
	if native.AppendPromptSeen != "repo observation: tests pass" {
		t.Fatalf("append prompt = %q, want observation", native.AppendPromptSeen)
	}
	if native.GenerateCalls != 0 {
		t.Fatalf("Generate calls = %d, want no-reply append/sleep path", native.GenerateCalls)
	}
	if report.EntryURI != "mlx://agent/no-reply" || report.TokenCount != 2 {
		t.Fatalf("report = %+v, want durable two-token state", report)
	}
}

func TestAgentMemoryWakeSleep_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	var session *Session
	if _, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil session) error = nil")
	}
	session = &Session{session: &sessionfake.Handle{}}
	if _, err := session.SleepAgentMemory(ctx, nil, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil store) error = nil")
	}
	if _, err := session.WakeAgentMemory(ctx, store, agent.WakeOptions{}); err == nil {
		t.Fatal("WakeAgentMemory(missing index) error = nil")
	}

	bundle := kvSnapshotIndexTestBundle()
	index, err := agent.NewMemvidIndex(bundle, agent.MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: spine.ModelInfoToMemory(spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}),
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
