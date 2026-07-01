// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

// TestMetalAdapter_BlockCacheServiceLiveModel drives the adapter's block-cache
// service surface on a real model: blockCacheService (the real-model build arm,
// incl. adapterTokenizerHashFromInfo and the Tokenize/WarmPrompt closures),
// CacheStats, CacheEntries, WarmCache, and ClearCache.
func TestMetalAdapter_BlockCacheServiceLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()

	stats, err := adapter.CacheStats(ctx)
	if err != nil {
		t.Fatalf("CacheStats: %v", err)
	}
	t.Logf("cache stats: %+v", stats)

	if _, err := adapter.CacheEntries(ctx, nil); err != nil {
		t.Fatalf("CacheEntries: %v", err)
	}

	// WarmCache builds (and tokenizes) a cache entry for a prompt — exercises
	// the Tokenize + WarmPrompt closures the service wires off the real model.
	if _, err := adapter.WarmCache(ctx, inference.CacheWarmRequest{
		Prompt: "The harbour bell rang twice at dusk.",
	}); err != nil {
		t.Fatalf("WarmCache: %v", err)
	}

	// ClearCache over the service — the ClearRuntime closure path.
	if _, err := adapter.ClearCache(ctx, nil); err != nil {
		t.Fatalf("ClearCache: %v", err)
	}
}

// TestEnableConversationContinuity_LiveModel drives the EnableConversationContinuity
// success arm: wiring the no-prompt-replay loop into a real loaded adapter, then
// proving a Chat routes through the continuity lane (the metaladapter.Chat
// continuity branch) rather than the stateless path.
func TestEnableConversationContinuity_LiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()
	off := false

	store := state.NewInMemoryStore(nil)
	cc, err := EnableConversationContinuity(adapter, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("EnableConversationContinuity: %v", err)
	}
	if cc == nil {
		t.Fatal("EnableConversationContinuity returned nil manager")
	}

	// A Chat now routes through the wired continuity loop. After one turn the
	// manager records a fresh conversation, proving the adapter.continuity
	// branch fired instead of the stateless path.
	reply := core.NewBuilder()
	for token := range adapter.Chat(ctx,
		[]inference.Message{{Role: "user", Content: "Reply with one short greeting."}},
		inference.WithMaxTokens(16), inference.WithTemperature(0), inference.WithEnableThinking(&off)) {
		reply.WriteString(token.Text)
	}
	if reply.Len() == 0 {
		t.Fatal("continuity-wired Chat produced nothing")
	}
	if stats := cc.Stats(); stats.FreshConversations == 0 {
		t.Errorf("continuity did not record the turn: %+v", stats)
	}
	t.Logf("continuity-wired Chat -> %q", reply.String())
}

// TestNativeTextModel_ClassifyBatchLiveModel drives the no-cgo native model's
// Classify (one greedy token per prompt) and BatchGenerate (one stream per
// prompt) — the partially-covered contract arms, against a real load.
func TestNativeTextModel_ClassifyBatchLiveModel(t *testing.T) {
	native, done := requireLiveNativeTextModel(t)
	defer done()
	ctx := context.Background()

	results, err := castClassify(native.Classify(ctx, []string{"The sky is", "Two plus two is"}))
	if err != nil {
		t.Fatalf("native Classify: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("native Classify returned %d results, want 2", len(results))
	}
	for i, r := range results {
		t.Logf("native Classify[%d] -> %q", i, r.Token.Text)
	}

	batch, err := castBatch(native.BatchGenerate(ctx, []string{"Hello there"}, inference.WithMaxTokens(6)))
	if err != nil {
		t.Fatalf("native BatchGenerate: %v", err)
	}
	if len(batch) != 1 {
		t.Fatalf("native BatchGenerate returned %d results, want 1", len(batch))
	}
	t.Logf("native BatchGenerate[0] -> %d tokens", len(batch[0].Tokens))
}
