// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestMetalAdapter_LiveModelSurface drives the inference.TextModel surface of
// *metaladapter against a real e2b model: the methods whose post-nil-guard
// bodies (Generate/Chat/Classify/BatchGenerate/Metrics/Info/ModelType/
// AcceptsImages/InspectAttention/Err/Close) are unreachable without a loaded
// *metal.Model. A short generate proves the forward path; Metrics afterwards
// proves the metrics translation off real counters.
func TestMetalAdapter_LiveModelSurface(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()
	off := false

	// Generate — the stateless decode loop, translating metal.Token ->
	// inference.Token off a raw (un-templated) continuation prompt. An instruct
	// checkpoint may stop immediately on a complete-looking prompt, so the path
	// (not the length) is what this asserts — the translation loop ran either
	// way; the early-break arm below uses Chat where multi-token output is sure.
	gen := core.NewBuilder()
	for token := range adapter.Generate(ctx, "Once upon a time,",
		inference.WithMaxTokens(8), inference.WithEnableThinking(&off)) {
		gen.WriteString(token.Text)
	}
	t.Logf("Generate -> %q", gen.String())

	// Break early out of a raw Generate stream — exercises the yield-returns-
	// false arm of metaladapter.Generate itself. A continuation prompt keeps the
	// instruct model producing, so two tokens before the break are reliable.
	early := 0
	for range adapter.Generate(ctx, "The numbers in order are one, two, three,",
		inference.WithMaxTokens(16), inference.WithEnableThinking(&off)) {
		early++
		if early == 2 {
			break
		}
	}
	if early != 2 {
		t.Fatalf("early-break Generate saw %d tokens, want 2", early)
	}

	// Metrics — real prefill/decode counters off the last run.
	metrics := adapter.Metrics()
	if metrics.PromptTokens == 0 && metrics.GeneratedTokens == 0 {
		t.Errorf("Metrics empty after a generate: %+v", metrics)
	}
	t.Logf("Metrics: prompt=%d gen=%d decode_tps=%.1f", metrics.PromptTokens, metrics.GeneratedTokens, metrics.DecodeTokensPerSec)

	// Info / ModelType — real architecture identity off the checkpoint.
	info := adapter.Info()
	if info.NumLayers == 0 || info.VocabSize == 0 {
		t.Errorf("Info missing structural fields: %+v", info)
	}
	if adapter.ModelType() == "" {
		t.Errorf("ModelType empty on a loaded model")
	}
	t.Logf("Info: arch=%s layers=%d vocab=%d quant=%d", info.Architecture, info.NumLayers, info.VocabSize, info.QuantBits)

	// AcceptsImages — capability probe off the real checkpoint.
	_ = adapter.AcceptsImages()

	// Err — a healthy loaded model reports no error.
	if err := adapter.Err(); err != nil {
		t.Errorf("Err on a healthy model: %v", err)
	}
}

// TestMetalAdapter_ChatLiveModel drives metaladapter.Chat (no continuity set, so
// it falls through to the stateless chat-template path) on a real model.
func TestMetalAdapter_ChatLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()
	off := false

	messages := []inference.Message{
		{Role: "user", Content: "Reply with the single word: ready."},
	}
	reply := core.NewBuilder()
	for token := range adapter.Chat(ctx, messages,
		inference.WithMaxTokens(8), inference.WithEnableThinking(&off)) {
		reply.WriteString(token.Text)
	}
	if reply.Len() == 0 {
		t.Fatalf("Chat produced nothing")
	}
	t.Logf("Chat -> %q", reply.String())

	// Early-break arm of the Chat yield loop.
	seen := 0
	for range adapter.Chat(ctx, messages, inference.WithMaxTokens(16), inference.WithEnableThinking(&off)) {
		seen++
		if seen == 1 {
			break
		}
	}
	if seen != 1 {
		t.Fatalf("early-break Chat saw %d tokens, want 1", seen)
	}
}

// TestMetalAdapter_ClassifyLiveModel drives metaladapter.Classify (with and
// without logits) on a real model — the result-translation body past the
// nil-model guard.
func TestMetalAdapter_ClassifyLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()

	results, err := adapter.Classify(ctx, []string{"The sky is", "Two plus two is"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Classify returned %d results, want 2", len(results))
	}
	for i, r := range results {
		t.Logf("Classify[%d] -> token %d %q", i, r.Token.ID, r.Token.Text)
	}

	// With logits — exercises the ReturnLogits arm and the Logits field copy.
	withLogits, err := adapter.Classify(ctx, []string{"Hello"}, inference.WithLogits())
	if err != nil {
		t.Fatalf("Classify(ReturnLogits): %v", err)
	}
	if len(withLogits) != 1 {
		t.Fatalf("Classify(logits) returned %d results, want 1", len(withLogits))
	}
}

// TestMetalAdapter_BatchGenerateLiveModel drives metaladapter.BatchGenerate on a
// real model — the per-result token-slice translation body.
func TestMetalAdapter_BatchGenerateLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()
	off := false

	results, err := adapter.BatchGenerate(ctx,
		[]string{"Say hi.", "Say bye."},
		inference.WithMaxTokens(6), inference.WithEnableThinking(&off))
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("BatchGenerate returned %d results, want 2", len(results))
	}
	for i, r := range results {
		if r.Err != nil {
			t.Errorf("BatchGenerate[%d] err: %v", i, r.Err)
		}
		t.Logf("BatchGenerate[%d] -> %d tokens", i, len(r.Tokens))
	}
}

// TestMetalAdapter_InspectAttentionLiveModel drives metaladapter.InspectAttention
// on a real model — the AttentionSnapshot translation past the nil guard.
func TestMetalAdapter_InspectAttentionLiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()
	ctx := context.Background()

	snapshot, err := adapter.InspectAttention(ctx, "The cat sat on the mat.")
	if err != nil {
		t.Fatalf("InspectAttention: %v", err)
	}
	if snapshot == nil {
		t.Fatal("InspectAttention returned nil snapshot")
	}
	if snapshot.NumLayers == 0 || snapshot.SeqLen == 0 {
		t.Errorf("InspectAttention snapshot missing fields: %+v", *snapshot)
	}
	t.Logf("InspectAttention: layers=%d heads=%d seqlen=%d arch=%s",
		snapshot.NumLayers, snapshot.NumHeads, snapshot.SeqLen, snapshot.Architecture)
}
