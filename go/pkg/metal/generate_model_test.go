// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
)

// TestGenerate_Model_Chat_Good drives the chat-formatted generation path on a
// synthetic model: messages → formatChat → Generate. No real weights — the tiny
// dense model from loadSyntheticTextModel runs the forward.
func TestGenerate_Model_Chat_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	messages := []ChatMessage{
		{Role: "user", Content: "hello"},
	}
	got := drainTokens(model.Chat(context.Background(), messages, GenerateConfig{MaxTokens: 1}))
	if err := model.Err(); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Chat produced no tokens")
	}
}

// TestGenerate_Model_ChatChunks_Good drives the chunked chat path:
// messages → formatChatChunks → GenerateChunks.
func TestGenerate_Model_ChatChunks_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	messages := []ChatMessage{
		{Role: "user", Content: "hello"},
	}
	got := drainTokens(model.ChatChunks(context.Background(), messages, 8, GenerateConfig{MaxTokens: 1}))
	if err := model.Err(); err != nil {
		t.Fatalf("ChatChunks: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("ChatChunks produced no tokens")
	}
}

// TestGenerate_Model_GenerateChunks_Good drives generation from bounded prompt
// chunks: encodePromptChunks → generateTokens.
func TestGenerate_Model_GenerateChunks_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	chunks := slices.Values([]string{"hello", "world"})
	got := drainTokens(model.GenerateChunks(context.Background(), chunks, GenerateConfig{MaxTokens: 1}))
	if err := model.Err(); err != nil {
		t.Fatalf("GenerateChunks: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateChunks produced no tokens")
	}
}

// TestGenerate_Model_WarmPromptCache_Good prefills and stores an exact
// token-prefix KV cache, then re-warms it (idempotent) to exercise the
// prompt-cache store + match-on-second-call paths.
func TestGenerate_Model_WarmPromptCache_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	ctx := context.Background()

	if err := model.WarmPromptCache(ctx, "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	// Second warm of the same prompt exercises the prefix-match path.
	if err := model.WarmPromptCache(ctx, "hello"); err != nil {
		t.Fatalf("WarmPromptCache (re-warm): %v", err)
	}
	// A subsequent generation should reuse the warmed cache without error.
	got := drainTokens(model.Generate(ctx, "hello", GenerateConfig{MaxTokens: 1}))
	if err := model.Err(); err != nil {
		t.Fatalf("Generate after WarmPromptCache: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens after WarmPromptCache")
	}
}

// TestGenerate_Model_WarmPromptCacheChunks_Good warms the prompt cache from
// bounded chunks.
func TestGenerate_Model_WarmPromptCacheChunks_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	chunks := slices.Values([]string{"hello", "world"})
	if err := model.WarmPromptCacheChunks(context.Background(), chunks); err != nil {
		t.Fatalf("WarmPromptCacheChunks: %v", err)
	}
}

// TestGenerate_Model_InspectAttention_Good extracts per-layer attention tensors
// from a single synthetic forward pass.
func TestGenerate_Model_InspectAttention_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	result, err := model.InspectAttention(context.Background(), "hello")
	if err != nil {
		t.Fatalf("InspectAttention: %v", err)
	}
	if result == nil {
		t.Fatal("InspectAttention returned nil result")
	}
	if result.NumLayers != 1 {
		t.Fatalf("InspectAttention NumLayers = %d, want 1 (single-layer fixture)", result.NumLayers)
	}
	if result.SeqLen <= 0 {
		t.Fatalf("InspectAttention SeqLen = %d, want positive", result.SeqLen)
	}
}

// TestGenerate_Model_InspectAttention_Bad rejects a prompt that tokenises to
// nothing after a BOS strip is not applicable here — the one-shot path keeps the
// BOS, so an empty prompt yields a lone BOS and still inspects. Drive the
// genuinely-empty path by closing the model's runtime first: requireTextRuntime
// must fail closed.
func TestGenerate_Model_InspectAttention_Bad(t *testing.T) {
	model := loadSyntheticTextModel(t)
	if err := model.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err := model.InspectAttention(context.Background(), "hello")
	if err == nil {
		t.Fatal("InspectAttention on closed runtime error = nil, want runtime error")
	}
	if !core.Contains(err.Error(), "runtime") && !core.Contains(err.Error(), "closed") && !core.Contains(err.Error(), "text") {
		t.Logf("InspectAttention closed-runtime error = %v", err)
	}
}
