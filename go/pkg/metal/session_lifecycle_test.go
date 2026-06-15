// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// loadSyntheticTextModel builds a tiny runnable dense model from synthetic
// safetensors fixtures (config + tokenizer + tinyDenseDecoderWeights) and loads
// it through LoadAndInit. No real weights are read — the model is fully
// synthetic and small enough to run a one-token forward on Metal, so the whole
// session lifecycle (Prefill/Append/Capture/Fork/Restore) is exercised without
// an AX-11 model load. The returned model is closed by t.Cleanup.
func loadSyntheticTextModel(t *testing.T) *Model {
	t.Helper()
	requireMetalRuntime(t)
	dir := t.TempDir()
	// Config dimensions must match tinyDenseDecoderWeights (hidden 8, head_dim
	// 4, vocab 5, intermediate 16) — writeMinimalConfig's 64/32/100 dims would
	// not, and the first forward would feed RoPE an empty array.
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "mistral",
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 5,
		"max_position_embeddings": 32,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000
	}`); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	weights := tinyDenseDecoderWeights()
	defer freeArrayMap(weights)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 32})
	if err != nil {
		t.Fatalf("LoadAndInit(mistral): %v", err)
	}
	t.Cleanup(func() { _ = model.Close() })
	return model
}

// drainTokens collects all tokens from a generation sequence.
func drainTokens(seq func(yield func(Token) bool)) []Token {
	var got []Token
	for token := range seq {
		got = append(got, token)
	}
	return got
}

// TestModelSession_Lifecycle_Good walks the full session lifecycle on a
// synthetic model in one flow: NewSession → Prefill → Generate → CaptureKV →
// Fork → RestoreKV → Reset. This is the gemma3 synthetic-model insight scaled
// to the session engine — every step runs a real (tiny) Metal forward.
func TestModelSession_Lifecycle_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	ctx := context.Background()

	handle := model.NewSession()
	session, ok := handle.(*ModelSession)
	if !ok {
		t.Fatalf("NewSession() type = %T, want *ModelSession", handle)
	}
	defer func() { _ = session.Close() }()

	if err := session.Prefill(ctx, "hello"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	got := drainTokens(session.Generate(ctx, GenerateConfig{MaxTokens: 1}))
	if err := session.Err(); err != nil {
		t.Fatalf("Generate after Prefill: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens after Prefill")
	}

	snapshot, err := session.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snapshot == nil {
		t.Fatal("CaptureKV returned nil snapshot")
	}

	forked, err := session.Fork(ctx)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	forkSession, ok := forked.(*ModelSession)
	if !ok {
		t.Fatalf("Fork() type = %T, want *ModelSession", forked)
	}
	defer func() { _ = forkSession.Close() }()

	// The fork carries independent generation state — it can generate on its own.
	forkGot := drainTokens(forkSession.Generate(ctx, GenerateConfig{MaxTokens: 1}))
	if err := forkSession.Err(); err != nil {
		t.Fatalf("forked Generate: %v", err)
	}
	if len(forkGot) == 0 {
		t.Fatal("forked session produced no tokens")
	}

	// Restore the captured snapshot back into the original session.
	if err := session.RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}

	// Reset clears retained state; a fresh prefill must then succeed.
	session.Reset()
	if err := session.Prefill(ctx, "hello"); err != nil {
		t.Fatalf("Prefill after Reset: %v", err)
	}
}

// TestModelSession_PrefillTokens_Good prefills from an already-tokenised prompt.
func TestModelSession_PrefillTokens_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.PrefillTokens(context.Background(), []int32{2, 3, 4}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	got := drainTokens(session.Generate(context.Background(), GenerateConfig{MaxTokens: 1}))
	if err := session.Err(); err != nil {
		t.Fatalf("Generate after PrefillTokens: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens after PrefillTokens")
	}
}

// TestModelSession_PrefillTokens_Bad rejects an empty token slice.
func TestModelSession_PrefillTokens_Bad(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	err := session.PrefillTokens(context.Background(), nil)
	if err == nil {
		t.Fatal("PrefillTokens(nil) error = nil, want empty-tokens error")
	}
	if !core.Contains(err.Error(), "empty") {
		t.Fatalf("PrefillTokens(nil) error = %v, want mention of empty", err)
	}
}

// TestModelSession_Prefill_Ugly rejects mutation (Prefill) after the session is
// closed — readyForMutation must fail closed rather than touch freed state.
func TestModelSession_Prefill_Ugly(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	if err := session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	err := session.Prefill(context.Background(), "hello")
	if err == nil {
		t.Fatal("Prefill after Close error = nil, want closed-session error")
	}
}

// TestModelSession_PrefillChunks_Good prefills from bounded prompt chunks.
func TestModelSession_PrefillChunks_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	chunks := slices.Values([]string{"hello", "world"})
	if err := session.PrefillChunks(context.Background(), chunks); err != nil {
		t.Fatalf("PrefillChunks: %v", err)
	}
	got := drainTokens(session.Generate(context.Background(), GenerateConfig{MaxTokens: 1}))
	if err := session.Err(); err != nil {
		t.Fatalf("Generate after PrefillChunks: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens after PrefillChunks")
	}
}

// TestModelSession_AppendPrompt_Bad covers the encode→strip-implicit-BOS→
// empty-guard branch: when an appended prompt tokenises to only the leading BOS
// (which is then stripped because a prefix already exists), the append must fail
// closed with errAppendPromptEmpty rather than prefill an empty block. The
// minimal fixture tokenizer maps every prompt to a lone BOS, so this is the
// reachable text-append guard path. (The forward primitive AppendPrompt shares
// with AppendTokens is covered by TestModelSession_AppendTokens_Good.)
func TestModelSession_AppendPrompt_Bad(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.Prefill(context.Background(), "hello"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	err := session.AppendPrompt(context.Background(), "hello")
	if err == nil {
		t.Fatal("AppendPrompt with BOS-only encoding error = nil, want empty-prompt error")
	}
	if !core.Contains(err.Error(), "empty") {
		t.Fatalf("AppendPrompt error = %v, want mention of empty", err)
	}
}

// TestModelSession_AppendPrompt_Ugly rejects an append before any prefill.
func TestModelSession_AppendPrompt_Ugly(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	err := session.AppendPrompt(context.Background(), "world")
	if err == nil {
		t.Fatal("AppendPrompt before Prefill error = nil, want no-prefill error")
	}
}

// TestModelSession_AppendTokens_Good appends tokens on top of a prefilled prefix.
func TestModelSession_AppendTokens_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.PrefillTokens(context.Background(), []int32{2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if err := session.AppendTokens(context.Background(), []int32{4}); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}
	got := drainTokens(session.Generate(context.Background(), GenerateConfig{MaxTokens: 1}))
	if err := session.Err(); err != nil {
		t.Fatalf("Generate after AppendTokens: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens after AppendTokens")
	}
}

// TestModelSession_AppendTokens_Bad rejects an empty token slice on append.
func TestModelSession_AppendTokens_Bad(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.PrefillTokens(context.Background(), []int32{2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	err := session.AppendTokens(context.Background(), nil)
	if err == nil {
		t.Fatal("AppendTokens(nil) error = nil, want empty-tokens error")
	}
}

// TestModelSession_AppendPromptChunks_Bad covers the chunk-encode→empty-guard
// branch: bounded chunks that tokenise to only stripped BOS must fail closed
// rather than append an empty block. The minimal fixture tokenizer maps every
// chunk to a lone BOS, so this drives the reachable chunk-append guard path.
func TestModelSession_AppendPromptChunks_Bad(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.Prefill(context.Background(), "hello"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	chunks := slices.Values([]string{"hello", "world"})
	err := session.AppendPromptChunks(context.Background(), chunks)
	if err == nil {
		t.Fatal("AppendPromptChunks with BOS-only encoding error = nil, want empty-prompt error")
	}
}

// TestModelSession_RangeKVBlocks_Good streams the retained KV timeline in blocks.
func TestModelSession_RangeKVBlocks_Good(t *testing.T) {
	model := loadSyntheticTextModel(t)
	session := model.NewSession().(*ModelSession)
	defer func() { _ = session.Close() }()

	if err := session.PrefillTokens(context.Background(), []int32{2, 3, 4}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var blocks int
	err := session.RangeKVBlocks(context.Background(), 2, KVSnapshotCaptureOptions{}, func(KVSnapshotBlock) (bool, error) {
		blocks++
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if blocks == 0 {
		t.Fatal("RangeKVBlocks yielded no blocks")
	}
}
