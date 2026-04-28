// SPDX-Licence-Identifier: EUPL-1.2

//go:build !nomlxlm

package mlxlm

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"dappco.re/go"

	"dappco.re/go/inference"
)

// mockScript returns the absolute path to testdata/mock_bridge.py.
func mockScript(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine test file path")
	}
	return core.JoinPath(core.PathDir(file), "testdata", "mock_bridge.py")
}

// loadMock spawns a model backed by the mock Python script.
func loadMock(t *testing.T, modelPath string) inference.TextModel {
	t.Helper()
	m, err := loadModel(context.Background(), modelPath, mockScript(t))
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// (a) Name returns "mlx_lm".
func TestBackend_Name_Good(t *testing.T) {
	b := &mlxlmBackend{}
	if got := b.Name(); got != "mlx_lm" {
		t.Errorf("Name() = %q, want %q", got, "mlx_lm")
	}
}

// (b) LoadModel spawns subprocess, sends load command, gets response.
func TestBackend_LoadModel_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	if m.ModelType() != "mock_model" {
		t.Errorf("ModelType() = %q, want %q", m.ModelType(), "mock_model")
	}
}

func TestOptionalFloat32Field_Good(t *testing.T) {
	type withMinP struct {
		MinP float32
	}

	got, ok := optionalFloat32Field(withMinP{MinP: 0.05}, "MinP")
	if !ok {
		t.Fatal("expected MinP field to be found")
	}
	if got != 0.05 {
		t.Fatalf("optionalFloat32Field() = %f, want %f", got, 0.05)
	}
}

func TestOptionalFloat32Field_MissingField_Good(t *testing.T) {
	type withoutMinP struct {
		TopP float32
	}

	if got, ok := optionalFloat32Field(withoutMinP{TopP: 0.9}, "MinP"); ok || got != 0 {
		t.Fatalf("optionalFloat32Field() = (%f, %v), want (0, false)", got, ok)
	}
}

// (c) Generate streams tokens from subprocess, all tokens received.
func TestBackend_Generate_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var tokens []inference.Token
	for tok := range m.Generate(ctx, "Hello", inference.WithMaxTokens(5)) {
		tokens = append(tokens, tok)
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(tokens) != 5 {
		t.Fatalf("got %d tokens, want 5", len(tokens))
	}

	// Verify token content matches mock.
	expected := []string{"Hello", " ", "world", "!", "\n"}
	for i, tok := range tokens {
		if tok.Text != expected[i] {
			t.Errorf("token[%d].Text = %q, want %q", i, tok.Text, expected[i])
		}
		wantID := int32(100 + i)
		if tok.ID != wantID {
			t.Errorf("token[%d].ID = %d, want %d", i, tok.ID, wantID)
		}
	}
}

// (d) Generate with context cancellation stops early.
func TestBackend_Generate_Cancel_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var count int
	for range m.Generate(ctx, "Hello", inference.WithMaxTokens(5)) {
		count++
		if count >= 2 {
			cancel()
		}
	}
	// We should have received at most a few tokens before cancellation took effect.
	if count > 5 {
		t.Errorf("expected early stop, got %d tokens", count)
	}
	if err := m.Err(); err != context.Canceled {
		t.Logf("Err() = %v (expected context.Canceled)", err)
	}
}

// (e) Chat formats messages correctly and streams tokens.
func TestBackend_Chat_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var tokens []inference.Token
	for tok := range m.Chat(ctx, []inference.Message{
		{Role: "user", Content: "Hi there"},
	}, inference.WithMaxTokens(5)) {
		tokens = append(tokens, tok)
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(tokens) != 5 {
		t.Fatalf("got %d tokens, want 5", len(tokens))
	}

	// Mock chat returns "I heard you".
	expected := []string{"I", " ", "heard", " ", "you"}
	for i, tok := range tokens {
		if tok.Text != expected[i] {
			t.Errorf("token[%d].Text = %q, want %q", i, tok.Text, expected[i])
		}
	}
}

// (f) Close kills subprocess cleanly.
func TestBackend_Close_Good(t *testing.T) {
	m, err := loadModel(context.Background(), "/fake/model/path", mockScript(t))
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}
	// Close should not error.
	if err := m.Close(); err != nil {
		t.Errorf("Close() = %v", err)
	}
}

// (g) Err returns error on subprocess failure.
func TestBackend_Generate_Error_Bad(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var count int
	for range m.Generate(ctx, "ERROR trigger", inference.WithMaxTokens(5)) {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 tokens on error, got %d", count)
	}
	if err := m.Err(); err == nil {
		t.Fatal("expected non-nil Err()")
	} else if !core.Contains(err.Error(), "simulated model error") {
		t.Errorf("Err() = %q, want to contain %q", err.Error(), "simulated model error")
	}
}

// (h) LoadModel with invalid path returns error.
func TestBackend_LoadModel_Bad(t *testing.T) {
	_, err := loadModel(context.Background(), "/path/with/FAIL/in/it", mockScript(t))
	if err == nil {
		t.Fatal("expected error for FAIL path")
	}
	if !core.Contains(err.Error(), "cannot open model") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "cannot open model")
	}
}

// (i) Backend auto-registers (check inference.Get("mlx_lm")).
func TestBackend_AutoRegister_Good(t *testing.T) {
	b, ok := inference.Get("mlx_lm")
	if !ok {
		t.Fatal("mlx_lm backend not registered")
	}
	if b.Name() != "mlx_lm" {
		t.Errorf("Name() = %q, want %q", b.Name(), "mlx_lm")
	}
}

// (j) Concurrent Generate calls are serialised (mu lock).
func TestBackend_Generate_Concurrent_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	const goroutines = 3
	var wg sync.WaitGroup
	wg.Add(goroutines)

	results := make([]int, goroutines)
	for i := range goroutines {
		go func(idx int) {
			defer wg.Done()
			var count int
			for range m.Generate(ctx, "Hello", inference.WithMaxTokens(5)) {
				count++
			}
			results[idx] = count
		}(i)
	}
	wg.Wait()

	// Each goroutine should have received all 5 tokens (serialised execution).
	for i, count := range results {
		if count != 5 {
			t.Errorf("goroutine %d got %d tokens, want 5", i, count)
		}
	}
}

// Additional: Classify returns unsupported error.
func TestBackend_Classify_Unsupported_Bad(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	_, err := m.Classify(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error from Classify")
	}
	if !core.Contains(err.Error(), "not supported") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not supported")
	}
}

// Additional: BatchGenerate returns unsupported error.
func TestBackend_BatchGenerate_Unsupported_Bad(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	_, err := m.BatchGenerate(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error from BatchGenerate")
	}
	if !core.Contains(err.Error(), "not supported") {
		t.Errorf("error = %q, want to contain %q", err.Error(), "not supported")
	}
}

// Additional: Info returns model metadata.
func TestBackend_Info_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	info := m.Info()
	if info.Architecture != "mock_model" {
		t.Errorf("Architecture = %q, want %q", info.Architecture, "mock_model")
	}
	if info.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want %d", info.VocabSize, 32000)
	}
	if info.NumLayers != 24 {
		t.Errorf("NumLayers = %d, want %d", info.NumLayers, 24)
	}
	if info.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want %d", info.HiddenSize, 2048)
	}
}

// Additional: Metrics returns zero values (not tracked by subprocess).
func TestBackend_Metrics_Zero_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	met := m.Metrics()
	if met.PromptTokens != 0 || met.GeneratedTokens != 0 {
		t.Errorf("expected zero metrics, got prompt=%d generated=%d",
			met.PromptTokens, met.GeneratedTokens)
	}
}

// Additional: Generate with fewer max_tokens than available tokens.
func TestBackend_Generate_MaxTokens_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var count int
	for range m.Generate(ctx, "Hello", inference.WithMaxTokens(3)) {
		count++
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if count != 3 {
		t.Errorf("got %d tokens, want 3", count)
	}
}

func TestBackend_InspectAttention_Good(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	inspector, ok := m.(inference.AttentionInspector)
	if !ok {
		t.Fatal("mlxlmModel does not implement AttentionInspector")
	}

	snap, err := inspector.InspectAttention(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("InspectAttention: %v", err)
	}

	if snap.NumLayers != 4 {
		t.Errorf("NumLayers = %d, want 4", snap.NumLayers)
	}
	if snap.NumHeads != 2 {
		t.Errorf("NumHeads (KV) = %d, want 2", snap.NumHeads)
	}
	if snap.NumQueryHeads != 8 {
		t.Errorf("NumQueryHeads = %d, want 8", snap.NumQueryHeads)
	}
	if snap.SeqLen != 3 {
		t.Errorf("SeqLen = %d, want 3", snap.SeqLen)
	}
	if snap.HeadDim != 4 {
		t.Errorf("HeadDim = %d, want 4", snap.HeadDim)
	}
	if snap.Architecture != "mock_model" {
		t.Errorf("Architecture = %q, want %q", snap.Architecture, "mock_model")
	}

	// Verify K arrays.
	if len(snap.Keys) != 4 {
		t.Fatalf("len(Keys) = %d, want 4", len(snap.Keys))
	}
	for i, layer := range snap.Keys {
		if len(layer) != 2 {
			t.Errorf("Keys[%d] has %d heads, want 2", i, len(layer))
		}
		for j, head := range layer {
			wantLen := 3 * 4 // seq_len * head_dim
			if len(head) != wantLen {
				t.Errorf("Keys[%d][%d] len = %d, want %d", i, j, len(head), wantLen)
			}
		}
	}

	// Verify Q arrays.
	if !snap.HasQueries() {
		t.Fatal("expected HasQueries() == true")
	}
	if len(snap.Queries) != 4 {
		t.Fatalf("len(Queries) = %d, want 4", len(snap.Queries))
	}
	for i, layer := range snap.Queries {
		if len(layer) != 8 {
			t.Errorf("Queries[%d] has %d heads, want 8", i, len(layer))
		}
	}
}

func TestBackend_InspectAttention_Error_Bad(t *testing.T) {
	m := loadMock(t, "/fake/model/path")
	inspector := m.(inference.AttentionInspector)

	_, err := inspector.InspectAttention(context.Background(), "ERROR trigger")
	if err == nil {
		t.Fatal("expected error for ERROR prompt")
	}
}

// TestBackend_Generate_EmptyPrompt_Ugly validates behaviour with an empty prompt string.
// The model should still produce tokens (or at least not panic).
func TestBackend_Generate_EmptyPrompt_Ugly(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var count int
	for range m.Generate(ctx, "", inference.WithMaxTokens(5)) {
		count++
	}
	// No panic is the key invariant; token count may vary with empty prompt.
	if err := m.Err(); err != nil {
		t.Logf("Err() = %v (empty prompt may not be supported — acceptable)", err)
	}
}

// TestBackend_Chat_EmptyMessages_Ugly validates behaviour with no messages in a Chat call.
// Should not panic; may return error or zero tokens.
func TestBackend_Chat_EmptyMessages_Ugly(t *testing.T) {
	m := loadMock(t, "/fake/model/path")

	ctx := context.Background()
	var count int
	for range m.Chat(ctx, []inference.Message{}, inference.WithMaxTokens(5)) {
		count++
	}
	// No panic is the key invariant; error or zero tokens are both acceptable.
	t.Logf("empty chat produced %d tokens, Err()=%v", count, m.Err())
}

// TestBackend_LoadModel_NonexistentScript_Ugly validates behaviour when the bridge
// script path does not exist. Should return an error on load or first use.
func TestBackend_LoadModel_NonexistentScript_Ugly(t *testing.T) {
	_, err := loadModel(context.Background(), "/fake/model/path", "/nonexistent/bridge.py")
	if err == nil {
		t.Fatal("expected error when bridge script does not exist")
	}
}
