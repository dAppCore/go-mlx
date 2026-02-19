//go:build darwin && arm64

package mlx_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
	mlx "forge.lthn.ai/core/go-mlx"
)

func TestMetalAvailable(t *testing.T) {
	// Metal backend should be registered via init()
	b, ok := inference.Get("metal")
	if !ok {
		t.Fatal("metal backend not registered")
	}
	if !b.Available() {
		t.Fatal("metal backend reports not available on darwin/arm64")
	}
}

func TestDefaultBackend(t *testing.T) {
	b, err := inference.Default()
	if err != nil {
		t.Fatalf("Default() error: %v", err)
	}
	if b.Name() != "metal" {
		t.Errorf("Default().Name() = %q, want %q", b.Name(), "metal")
	}
}

func TestGetBackend(t *testing.T) {
	b, ok := inference.Get("metal")
	if !ok {
		t.Fatal("Get(\"metal\") returned false")
	}
	if b.Name() != "metal" {
		t.Errorf("Name() = %q, want %q", b.Name(), "metal")
	}

	_, ok = inference.Get("nonexistent")
	if ok {
		t.Error("Get(\"nonexistent\") should return false")
	}
}

func TestListBackends(t *testing.T) {
	names := inference.List()
	found := false
	for _, name := range names {
		if name == "metal" {
			found = true
		}
	}
	if !found {
		t.Errorf("List() = %v, want \"metal\" included", names)
	}
}

func TestLoadModel_NoBackend(t *testing.T) {
	_, err := inference.LoadModel("/nonexistent/path")
	if err == nil {
		t.Error("expected error for nonexistent model path")
	}
}

func TestLoadModel_WithBackend(t *testing.T) {
	_, err := inference.LoadModel("/nonexistent/path", inference.WithBackend("nonexistent"))
	if err == nil {
		t.Error("expected error for nonexistent backend")
	}
}

func TestOptions(t *testing.T) {
	cfg := inference.ApplyGenerateOpts([]inference.GenerateOption{
		inference.WithMaxTokens(64),
		inference.WithTemperature(0.7),
		inference.WithTopK(40),
		inference.WithTopP(0.9),
		inference.WithStopTokens(1, 2, 3),
		inference.WithRepeatPenalty(1.1),
	})
	if cfg.MaxTokens != 64 {
		t.Errorf("MaxTokens = %d, want 64", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.7 {
		t.Errorf("Temperature = %f, want 0.7", cfg.Temperature)
	}
	if cfg.TopK != 40 {
		t.Errorf("TopK = %d, want 40", cfg.TopK)
	}
	if cfg.TopP != 0.9 {
		t.Errorf("TopP = %f, want 0.9", cfg.TopP)
	}
	if len(cfg.StopTokens) != 3 {
		t.Errorf("StopTokens len = %d, want 3", len(cfg.StopTokens))
	}
	if cfg.RepeatPenalty != 1.1 {
		t.Errorf("RepeatPenalty = %f, want 1.1", cfg.RepeatPenalty)
	}
}

func TestDefaults(t *testing.T) {
	cfg := inference.DefaultGenerateConfig()
	if cfg.MaxTokens != 256 {
		t.Errorf("default MaxTokens = %d, want 256", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.0 {
		t.Errorf("default Temperature = %f, want 0.0", cfg.Temperature)
	}
}

func TestLoadOptions(t *testing.T) {
	cfg := inference.ApplyLoadOpts([]inference.LoadOption{
		inference.WithBackend("metal"),
		inference.WithContextLen(4096),
		inference.WithGPULayers(32),
	})
	if cfg.Backend != "metal" {
		t.Errorf("Backend = %q, want %q", cfg.Backend, "metal")
	}
	if cfg.ContextLen != 4096 {
		t.Errorf("ContextLen = %d, want 4096", cfg.ContextLen)
	}
	if cfg.GPULayers != 32 {
		t.Errorf("GPULayers = %d, want 32", cfg.GPULayers)
	}
}

func TestLoadOptionsDefaults(t *testing.T) {
	cfg := inference.ApplyLoadOpts(nil)
	if cfg.GPULayers != -1 {
		t.Errorf("default GPULayers = %d, want -1", cfg.GPULayers)
	}
}

// gemma3ModelPath returns the path to a Gemma3-1B model on disk, or skips.
func gemma3ModelPath(t *testing.T) string {
	t.Helper()
	paths := []string{
		"/Volumes/Data/lem/gemma-3-1b-it-base",
		"/Volumes/Data/lem/safetensors/gemma-3/",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no Gemma3 model available")
	return ""
}

// TestLoadModel_Generate requires a model on disk. Skipped in CI.
func TestLoadModel_Generate(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	if m.ModelType() != "gemma3" {
		t.Errorf("ModelType() = %q, want %q", m.ModelType(), "gemma3")
	}

	ctx := context.Background()
	var count int
	for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(16)) {
		count++
		t.Logf("[%d] %q", tok.ID, tok.Text)
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if count == 0 {
		t.Error("Generate produced no tokens")
	}
	t.Logf("Generated %d tokens", count)
}

// TestGemma3_1B_Inference validates end-to-end inference with Gemma3-1B.
// Reports tokens/sec for prefill and decode phases.
func TestGemma3_1B_Inference(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	loadStart := time.Now()
	m, err := inference.LoadModel(modelPath)
	loadDur := time.Since(loadStart)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()
	t.Logf("Model loaded in %s", loadDur)

	if m.ModelType() != "gemma3" {
		t.Fatalf("ModelType() = %q, want %q", m.ModelType(), "gemma3")
	}

	// Generate with greedy sampling (temperature=0) for deterministic output.
	ctx := context.Background()
	const maxTokens = 64

	genStart := time.Now()
	var tokens []inference.Token
	var output strings.Builder
	for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(maxTokens)) {
		tokens = append(tokens, tok)
		output.WriteString(tok.Text)
	}
	genDur := time.Since(genStart)

	if err := m.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	nTokens := len(tokens)
	if nTokens == 0 {
		t.Fatal("Generate produced no tokens")
	}

	tps := float64(nTokens) / genDur.Seconds()
	t.Logf("Generated %d tokens in %s (%.1f tok/s)", nTokens, genDur, tps)
	t.Logf("Output: %s", output.String())

	// Log individual tokens for debugging.
	for i, tok := range tokens {
		t.Logf("  [%d] id=%d %q", i, tok.ID, tok.Text)
	}

	// Sanity: the output should contain something related to "4".
	if !strings.Contains(output.String(), "4") {
		t.Errorf("Expected output to contain '4' for 'What is 2+2?', got: %s", output.String())
	}
}

// TestGemma3_1B_Chat validates chat template formatting and generation.
func TestGemma3_1B_Chat(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	var output strings.Builder
	var count int
	for tok := range m.Chat(ctx, []inference.Message{
		{Role: "user", Content: "Reply with exactly one word: the capital of France."},
	}, inference.WithMaxTokens(16)) {
		output.WriteString(tok.Text)
		count++
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if count == 0 {
		t.Fatal("Chat produced no tokens")
	}
	t.Logf("Chat output (%d tokens): %s", count, output.String())
}

// TestGemma3_1B_ContextCancel validates that context cancellation stops generation.
func TestGemma3_1B_ContextCancel(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var count int
	for range m.Generate(ctx, "Tell me a long story about dragons.", inference.WithMaxTokens(1000)) {
		count++
		if count >= 5 {
			cancel()
		}
	}
	if count > 20 {
		t.Errorf("Expected generation to stop near 5 tokens after cancel, got %d", count)
	}
	if err := m.Err(); err != context.Canceled {
		t.Logf("Err() = %v (expected context.Canceled or nil)", err)
	}
	t.Logf("Stopped after %d tokens", count)
}

// --- Qwen2 (DeepSeek R1 7B) tests ---

func qwen2ModelPath(t *testing.T) string {
	t.Helper()
	paths := []string{
		"/Volumes/Data/lem/LEK-DeepSeek-R1-7B",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no Qwen2/DeepSeek model available")
	return ""
}

// TestQwen2_Inference validates Qwen2 arch (DeepSeek R1 7B) end-to-end.
func TestQwen2_Inference(t *testing.T) {
	modelPath := qwen2ModelPath(t)

	loadStart := time.Now()
	m, err := inference.LoadModel(modelPath)
	loadDur := time.Since(loadStart)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()
	t.Logf("Model loaded in %s", loadDur)

	if m.ModelType() != "qwen2" {
		t.Errorf("ModelType() = %q, want %q", m.ModelType(), "qwen2")
	}

	ctx := context.Background()
	genStart := time.Now()
	var tokens []inference.Token
	var output strings.Builder
	for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(32)) {
		tokens = append(tokens, tok)
		output.WriteString(tok.Text)
	}
	genDur := time.Since(genStart)

	if err := m.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	nTokens := len(tokens)
	if nTokens == 0 {
		t.Fatal("Generate produced no tokens")
	}

	tps := float64(nTokens) / genDur.Seconds()
	t.Logf("Generated %d tokens in %s (%.1f tok/s)", nTokens, genDur, tps)
	t.Logf("Output: %s", output.String())
	for i, tok := range tokens {
		t.Logf("  [%d] id=%d %q", i, tok.ID, tok.Text)
	}
}

// TestQwen2_Chat validates chat template for Qwen2 models.
func TestQwen2_Chat(t *testing.T) {
	modelPath := qwen2ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	var output strings.Builder
	var count int
	for tok := range m.Chat(ctx, []inference.Message{
		{Role: "user", Content: "Reply with exactly one word: the capital of France."},
	}, inference.WithMaxTokens(32)) {
		output.WriteString(tok.Text)
		count++
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if count == 0 {
		t.Fatal("Chat produced no tokens")
	}
	t.Logf("Chat output (%d tokens): %s", count, output.String())
}

// --- Llama 3.1 8B tests ---

func llamaModelPath(t *testing.T) string {
	t.Helper()
	paths := []string{
		"/Volumes/Data/lem/Llama-3.1-8B-Instruct-4bit",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no Llama model available")
	return ""
}

// TestLlama_Inference validates Llama 3.1 8B end-to-end.
func TestLlama_Inference(t *testing.T) {
	modelPath := llamaModelPath(t)

	loadStart := time.Now()
	m, err := inference.LoadModel(modelPath)
	loadDur := time.Since(loadStart)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()
	t.Logf("Model loaded in %s", loadDur)

	if m.ModelType() != "llama" {
		t.Errorf("ModelType() = %q, want %q", m.ModelType(), "llama")
	}

	ctx := context.Background()
	genStart := time.Now()
	var tokens []inference.Token
	var output strings.Builder
	for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(32)) {
		tokens = append(tokens, tok)
		output.WriteString(tok.Text)
	}
	genDur := time.Since(genStart)

	if err := m.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	nTokens := len(tokens)
	if nTokens == 0 {
		t.Fatal("Generate produced no tokens")
	}

	tps := float64(nTokens) / genDur.Seconds()
	t.Logf("Generated %d tokens in %s (%.1f tok/s)", nTokens, genDur, tps)
	t.Logf("Output: %s", output.String())
	for i, tok := range tokens {
		t.Logf("  [%d] id=%d %q", i, tok.ID, tok.Text)
	}
}

// --- Batch Inference tests (Gemma3-1B) ---

// TestClassify_Batch validates batched prefill-only classification.
func TestClassify_Batch(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	prompts := []string{
		"The capital of France is",
		"2 + 2 =",
		"The colour of the sky is",
		"Go is a programming",
	}

	start := time.Now()
	results, err := m.Classify(ctx, prompts)
	dur := time.Since(start)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	if len(results) != len(prompts) {
		t.Fatalf("Classify returned %d results, want %d", len(results), len(prompts))
	}

	for i, r := range results {
		if r.Token.ID == 0 {
			t.Errorf("prompt %d: got pad token (id=0)", i)
		}
		t.Logf("prompt %d %q → token %d %q", i, prompts[i], r.Token.ID, r.Token.Text)
	}
	t.Logf("Classified %d prompts in %s (%.1f prompts/s)", len(prompts), dur, float64(len(prompts))/dur.Seconds())
}

// TestClassify_WithLogits validates that logits are returned when requested.
func TestClassify_WithLogits(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	results, err := m.Classify(ctx, []string{"Hello world"}, inference.WithLogits())
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if len(results[0].Logits) == 0 {
		t.Fatal("expected non-empty logits with WithLogits()")
	}
	t.Logf("Logits length: %d (vocab size)", len(results[0].Logits))
}

// TestBatchGenerate validates batched autoregressive generation.
func TestBatchGenerate(t *testing.T) {
	modelPath := gemma3ModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	prompts := []string{
		"The capital of France is",
		"2 + 2 =",
	}

	start := time.Now()
	results, err := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(16))
	dur := time.Since(start)
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}

	if len(results) != len(prompts) {
		t.Fatalf("BatchGenerate returned %d results, want %d", len(results), len(prompts))
	}

	for i, r := range results {
		if r.Err != nil {
			t.Errorf("prompt %d error: %v", i, r.Err)
			continue
		}
		if len(r.Tokens) == 0 {
			t.Errorf("prompt %d: no tokens generated", i)
			continue
		}
		var output strings.Builder
		for _, tok := range r.Tokens {
			output.WriteString(tok.Text)
		}
		t.Logf("prompt %d %q → %d tokens: %s", i, prompts[i], len(r.Tokens), output.String())
	}
	t.Logf("Batch generated in %s", dur)
}

// TestLlama_Chat validates chat template for Llama 3 models.
func TestLlama_Chat(t *testing.T) {
	modelPath := llamaModelPath(t)

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	ctx := context.Background()
	var output strings.Builder
	var count int
	for tok := range m.Chat(ctx, []inference.Message{
		{Role: "user", Content: "Reply with exactly one word: the capital of France."},
	}, inference.WithMaxTokens(32)) {
		output.WriteString(tok.Text)
		count++
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if count == 0 {
		t.Fatal("Chat produced no tokens")
	}
	t.Logf("Chat output (%d tokens): %s", count, output.String())
}
