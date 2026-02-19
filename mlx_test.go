//go:build darwin && arm64

package mlx_test

import (
	"context"
	"os"
	"testing"

	"forge.lthn.ai/core/go-inference"
	_ "forge.lthn.ai/core/go-mlx"
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

// TestLoadModel_Generate requires a model on disk. Skipped in CI.
func TestLoadModel_Generate(t *testing.T) {
	const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available at", modelPath)
	}

	m, err := inference.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

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
