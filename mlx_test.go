//go:build darwin && arm64

package mlx_test

import (
	"context"
	"os"
	"testing"

	"forge.lthn.ai/core/go-mlx"
)

func TestMetalAvailable(t *testing.T) {
	if !mlx.MetalAvailable() {
		t.Fatal("MetalAvailable() = false on darwin/arm64")
	}
}

func TestDefaultBackend(t *testing.T) {
	b, err := mlx.Default()
	if err != nil {
		t.Fatalf("Default() error: %v", err)
	}
	if b.Name() != "metal" {
		t.Errorf("Default().Name() = %q, want %q", b.Name(), "metal")
	}
}

func TestGetBackend(t *testing.T) {
	b, ok := mlx.Get("metal")
	if !ok {
		t.Fatal("Get(\"metal\") returned false")
	}
	if b.Name() != "metal" {
		t.Errorf("Name() = %q, want %q", b.Name(), "metal")
	}

	_, ok = mlx.Get("nonexistent")
	if ok {
		t.Error("Get(\"nonexistent\") should return false")
	}
}

func TestLoadModel_NoBackend(t *testing.T) {
	_, err := mlx.LoadModel("/nonexistent/path")
	if err == nil {
		t.Error("expected error for nonexistent model path")
	}
}

func TestLoadModel_WithBackend(t *testing.T) {
	_, err := mlx.LoadModel("/nonexistent/path", mlx.WithBackend("nonexistent"))
	if err == nil {
		t.Error("expected error for nonexistent backend")
	}
}

func TestOptions(t *testing.T) {
	cfg := mlx.ApplyGenerateOpts([]mlx.GenerateOption{
		mlx.WithMaxTokens(64),
		mlx.WithTemperature(0.7),
		mlx.WithTopK(40),
		mlx.WithTopP(0.9),
		mlx.WithStopTokens(1, 2, 3),
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
}

func TestDefaults(t *testing.T) {
	cfg := mlx.DefaultGenerateConfig()
	if cfg.MaxTokens != 256 {
		t.Errorf("default MaxTokens = %d, want 256", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.0 {
		t.Errorf("default Temperature = %f, want 0.0", cfg.Temperature)
	}
}

// TestLoadModel_Generate requires a model on disk. Skipped in CI.
func TestLoadModel_Generate(t *testing.T) {
	const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available at", modelPath)
	}

	m, err := mlx.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	if m.ModelType() != "gemma3" {
		t.Errorf("ModelType() = %q, want %q", m.ModelType(), "gemma3")
	}

	ctx := context.Background()
	var count int
	for tok := range m.Generate(ctx, "What is 2+2?", mlx.WithMaxTokens(16)) {
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
