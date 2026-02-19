//go:build darwin && arm64

package metal

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// InternalModel is the common interface for all transformer model architectures.
type InternalModel interface {
	// Forward runs the model forward pass on token IDs with KV caches.
	Forward(tokens *Array, caches []Cache) *Array

	// NewCache creates per-layer KV caches for generation.
	NewCache() []Cache

	// NumLayers returns the number of transformer layers.
	NumLayers() int

	// Tokenizer returns the model's tokenizer.
	Tokenizer() *Tokenizer

	// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
	ModelType() string

	// ApplyLoRA wraps target projection layers with LoRA adapters for training.
	// Returns the adapter which holds references to all LoRA layers.
	ApplyLoRA(cfg LoRAConfig) *LoRAAdapter
}

// QuantizationConfig holds quantization parameters from config.json.
type QuantizationConfig struct {
	GroupSize int `json:"group_size"`
	Bits      int `json:"bits"`
}

// resolveWeight looks up a weight with optional "language_model." prefix.
func resolveWeight(weights map[string]*Array, name string) *Array {
	if w, ok := weights[name]; ok {
		return w
	}
	if w, ok := weights["language_model."+name]; ok {
		return w
	}
	return nil
}

// loadModel auto-detects the model architecture from config.json and loads it.
func loadModel(modelPath string) (InternalModel, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("model: load config: %w", err)
	}

	var probe struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("model: parse model_type: %w", err)
	}

	switch probe.ModelType {
	case "qwen3", "qwen2":
		return LoadQwen3(modelPath)
	case "gemma3", "gemma3_text", "gemma2":
		return LoadGemma3(modelPath)
	default:
		return nil, fmt.Errorf("model: unsupported architecture %q", probe.ModelType)
	}
}
