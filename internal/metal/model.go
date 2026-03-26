//go:build darwin && arm64

package metal

import (
	"dappco.re/go/core"

	coreio "forge.lthn.ai/core/go-io"
	coreerr "forge.lthn.ai/core/go-log"
)

// InternalModel is the common interface for all transformer model architectures.
type InternalModel interface {
	// Forward runs the model forward pass on token IDs with KV caches.
	Forward(tokens *Array, caches []Cache) *Array

	// ForwardMasked runs the forward pass with an explicit attention mask.
	// mask shape: [B, 1, L, L] — additive mask (0 = attend, -inf = ignore).
	// Used for batched inference with padded sequences.
	ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array

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
	str, err := coreio.Local.Read(core.JoinPath(modelPath, "config.json"))
	if err != nil {
		return nil, coreerr.E("model.loadModel", "load config", err)
	}
	data := []byte(str)

	var probe struct {
		ModelType string `json:"model_type"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return nil, coreerr.E("model.loadModel", "parse model_type", nil)
	}

	switch probe.ModelType {
	case "qwen3", "qwen2", "llama":
		return LoadQwen3(modelPath)
	case "gemma3", "gemma3_text", "gemma2":
		return LoadGemma3(modelPath)
	default:
		return nil, coreerr.E("model.loadModel", "unsupported architecture: "+probe.ModelType, nil)
	}
}
