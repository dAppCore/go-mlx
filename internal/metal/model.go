//go:build darwin && arm64

package metal

import (
	"strings"

	"dappco.re/go/core"

	coreio "dappco.re/go/core/io"
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
	candidates := []string{name}
	if strings.HasPrefix(name, "model.") {
		suffix := strings.TrimPrefix(name, "model.")
		candidates = append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	} else {
		candidates = append(candidates,
			"model."+name,
			"language_model."+name,
			"language_model.model."+name,
			"model.language_model."+name,
			"model.language_model.model."+name,
		)
	}
	for _, candidate := range candidates {
		if w, ok := weights[candidate]; ok {
			return w
		}
	}
	return nil
}

func probeModelType(data []byte) (string, error) {
	var probe struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return "", core.E("model.probeModelType", "parse model_type", nil)
	}
	if probe.ModelType != "" {
		return probe.ModelType, nil
	}
	if probe.TextConfig.ModelType != "" {
		return probe.TextConfig.ModelType, nil
	}
	for _, arch := range probe.Architectures {
		switch {
		case strings.Contains(arch, "Gemma4"):
			return "gemma4_text", nil
		case strings.Contains(arch, "Gemma3"):
			return "gemma3", nil
		case strings.Contains(arch, "Gemma2"):
			return "gemma2", nil
		case strings.Contains(arch, "Qwen3"):
			return "qwen3", nil
		case strings.Contains(arch, "Qwen2"):
			return "qwen2", nil
		case strings.Contains(arch, "Llama"):
			return "llama", nil
		}
	}
	return "", nil
}

// loadModel auto-detects the model architecture from config.json and loads it.
// Supports "gemma3", "gemma3_text", "gemma2", "gemma4", "gemma4_text",
// "qwen3", "qwen2", and "llama".
func loadModel(modelPath string) (InternalModel, error) {
	root := resolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("model.loadModel", "load config", err)
	}
	data := []byte(str)

	modelType, err := probeModelType(data)
	if err != nil {
		return nil, core.E("model.loadModel", "parse model_type", err)
	}

	switch modelType {
	case "qwen3", "qwen2", "llama":
		return LoadQwen3(modelPath)
	case "gemma3", "gemma3_text", "gemma2":
		return LoadGemma3(modelPath)
	case "gemma4", "gemma4_text":
		return LoadGemma4(modelPath)
	default:
		return nil, core.E("model.loadModel", "unsupported architecture: "+modelType, nil)
	}
}
