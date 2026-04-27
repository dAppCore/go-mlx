// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go/core"

	coreio "dappco.re/go/io"
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

func weightCandidates(name string) []string {
	candidates := []string{name}
	if core.HasPrefix(name, "model.") {
		suffix := core.TrimPrefix(name, "model.")
		return append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	return append(candidates,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

// resolveWeight looks up a weight with optional "language_model." prefix.
func resolveWeight(weights map[string]*Array, name string) *Array {
	for _, candidate := range weightCandidates(name) {
		if w, ok := weights[candidate]; ok {
			return w
		}
	}
	return nil
}

func hasResolvedWeight(weights map[string]*Array, name string) bool {
	for _, candidate := range weightCandidates(name) {
		if _, ok := weights[candidate]; ok {
			return true
		}
	}
	return false
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
		case core.Contains(arch, "Gemma4ForConditionalGeneration"),
			core.Contains(arch, "Gemma4Multimodal"),
			core.Contains(arch, "Gemma4Vision"):
			return "gemma4", nil
		case core.Contains(arch, "Gemma4"):
			return "gemma4_text", nil
		case core.Contains(arch, "Gemma3"):
			return "gemma3", nil
		case core.Contains(arch, "Gemma2"):
			return "gemma2", nil
		case core.Contains(arch, "Qwen3"):
			return "qwen3", nil
		case core.Contains(arch, "Qwen2"):
			return "qwen2", nil
		case core.Contains(arch, "Llama"):
			return "llama", nil
		}
	}
	return "", nil
}

func loadGemma4TextModel(modelPath string) (*Gemma4Model, error) {
	m, err := LoadGemma4(modelPath)
	if err != nil {
		return nil, err
	}
	if m.VisionTower != nil || m.MultiModalProjector != nil {
		closeGemma4Vision(m.VisionTower, m.MultiModalProjector)
		m.VisionTower = nil
		m.MultiModalProjector = nil
		ClearCache()
	}
	m.modelType = "gemma4_text"
	if m.Cfg != nil {
		m.Cfg.ModelType = "gemma4_text"
		m.Cfg.VisionConfig = nil
	}
	return m, nil
}

func loadGemma4MultiModalModel(modelPath string) (*Gemma4Model, error) {
	m, err := LoadGemma4(modelPath)
	if err != nil {
		return nil, err
	}
	m.modelType = "gemma4"
	if m.Cfg != nil {
		m.Cfg.ModelType = "gemma4"
	}
	return m, nil
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
	case "gemma4_text":
		return loadGemma4TextModel(modelPath)
	case "gemma4":
		return loadGemma4MultiModalModel(modelPath)
	default:
		return nil, core.E("model.loadModel", "unsupported architecture: "+modelType, nil)
	}
}
