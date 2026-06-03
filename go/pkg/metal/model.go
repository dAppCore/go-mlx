// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"

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

// LastTokenLogitsModel is an optional fast prefill path for architectures that
// can project only the final sequence position instead of allocating
// [batch, sequence, vocab] logits for long context warmup.
type LastTokenLogitsModel interface {
	ForwardLastTokenLogits(tokens *Array, mask *Array, caches []Cache) *Array
}

// GreedyTokenModel is an optional decode path for deterministic generation.
// It returns the next token directly, avoiding a retained logits tensor when
// sampling is exactly greedy and no repeat penalty or probe sink is active.
type GreedyTokenModel interface {
	ForwardGreedyToken(tokens *Array, mask *Array, caches []Cache) *Array
}

// SuppressedGreedyTokenModel can produce a greedy token while masking out
// template or modality token IDs that must not be sampled.
type SuppressedGreedyTokenModel interface {
	ForwardGreedyTokenWithSuppression(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32) *Array
}

// QuantizationConfig holds quantization parameters from config.json.
type QuantizationConfig struct {
	GroupSize int    `json:"group_size"`
	Bits      int    `json:"bits"`
	Mode      string `json:"mode"`
}

func normalizeQuantizationMode(mode string) string {
	mode = core.Lower(core.Trim(mode))
	if mode == "" {
		return "affine"
	}
	return mode
}

func isAffineQuantizationMode(mode string) bool {
	return normalizeQuantizationMode(mode) == "affine"
}

func requiresDenseQuantizedMatmulFallback(mode string) bool {
	// Older local metallib builds exposed MXFP8 dequantize without MXFP8 qmm.
	// Keep a diagnostic fallback available, but prefer native MLX kernels by
	// default on v0.31.1+.
	return normalizeQuantizationMode(mode) == "mxfp8" &&
		core.Env("GO_MLX_ENABLE_MXFP8_DENSE_FALLBACK") == "1"
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
		modelType := normalizeProbeModelType(probe.ModelType)
		if modelType == "gemma4" && normalizeProbeModelType(probe.TextConfig.ModelType) == "gemma4_text" {
			return "gemma4_text", nil
		}
		if modelType == "bert" && architecturesContainRerankModel(probe.Architectures) {
			return "bert_rerank", nil
		}
		return modelType, nil
	}
	if probe.TextConfig.ModelType != "" {
		return normalizeProbeModelType(probe.TextConfig.ModelType), nil
	}
	for _, arch := range probe.Architectures {
		switch {
		case isQwen36MoEArchitecture(arch):
			return "qwen3_6_moe", nil
		case isQwen36Architecture(arch):
			return "qwen3_6", nil
		case isQwen3MoEArchitecture(arch):
			return "qwen3_moe", nil
		case isQwen3NextArchitecture(arch):
			return "qwen3_next", nil
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
		case core.Contains(arch, "Mistral"):
			return "mistral", nil
		case core.Contains(arch, "Hermes"):
			return "hermes", nil
		case core.Contains(arch, "Granite"):
			return "granite", nil
		case core.Contains(arch, "Phi"):
			return "phi", nil
		case core.Contains(arch, "Glm") || core.Contains(arch, "GLM"):
			return "glm", nil
		case core.Contains(arch, "MiniMaxM2"):
			return "minimax_m2", nil
		case core.Contains(arch, "Mixtral"):
			return "mixtral", nil
		case core.Contains(arch, "Deepseek") || core.Contains(arch, "DeepSeek"):
			return "deepseek", nil
		case core.Contains(arch, "GptOss") || core.Contains(arch, "GPTOSS"):
			return "gpt_oss", nil
		case core.Contains(arch, "Kimi") || core.Contains(arch, "Moonshot"):
			return "kimi", nil
		case core.Contains(arch, "BertForSequenceClassification") ||
			core.Contains(arch, "RobertaForSequenceClassification") ||
			core.Contains(arch, "XLMRobertaForSequenceClassification") ||
			core.Contains(arch, "DebertaV2ForSequenceClassification"):
			return "bert_rerank", nil
		case core.Contains(arch, "Bert"):
			return "bert", nil
		}
	}
	return "", nil
}

func architecturesContainRerankModel(architectures []string) bool {
	for _, arch := range architectures {
		if core.Contains(arch, "BertForSequenceClassification") ||
			core.Contains(arch, "RobertaForSequenceClassification") ||
			core.Contains(arch, "XLMRobertaForSequenceClassification") ||
			core.Contains(arch, "DebertaV2ForSequenceClassification") {
			return true
		}
	}
	return false
}

func normalizeProbeModelType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, ".", "_")
	switch value {
	case "qwen2_5", "qwen25":
		return "qwen2"
	case "qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_6_text", "qwen35", "qwen36":
		return "qwen3_6"
	case "qwen3_5_moe", "qwen3_6_moe", "qwen35_moe", "qwen36_moe":
		return "qwen3_6_moe"
	case "minimaxm2", "minimax_m2":
		return "minimax_m2"
	case "mixtral":
		return "mixtral"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "kimi", "moonshot":
		return "kimi"
	case "bert", "bert_model":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	case "phi3", "phi4":
		return "phi"
	default:
		return value
	}
}

func compactArchitectureName(value string) string {
	compact := core.Lower(value)
	compact = core.Replace(compact, "_", "")
	compact = core.Replace(compact, "-", "")
	return core.Replace(compact, ".", "")
}

func isQwen36MoEArchitecture(value string) bool {
	compact := compactArchitectureName(value)
	return core.Contains(compact, "qwen35moe") || core.Contains(compact, "qwen36moe")
}

func isQwen36Architecture(value string) bool {
	compact := compactArchitectureName(value)
	return core.Contains(compact, "qwen35") || core.Contains(compact, "qwen36")
}

func isQwen3MoEArchitecture(value string) bool {
	return core.Contains(compactArchitectureName(value), "qwen3moe")
}

func isQwen3NextArchitecture(value string) bool {
	return core.Contains(compactArchitectureName(value), "qwen3next")
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
// "qwen3", "qwen3_next", "qwen2", "llama", and recognized staged
// architectures such as "qwen3_6" and "minimax_m2". Gemma 4 assistant checkpoints are
// attached MTP drafters; load them through LoadGemma4AssistantPair or the
// public LoadSpeculativePair path rather than as standalone InternalModel
// values.
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

	// gemma4_assistant is an attached MTP drafter, not a standalone model.
	if modelType == "gemma4_assistant" {
		return nil, core.E("model.loadModel", "gemma4_assistant is an attached MTP drafter; use LoadSpeculativePair or LoadGemma4AssistantPair with a Gemma 4 target", nil)
	}
	// Dispatch via the loader registry (model_registry.go) — no central switch.
	if loader := lookupModelLoader(modelType); loader != nil {
		return loader(modelPath, data)
	}
	return nil, core.E("model.loadModel", "unsupported architecture: "+modelType, nil)
}
