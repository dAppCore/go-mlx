// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// modelConfigProbe is the loose JSON shape used to inspect HuggingFace
// config.json before deciding pack metadata. Shared by model_pack.go.
type modelConfigProbe struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
	NumLabels             int      `json:"num_labels"`
	TextConfig            struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		HiddenSize            int    `json:"hidden_size"`
		NumHiddenLayers       int    `json:"num_hidden_layers"`
		MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	} `json:"text_config"`
	Quantization *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization"`
	QuantizationConfig *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization_config"`
}

// readModelConfig reads + decodes config.json from a model directory.
//
//	probe, err := readModelConfig(modelDir)
func readModelConfig(dir string) (*modelConfigProbe, error) {
	return readModelConfigAt(core.PathJoin(dir, "config.json"))
}

// readModelConfigAt reads + decodes config.json from a pre-built path.
// Used by inspectModelPackConfig to reuse the path it already builds
// for issue reporting — avoids redoing filepath.Join.
func readModelConfigAt(path string) (*modelConfigProbe, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	var config modelConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return nil, result.Value.(error)
	}
	return &config, nil
}

func (probe *modelConfigProbe) architecture() string {
	if probe == nil {
		return ""
	}
	// Resolve architectures[] once: bert_rerank takes priority over
	// ModelType (cross-encoders carry it in the class name). Only the
	// bert_rerank case can short-circuit; firstResolved is the fallback
	// when neither ModelType nor TextConfig.ModelType is set, so we
	// only compute it if we'll actually need it — skipping the
	// classify-and-discard work when ModelType already covers us.
	needFirstResolved := probe.ModelType == "" && probe.TextConfig.ModelType == ""
	var firstResolved string
	for _, architecture := range probe.Architectures {
		modelType := architectureFromTransformersName(architecture)
		if modelType == "bert_rerank" {
			return modelType
		}
		if needFirstResolved && modelType != "" && firstResolved == "" {
			firstResolved = modelType
		}
	}
	if probe.ModelType != "" {
		modelType := normalizeKnownArchitecture(probe.ModelType)
		if modelType == "gemma4" && normalizeKnownArchitecture(probe.TextConfig.ModelType) == "gemma4_text" {
			return "gemma4_text"
		}
		return modelType
	}
	if probe.TextConfig.ModelType != "" {
		return normalizeKnownArchitecture(probe.TextConfig.ModelType)
	}
	return firstResolved
}

func (probe *modelConfigProbe) numLayers() int {
	if probe == nil {
		return 0
	}
	if probe.NumHiddenLayers > 0 {
		return probe.NumHiddenLayers
	}
	return probe.TextConfig.NumHiddenLayers
}

func (probe *modelConfigProbe) vocabSize() int {
	if probe == nil {
		return 0
	}
	if probe.VocabSize > 0 {
		return probe.VocabSize
	}
	return probe.TextConfig.VocabSize
}

func (probe *modelConfigProbe) hiddenSize() int {
	if probe == nil {
		return 0
	}
	if probe.HiddenSize > 0 {
		return probe.HiddenSize
	}
	return probe.TextConfig.HiddenSize
}

func (probe *modelConfigProbe) contextLength() int {
	if probe == nil {
		return 0
	}
	if probe.MaxPositionEmbeddings > 0 {
		return probe.MaxPositionEmbeddings
	}
	return probe.TextConfig.MaxPositionEmbeddings
}

func (probe *modelConfigProbe) quantBits() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.Bits
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.Bits
	}
	return 0
}

func (probe *modelConfigProbe) quantGroup() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.GroupSize
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.GroupSize
	}
	return 0
}

// normalizeKnownArchitecture canonicalises an architecture identifier
// across HF/JANG variations. Shared between modelConfigProbe and
// architectureFromTransformersName.
//
//	id := normalizeKnownArchitecture("MiniMax-M2")  // → "minimax_m2"
func normalizeKnownArchitecture(value string) string {
	value = normalizeASCIIIdentifier(value)
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
	case "mistral":
		return "mistral"
	case "phi", "phi3", "phi4":
		return "phi"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "bert":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	default:
		return value
	}
}

// architectureFromTransformersName maps a HuggingFace transformers
// architecture class name (e.g. "Qwen2ForCausalLM") to a canonical
// model-type id used by go-mlx.
//
//	id := architectureFromTransformersName("Qwen3MoeForCausalLM")  // → "qwen3_moe"
func architectureFromTransformersName(architecture string) string {
	compact := compactArchitectureName(architecture)
	switch {
	case core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification"):
		return "bert_rerank"
	case core.Contains(compact, "qwen35moe") || core.Contains(compact, "qwen36moe"):
		return "qwen3_6_moe"
	case core.Contains(compact, "qwen35") || core.Contains(compact, "qwen36"):
		return "qwen3_6"
	case core.Contains(compact, "qwen3moe"):
		return "qwen3_moe"
	case core.Contains(compact, "qwen3next"):
		return "qwen3_next"
	case core.Contains(compact, "gemma4assistant"):
		return "gemma4_assistant"
	case core.Contains(architecture, "Gemma4"):
		return "gemma4_text"
	case core.Contains(architecture, "Gemma3"):
		return "gemma3"
	case core.Contains(architecture, "Gemma2"):
		return "gemma2"
	case core.Contains(architecture, "Qwen3"):
		return "qwen3"
	case core.Contains(architecture, "Qwen2"):
		return "qwen2"
	case core.Contains(architecture, "Llama"):
		return "llama"
	case core.Contains(architecture, "MiniMaxM2"):
		return "minimax_m2"
	case core.Contains(architecture, "Mixtral"):
		return "mixtral"
	case core.Contains(architecture, "Mistral"):
		return "mistral"
	case core.Contains(architecture, "Phi"):
		return "phi"
	case core.Contains(architecture, "Deepseek") || core.Contains(architecture, "DeepSeek"):
		return "deepseek"
	case core.Contains(architecture, "GptOss") || core.Contains(architecture, "GPTOSS"):
		return "gpt_oss"
	case core.Contains(architecture, "Bert"):
		return "bert"
	default:
		return ""
	}
}

func compactArchitectureName(value string) string {
	buf := make([]byte, 0, len(value))
	for i := 0; i < len(value); i++ {
		c := value[i]
		switch c {
		case '_', '-', '.':
			continue
		}
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf = append(buf, c)
	}
	return core.AsString(buf)
}

// normalizeASCIIIdentifier trims ASCII whitespace, lowercases A-Z, and
// folds '-' and '.' to '_' in a single pass. Used by
// normalizeKnownArchitecture for the canonicalisation step before the
// known-id switch. Architecture identifiers are pure ASCII so the
// scalar byte loop replaces three Replace/Lower/Trim allocations with
// at most one.
//
//	id := normalizeASCIIIdentifier("  MiniMax-M2 ")  // → "minimax_m2"
func normalizeASCIIIdentifier(value string) string {
	start := 0
	end := len(value)
	for start < end {
		c := value[start]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		start++
	}
	for end > start {
		c := value[end-1]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		end--
	}
	needsChange := false
	for i := start; i < end; i++ {
		c := value[i]
		if (c >= 'A' && c <= 'Z') || c == '-' || c == '.' {
			needsChange = true
			break
		}
	}
	if !needsChange {
		if start == 0 && end == len(value) {
			return value
		}
		return value[start:end]
	}
	buf := make([]byte, end-start)
	for i := range buf {
		c := value[start+i]
		switch c {
		case '-', '.':
			c = '_'
		default:
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
		}
		buf[i] = c
	}
	return core.AsString(buf)
}
