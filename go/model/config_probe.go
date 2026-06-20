// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/profile"
)

// modelConfigProbe is the loose JSON shape used to inspect HuggingFace
// config.json before deciding pack metadata. Shared by model_pack.go.
type modelConfigProbe struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	HeadDim               int      `json:"head_dim"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
	NumLabels             int      `json:"num_labels"`
	TextConfig            struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		HiddenSize            int    `json:"hidden_size"`
		NumHiddenLayers       int    `json:"num_hidden_layers"`
		NumKeyValueHeads      int    `json:"num_key_value_heads"`
		HeadDim               int    `json:"head_dim"`
		MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	} `json:"text_config"`
	Quantization       quantBlock `json:"quantization"`
	QuantizationConfig quantBlock `json:"quantization_config"`
}

// quantBlock is the {bits, group_size} shape carried by the config.json
// "quantization" / "quantization_config" objects. Stored as a value (with
// an explicit Present flag) rather than a *struct so the walker can fill it
// in place inside the already-heap-allocated probe — the pointer form cost
// one extra heap allocation per config that declares a quant block, on the
// once-per-Inspect parse path that model-picker / HF discovery sweeps
// multiply across every candidate. Present mirrors the old pointer-nil
// semantics: it flips true the moment the block is seen, even when empty
// (`{"quantization":{}}`), matching encoding/json decoding into a *struct.
type quantBlock struct {
	Present   bool
	Bits      int `json:"bits"`
	GroupSize int `json:"group_size"`
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
	// Walk config.json with the hand-rolled probe walker directly rather
	// than through core.JSONUnmarshal: the stdlib entry point runs a
	// checkValid pre-scan that allocates a scanner parse-state stack on
	// every call (≈3 allocs + 170 B), overhead the walker cannot dodge
	// while reached via json.Unmarshal. parseConfigProbeStrict skips that
	// scan and re-applies the same trailing-byte rule, so the error
	// contract is byte-identical (a NotExist read still surfaces above;
	// a malformed body still returns a non-nil parse error here).
	var config modelConfigProbe
	if err := parseConfigProbeStrict(read.Value.([]byte), &config); err != nil {
		return nil, err
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
		modelType := profile.ArchitectureFromTransformersName(architecture)
		if modelType == "bert_rerank" {
			return modelType
		}
		if needFirstResolved && modelType != "" && firstResolved == "" {
			firstResolved = modelType
		}
	}
	if probe.ModelType != "" {
		modelType := profile.NormalizeArchitecture(probe.ModelType)
		if probe.ModelType == "gemma4" && modelType == "gemma4" && profile.NormalizeArchitecture(probe.TextConfig.ModelType) == "gemma4_text" {
			return "gemma4_text"
		}
		return modelType
	}
	if probe.TextConfig.ModelType != "" {
		return profile.NormalizeArchitecture(probe.TextConfig.ModelType)
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

// numKeyValueHeads reports the declared KV head count — the GQA width that,
// with headDim, gives the true per-token KV-cache size (far smaller than
// hidden_size under grouped-query attention). The memory planner uses it to
// size context from the real cache cost, not a hidden-size over-estimate.
func (probe *modelConfigProbe) numKeyValueHeads() int {
	if probe == nil {
		return 0
	}
	if probe.NumKeyValueHeads > 0 {
		return probe.NumKeyValueHeads
	}
	return probe.TextConfig.NumKeyValueHeads
}

// headDim reports the declared per-head dimension (num_kv_heads * head_dim is
// the KV width per layer). Zero when the config does not declare it.
func (probe *modelConfigProbe) headDim() int {
	if probe == nil {
		return 0
	}
	if probe.HeadDim > 0 {
		return probe.HeadDim
	}
	return probe.TextConfig.HeadDim
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
	if probe.Quantization.Present {
		return probe.Quantization.Bits
	}
	if probe.QuantizationConfig.Present {
		return probe.QuantizationConfig.Bits
	}
	return 0
}

func (probe *modelConfigProbe) quantGroup() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization.Present {
		return probe.Quantization.GroupSize
	}
	if probe.QuantizationConfig.Present {
		return probe.QuantizationConfig.GroupSize
	}
	return 0
}
