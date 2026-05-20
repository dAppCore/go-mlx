// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"io"
	"io/fs"
	"sort"
	"strconv"

	core "dappco.re/go"
)

const maxGGUFCollectionEntries uint64 = 1 << 20

const (
	ggufValueTypeUint8   = 0
	ggufValueTypeInt8    = 1
	ggufValueTypeUint16  = 2
	ggufValueTypeInt16   = 3
	ValueTypeUint32      = 4
	ggufValueTypeInt32   = 5
	ggufValueTypeFloat32 = 6
	ggufValueTypeBool    = 7
	ValueTypeString      = 8
	ggufValueTypeArray   = 9
	ggufValueTypeUint64  = 10
	ggufValueTypeInt64   = 11
	ggufValueTypeFloat64 = 12
)

const (
	ggufTensorTypeF32      = 0
	ggufTensorTypeF16      = 1
	TensorTypeQ4_0         = 2
	ggufTensorTypeQ4_1     = 3
	ggufTensorTypeQ5_0     = 6
	ggufTensorTypeQ5_1     = 7
	TensorTypeQ8_0         = 8
	ggufTensorTypeQ8_1     = 9
	ggufTensorTypeQ2K      = 10
	ggufTensorTypeQ3K      = 11
	ggufTensorTypeQ4K      = 12
	ggufTensorTypeQ5K      = 13
	ggufTensorTypeQ6K      = 14
	ggufTensorTypeQ8K      = 15
	ggufTensorTypeIQ2XXS   = 16
	ggufTensorTypeIQ2XS    = 17
	ggufTensorTypeIQ3XXS   = 18
	ggufTensorTypeIQ1S     = 19
	ggufTensorTypeIQ4NL    = 20
	ggufTensorTypeIQ3S     = 21
	ggufTensorTypeIQ2S     = 22
	ggufTensorTypeIQ4XS    = 23
	ggufTensorTypeI8       = 24
	ggufTensorTypeI16      = 25
	ggufTensorTypeI32      = 26
	ggufTensorTypeI64      = 27
	ggufTensorTypeF64      = 28
	ggufTensorTypeIQ1M     = 29
	ggufTensorTypeBF16     = 30
	ggufTensorTypeQ4_0_4_4 = 31
	ggufTensorTypeQ4_0_4_8 = 32
	ggufTensorTypeQ4_0_8_8 = 33
	ggufTensorTypeTQ1_0    = 34
	ggufTensorTypeTQ2_0    = 35
	ggufTensorTypeMXFP4    = 38
	ggufTensorTypeNVFP4    = 39
)

// Info summarises the metadata of a GGUF checkpoint.
type Info struct {
	Path             string
	Architecture     string
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	ContextLength    int
	QuantBits        int
	QuantGroup       int
	QuantType        string
	QuantFamily      string
	Quantization     QuantizationInfo
	Tensors          []TensorInfo
	ValidationIssues []ValidationIssue
	TensorCount      int
	MetadataCount    int
}

// Valid reports whether tensor metadata passed basic shape/dtype validation.
func (info Info) Valid() bool {
	for _, issue := range info.ValidationIssues {
		if issue.Severity == GGUFValidationError {
			return false
		}
	}
	return true
}

// ValidationSeverity classifies GGUF metadata validation findings.
type ValidationSeverity string

const (
	GGUFValidationWarning ValidationSeverity = "warning"
	GGUFValidationError   ValidationSeverity = "error"
)

// ValidationIssue describes one GGUF tensor metadata validation issue.
type ValidationIssue struct {
	Severity ValidationSeverity `json:"severity"`
	Code     string             `json:"code"`
	Message  string             `json:"message"`
	Tensor   string             `json:"tensor,omitempty"`
}

// TensorInfo describes one tensor entry from the GGUF directory.
type TensorInfo struct {
	Name      string   `json:"name"`
	Type      uint32   `json:"type"`
	TypeName  string   `json:"type_name,omitempty"`
	DType     string   `json:"dtype,omitempty"`
	Bits      int      `json:"bits,omitempty"`
	BlockSize int      `json:"block_size,omitempty"`
	Shape     []uint64 `json:"shape,omitempty"`
	Elements  uint64   `json:"elements,omitempty"`
	Offset    uint64   `json:"offset,omitempty"`
	Quantized bool     `json:"quantized,omitempty"`
}

// TensorTypeSummary counts tensor dtypes found in a GGUF file.
type TensorTypeSummary struct {
	Type      uint32 `json:"type"`
	Name      string `json:"name"`
	DType     string `json:"dtype,omitempty"`
	Bits      int    `json:"bits,omitempty"`
	BlockSize int    `json:"block_size,omitempty"`
	Count     int    `json:"count"`
	Quantized bool   `json:"quantized,omitempty"`
}

// QuantizationInfo captures GGML quantization metadata beyond bit width.
type QuantizationInfo struct {
	Type         string              `json:"type,omitempty"`
	Family       string              `json:"family,omitempty"`
	Bits         int                 `json:"bits,omitempty"`
	GroupSize    int                 `json:"group_size,omitempty"`
	FileType     int                 `json:"file_type,omitempty"`
	FileTypeName string              `json:"file_type_name,omitempty"`
	Version      int                 `json:"version,omitempty"`
	Mixed        bool                `json:"mixed,omitempty"`
	TensorTypes  []TensorTypeSummary `json:"tensor_types,omitempty"`
}

// DiscoveredModel is a loadable model discovered on disk.
type DiscoveredModel struct {
	Path        string
	ModelType   string
	QuantBits   int
	QuantGroup  int
	QuantType   string
	QuantFamily string
	NumFiles    int
	Format      string
}

type ggufTensorInfo struct {
	Name   string
	Type   uint32
	Shape  []uint64
	Offset uint64
}

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

// ReadInfo reads GGUF metadata without loading model weights into MLX.
func ReadInfo(modelPath string) (Info, error) {
	ggufPath, err := resolveGGUFFile(modelPath)
	if err != nil {
		return Info{}, err
	}

	metadata, tensors, err := parseGGUF(ggufPath)
	if err != nil {
		return Info{}, err
	}

	absolutePath := ggufPath
	if abs := core.PathAbs(ggufPath); abs.OK {
		absolutePath = abs.Value.(string)
	}

	config, _ := readModelConfig(core.PathDir(ggufPath))
	architecture := firstNonEmpty(
		metadataString(metadata["general.architecture"]),
		config.architecture(),
	)
	quantBits := config.quantBits()
	if quantBits == 0 {
		quantBits = inferQuantBits(tensors)
	}
	tensorInfos, validationIssues := buildGGUFTensorInfos(tensors)
	quantization := inferGGUFQuantization(metadata, tensorInfos)
	if quantization.Bits == 0 {
		quantization.Bits = quantBits
	}
	quantization.GroupSize = firstPositive(config.quantGroup(), quantization.GroupSize, quantizationGroupFromTensorTypes(quantization.TensorTypes))
	if quantBits == 0 {
		quantBits = quantization.Bits
	}

	info := Info{
		Path:             absolutePath,
		Architecture:     architecture,
		VocabSize:        firstPositive(config.vocabSize(), inferGGUFVocabSize(metadata, architecture)),
		HiddenSize:       firstPositive(config.hiddenSize(), inferGGUFHiddenSize(metadata, architecture)),
		NumLayers:        config.numLayers(),
		ContextLength:    firstPositive(config.contextLength(), inferGGUFContextLength(metadata, architecture)),
		QuantBits:        quantBits,
		QuantGroup:       quantization.GroupSize,
		QuantType:        quantization.Type,
		QuantFamily:      quantization.Family,
		Quantization:     quantization,
		Tensors:          tensorInfos,
		ValidationIssues: validationIssues,
		TensorCount:      len(tensors),
		MetadataCount:    len(metadata),
	}
	if info.NumLayers == 0 {
		info.NumLayers = inferLayerCount(metadata, tensors, info.Architecture)
	}

	return info, nil
}

// DiscoverModels returns loadable safetensors and GGUF models beneath basePath.
func DiscoverModels(basePath string) []DiscoveredModel {
	resolvedPath := basePath
	if abs := core.PathAbs(basePath); abs.OK {
		resolvedPath = abs.Value.(string)
	}

	if stat := core.Stat(resolvedPath); stat.OK && !stat.Value.(core.FsFileInfo).IsDir() {
		if core.HasSuffix(core.Lower(resolvedPath), ".gguf") {
			ggufInfo, err := ReadInfo(resolvedPath)
			if err == nil {
				return []DiscoveredModel{{
					Path:        ggufInfo.Path,
					ModelType:   ggufInfo.Architecture,
					QuantBits:   ggufInfo.QuantBits,
					QuantGroup:  ggufInfo.QuantGroup,
					QuantType:   ggufInfo.QuantType,
					QuantFamily: ggufInfo.QuantFamily,
					NumFiles:    1,
					Format:      "gguf",
				}}
			}
		}
		return nil
	}

	var models []DiscoveredModel
	if err := core.PathWalkDir(resolvedPath, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || !d.IsDir() {
			return nil
		}
		if model, ok := probeDiscoveredModel(path); ok {
			models = append(models, model)
		}
		return nil
	}); err != nil {
		return nil
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].Path < models[j].Path
	})
	return models
}

func probeDiscoveredModel(dir string) (DiscoveredModel, bool) {
	config, configErr := readModelConfig(dir)

	safetensors := core.PathGlob(core.PathJoin(dir, "*.safetensors"))
	if len(safetensors) > 0 {
		if configErr != nil {
			return DiscoveredModel{}, false
		}
		return DiscoveredModel{
			Path:       dir,
			ModelType:  config.architecture(),
			QuantBits:  config.quantBits(),
			QuantGroup: config.quantGroup(),
			NumFiles:   len(safetensors),
			Format:     "safetensors",
		}, true
	}

	ggufs := core.PathGlob(core.PathJoin(dir, "*.gguf"))
	if len(ggufs) != 1 {
		return DiscoveredModel{}, false
	}

	info, err := ReadInfo(ggufs[0])
	if err != nil {
		return DiscoveredModel{}, false
	}
	modelType := info.Architecture
	if modelType == "" && configErr == nil {
		modelType = config.architecture()
	}
	return DiscoveredModel{
		Path:        info.Path,
		ModelType:   modelType,
		QuantBits:   info.QuantBits,
		QuantGroup:  info.QuantGroup,
		QuantType:   info.QuantType,
		QuantFamily: info.QuantFamily,
		NumFiles:    1,
		Format:      "gguf",
	}, true
}

func resolveGGUFFile(modelPath string) (string, error) {
	if core.HasSuffix(core.Lower(modelPath), ".gguf") {
		return modelPath, nil
	}

	ggufs := core.PathGlob(core.PathJoin(modelPath, "*.gguf"))
	switch len(ggufs) {
	case 0:
		return "", core.NewError("mlx: no .gguf file found")
	case 1:
		return ggufs[0], nil
	default:
		return "", core.NewError("mlx: multiple .gguf files found")
	}
}

func parseGGUF(path string) (map[string]any, []ggufTensorInfo, error) {
	open := core.Open(path)
	if !open.OK {
		return nil, nil, core.Errorf("mlx: open gguf: %w", open.Value.(error))
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()

	var magic [4]byte
	if _, err := io.ReadFull(file, magic[:]); err != nil {
		return nil, nil, core.Errorf("mlx: read gguf magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, nil, core.NewError("mlx: invalid gguf magic")
	}

	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, nil, core.Errorf("mlx: read gguf version: %w", err)
	}
	if version < 2 {
		return nil, nil, core.Errorf("mlx: unsupported gguf version %d", version)
	}

	var tensorCount uint64
	if err := binary.Read(file, binary.LittleEndian, &tensorCount); err != nil {
		return nil, nil, core.Errorf("mlx: read gguf tensor count: %w", err)
	}
	var metadataCount uint64
	if err := binary.Read(file, binary.LittleEndian, &metadataCount); err != nil {
		return nil, nil, core.Errorf("mlx: read gguf metadata count: %w", err)
	}
	if tensorCount > maxGGUFCollectionEntries {
		return nil, nil, core.Errorf("mlx: gguf tensor count %d exceeds limit %d", tensorCount, maxGGUFCollectionEntries)
	}
	if metadataCount > maxGGUFCollectionEntries {
		return nil, nil, core.Errorf("mlx: gguf metadata count %d exceeds limit %d", metadataCount, maxGGUFCollectionEntries)
	}

	metadata := make(map[string]any, int(metadataCount))
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readGGUFString(file)
		if err != nil {
			return nil, nil, core.Errorf("mlx: read gguf metadata key: %w", err)
		}
		var valueType uint32
		if err := binary.Read(file, binary.LittleEndian, &valueType); err != nil {
			return nil, nil, core.Errorf("mlx: read gguf metadata type: %w", err)
		}
		value, err := readGGUFValue(file, valueType)
		if err != nil {
			return nil, nil, core.Errorf("mlx: read gguf metadata value for %q: %w", key, err)
		}
		metadata[key] = value
	}

	tensors := make([]ggufTensorInfo, 0, int(tensorCount))
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readGGUFString(file)
		if err != nil {
			return nil, nil, core.Errorf("mlx: read gguf tensor name: %w", err)
		}
		var ndim uint32
		if err := binary.Read(file, binary.LittleEndian, &ndim); err != nil {
			return nil, nil, core.Errorf("mlx: read gguf tensor ndim: %w", err)
		}
		shape := make([]uint64, 0, int(ndim))
		for range ndim {
			var dim uint64
			if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
				return nil, nil, core.Errorf("mlx: read gguf tensor dimension: %w", err)
			}
			shape = append(shape, dim)
		}
		var tensorType uint32
		if err := binary.Read(file, binary.LittleEndian, &tensorType); err != nil {
			return nil, nil, core.Errorf("mlx: read gguf tensor type: %w", err)
		}
		var offset uint64
		if err := binary.Read(file, binary.LittleEndian, &offset); err != nil {
			return nil, nil, core.Errorf("mlx: read gguf tensor offset: %w", err)
		}
		tensors = append(tensors, ggufTensorInfo{Name: name, Type: tensorType, Shape: shape, Offset: offset})
	}

	return metadata, tensors, nil
}

func readGGUFString(reader io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 16<<20 {
		return "", core.NewError("gguf string is unreasonably large")
	}
	buffer := make([]byte, length)
	if _, err := io.ReadFull(reader, buffer); err != nil {
		return "", err
	}
	return string(buffer), nil
}

func readGGUFValue(reader io.Reader, valueType uint32) (any, error) {
	switch valueType {
	case ggufValueTypeUint8:
		return readGGUFBinary[uint8](reader)
	case ggufValueTypeInt8:
		return readGGUFBinary[int8](reader)
	case ggufValueTypeUint16:
		return readGGUFBinary[uint16](reader)
	case ggufValueTypeInt16:
		return readGGUFBinary[int16](reader)
	case ValueTypeUint32:
		return readGGUFBinary[uint32](reader)
	case ggufValueTypeInt32:
		return readGGUFBinary[int32](reader)
	case ggufValueTypeFloat32:
		return readGGUFBinary[float32](reader)
	case ggufValueTypeBool:
		value, err := readGGUFBinary[uint8](reader)
		return value != 0, err
	case ValueTypeString:
		return readGGUFString(reader)
	case ggufValueTypeArray:
		var elementType uint32
		if err := binary.Read(reader, binary.LittleEndian, &elementType); err != nil {
			return nil, err
		}
		var length uint64
		if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		if length > maxGGUFCollectionEntries {
			return nil, core.Errorf("gguf array length %d exceeds limit %d", length, maxGGUFCollectionEntries)
		}
		values := make([]any, 0, int(length))
		for i := uint64(0); i < length; i++ {
			value, err := readGGUFValue(reader, elementType)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
		}
		return values, nil
	case ggufValueTypeUint64:
		return readGGUFBinary[uint64](reader)
	case ggufValueTypeInt64:
		return readGGUFBinary[int64](reader)
	case ggufValueTypeFloat64:
		return readGGUFBinary[float64](reader)
	default:
		return nil, core.Errorf("unsupported gguf metadata type %d", valueType)
	}
}

func readGGUFBinary[T any](reader io.Reader) (T, error) {
	var value T
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readModelConfig(dir string) (*modelConfigProbe, error) {
	read := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !read.OK {
		return nil, read.Value.(error)
	}
	var config modelConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return nil, result.Value.(error)
	}
	return &config, nil
}

func normalizeKnownArchitecture(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	switch value {
	case "qwen3_5":
		return "qwen3_next"
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

func architectureFromTransformersName(architecture string) string {
	compact := core.Lower(core.Replace(core.Replace(architecture, "_", ""), "-", ""))
	switch {
	case core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification"):
		return "bert_rerank"
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

func (probe *modelConfigProbe) architecture() string {
	if probe == nil {
		return ""
	}
	for _, architecture := range probe.Architectures {
		if modelType := architectureFromTransformersName(architecture); modelType == "bert_rerank" {
			return modelType
		}
	}
	if probe.ModelType != "" {
		return normalizeKnownArchitecture(probe.ModelType)
	}
	if probe.TextConfig.ModelType != "" {
		return normalizeKnownArchitecture(probe.TextConfig.ModelType)
	}
	for _, architecture := range probe.Architectures {
		if modelType := architectureFromTransformersName(architecture); modelType != "" {
			return modelType
		}
	}
	return ""
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

func metadataString(value any) string {
	switch concrete := value.(type) {
	case string:
		return concrete
	default:
		return ""
	}
}

func metadataInt(value any) int {
	switch concrete := value.(type) {
	case uint8:
		return int(concrete)
	case int8:
		return int(concrete)
	case uint16:
		return int(concrete)
	case int16:
		return int(concrete)
	case uint32:
		return int(concrete)
	case int32:
		return int(concrete)
	case uint64:
		return int(concrete)
	case int64:
		return int(concrete)
	case float32:
		return int(concrete)
	case float64:
		return int(concrete)
	default:
		return 0
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func inferGGUFVocabSize(metadata map[string]any, architecture string) int {
	return firstPositive(
		metadataIntForSuffix(metadata, architecture, "vocab_size", "n_vocab"),
		metadataArrayLen(metadata["tokenizer.ggml.tokens"]),
	)
}

func inferGGUFHiddenSize(metadata map[string]any, architecture string) int {
	return metadataIntForSuffix(metadata, architecture, "embedding_length", "hidden_size", "n_embd")
}

func inferGGUFContextLength(metadata map[string]any, architecture string) int {
	return metadataIntForSuffix(metadata, architecture, "context_length", "max_position_embeddings", "n_ctx")
}

func metadataIntForSuffix(metadata map[string]any, architecture string, suffixes ...string) int {
	prefixes := []string{"general"}
	if architecture != "" {
		prefixes = append([]string{architecture}, prefixes...)
		if parts := core.SplitN(architecture, "_", 2); len(parts) == 2 && parts[0] != "" && parts[0] != architecture {
			base := parts[0]
			prefixes = append([]string{base}, prefixes...)
		}
	}
	for _, prefix := range prefixes {
		for _, suffix := range suffixes {
			if value := metadataInt(metadata[prefix+"."+suffix]); value > 0 {
				return value
			}
		}
	}
	for _, suffix := range suffixes {
		if value := metadataInt(metadata[suffix]); value > 0 {
			return value
		}
	}
	return 0
}

func metadataArrayLen(value any) int {
	switch concrete := value.(type) {
	case []any:
		return len(concrete)
	case []string:
		return len(concrete)
	default:
		return 0
	}
}

func inferLayerCount(metadata map[string]any, tensors []ggufTensorInfo, architecture string) int {
	if architecture != "" {
		for _, key := range []string{
			architecture + ".block_count",
			architecture + ".n_layer",
			architecture + ".num_hidden_layers",
		} {
			if count := metadataInt(metadata[key]); count > 0 {
				return count
			}
		}
	}

	maxLayer := -1
	for _, tensor := range tensors {
		if index := extractLayerIndex(tensor.Name); index > maxLayer {
			maxLayer = index
		}
	}
	if maxLayer >= 0 {
		return maxLayer + 1
	}
	return 0
}

func extractLayerIndex(name string) int {
	for _, marker := range []string{"model.layers.", "layers.", "blk.", "block."} {
		index := indexString(name, marker)
		if index < 0 {
			continue
		}
		start := index + len(marker)
		end := start
		for end < len(name) && name[end] >= '0' && name[end] <= '9' {
			end++
		}
		if end == start {
			continue
		}
		layer, err := strconv.Atoi(name[start:end])
		if err == nil {
			return layer
		}
	}
	return -1
}

func inferQuantBits(tensors []ggufTensorInfo) int {
	counts := map[int]int{}
	for _, tensor := range tensors {
		bits := ggufTensorBits(tensor.Type)
		if bits > 0 {
			counts[bits]++
		}
	}

	bestBits := 0
	bestCount := 0
	for bits, count := range counts {
		if count > bestCount || (count == bestCount && bits > bestBits) {
			bestBits = bits
			bestCount = count
		}
	}
	return bestBits
}

func ggufTensorBits(tensorType uint32) int {
	details := ggufTensorTypeDetails(tensorType)
	if !details.Known || !details.Quantized {
		return 0
	}
	return details.Bits
}

type ggufTensorTypeDetailsInfo struct {
	Name      string
	DType     string
	Bits      int
	BlockSize int
	Quantized bool
	Known     bool
}

func ggufTensorTypeDetails(tensorType uint32) ggufTensorTypeDetailsInfo {
	switch tensorType {
	case ggufTensorTypeF32:
		return ggufTensorTypeDetailsInfo{Name: "f32", DType: "float32", Bits: 32, Known: true}
	case ggufTensorTypeF16:
		return ggufTensorTypeDetailsInfo{Name: "f16", DType: "float16", Bits: 16, Known: true}
	case TensorTypeQ4_0:
		return ggufTensorTypeDetailsInfo{Name: "q4_0", DType: "ggml_q4_0", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ4_1:
		return ggufTensorTypeDetailsInfo{Name: "q4_1", DType: "ggml_q4_1", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ5_0:
		return ggufTensorTypeDetailsInfo{Name: "q5_0", DType: "ggml_q5_0", Bits: 5, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ5_1:
		return ggufTensorTypeDetailsInfo{Name: "q5_1", DType: "ggml_q5_1", Bits: 5, BlockSize: 32, Quantized: true, Known: true}
	case TensorTypeQ8_0:
		return ggufTensorTypeDetailsInfo{Name: "q8_0", DType: "ggml_q8_0", Bits: 8, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ8_1:
		return ggufTensorTypeDetailsInfo{Name: "q8_1", DType: "ggml_q8_1", Bits: 8, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ2K:
		return ggufTensorTypeDetailsInfo{Name: "q2_k", DType: "ggml_q2_k", Bits: 2, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeQ3K:
		return ggufTensorTypeDetailsInfo{Name: "q3_k", DType: "ggml_q3_k", Bits: 3, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeQ4K:
		return ggufTensorTypeDetailsInfo{Name: "q4_k", DType: "ggml_q4_k", Bits: 4, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeQ5K:
		return ggufTensorTypeDetailsInfo{Name: "q5_k", DType: "ggml_q5_k", Bits: 5, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeQ6K:
		return ggufTensorTypeDetailsInfo{Name: "q6_k", DType: "ggml_q6_k", Bits: 6, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeQ8K:
		return ggufTensorTypeDetailsInfo{Name: "q8_k", DType: "ggml_q8_k", Bits: 8, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ2XXS:
		return ggufTensorTypeDetailsInfo{Name: "iq2_xxs", DType: "ggml_iq2_xxs", Bits: 2, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ2XS:
		return ggufTensorTypeDetailsInfo{Name: "iq2_xs", DType: "ggml_iq2_xs", Bits: 2, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ3XXS:
		return ggufTensorTypeDetailsInfo{Name: "iq3_xxs", DType: "ggml_iq3_xxs", Bits: 3, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ1S:
		return ggufTensorTypeDetailsInfo{Name: "iq1_s", DType: "ggml_iq1_s", Bits: 1, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ4NL:
		return ggufTensorTypeDetailsInfo{Name: "iq4_nl", DType: "ggml_iq4_nl", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeIQ3S:
		return ggufTensorTypeDetailsInfo{Name: "iq3_s", DType: "ggml_iq3_s", Bits: 3, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ2S:
		return ggufTensorTypeDetailsInfo{Name: "iq2_s", DType: "ggml_iq2_s", Bits: 2, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeIQ4XS:
		return ggufTensorTypeDetailsInfo{Name: "iq4_xs", DType: "ggml_iq4_xs", Bits: 4, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeI8:
		return ggufTensorTypeDetailsInfo{Name: "i8", DType: "int8", Bits: 8, Known: true}
	case ggufTensorTypeI16:
		return ggufTensorTypeDetailsInfo{Name: "i16", DType: "int16", Bits: 16, Known: true}
	case ggufTensorTypeI32:
		return ggufTensorTypeDetailsInfo{Name: "i32", DType: "int32", Bits: 32, Known: true}
	case ggufTensorTypeI64:
		return ggufTensorTypeDetailsInfo{Name: "i64", DType: "int64", Bits: 64, Known: true}
	case ggufTensorTypeF64:
		return ggufTensorTypeDetailsInfo{Name: "f64", DType: "float64", Bits: 64, Known: true}
	case ggufTensorTypeIQ1M:
		return ggufTensorTypeDetailsInfo{Name: "iq1_m", DType: "ggml_iq1_m", Bits: 1, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeBF16:
		return ggufTensorTypeDetailsInfo{Name: "bf16", DType: "bfloat16", Bits: 16, Known: true}
	case ggufTensorTypeQ4_0_4_4:
		return ggufTensorTypeDetailsInfo{Name: "q4_0_4_4", DType: "ggml_q4_0_4_4", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ4_0_4_8:
		return ggufTensorTypeDetailsInfo{Name: "q4_0_4_8", DType: "ggml_q4_0_4_8", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeQ4_0_8_8:
		return ggufTensorTypeDetailsInfo{Name: "q4_0_8_8", DType: "ggml_q4_0_8_8", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeTQ1_0:
		return ggufTensorTypeDetailsInfo{Name: "tq1_0", DType: "ggml_tq1_0", Bits: 1, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeTQ2_0:
		return ggufTensorTypeDetailsInfo{Name: "tq2_0", DType: "ggml_tq2_0", Bits: 2, BlockSize: 256, Quantized: true, Known: true}
	case ggufTensorTypeMXFP4:
		return ggufTensorTypeDetailsInfo{Name: "mxfp4", DType: "ggml_mxfp4", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	case ggufTensorTypeNVFP4:
		return ggufTensorTypeDetailsInfo{Name: "nvfp4", DType: "ggml_nvfp4", Bits: 4, BlockSize: 32, Quantized: true, Known: true}
	default:
		return ggufTensorTypeDetailsInfo{}
	}
}

func buildGGUFTensorInfos(tensors []ggufTensorInfo) ([]TensorInfo, []ValidationIssue) {
	infos := make([]TensorInfo, 0, len(tensors))
	var issues []ValidationIssue
	for _, tensor := range tensors {
		details := ggufTensorTypeDetails(tensor.Type)
		info := TensorInfo{
			Name:      tensor.Name,
			Type:      tensor.Type,
			TypeName:  details.Name,
			DType:     details.DType,
			Bits:      details.Bits,
			BlockSize: details.BlockSize,
			Shape:     append([]uint64(nil), tensor.Shape...),
			Elements:  ggufTensorElements(tensor.Shape),
			Offset:    tensor.Offset,
			Quantized: details.Quantized,
		}
		infos = append(infos, info)

		if !details.Known {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "unknown_tensor_type",
				Message:  core.Sprintf("tensor has unknown GGML type id %d", tensor.Type),
				Tensor:   tensor.Name,
			})
		}
		if len(tensor.Shape) == 0 {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "invalid_tensor_shape",
				Message:  "tensor has no shape dimensions",
				Tensor:   tensor.Name,
			})
		}
		for _, dim := range tensor.Shape {
			if dim == 0 {
				issues = append(issues, ValidationIssue{
					Severity: GGUFValidationError,
					Code:     "invalid_tensor_dimension",
					Message:  "tensor shape contains a zero dimension",
					Tensor:   tensor.Name,
				})
				break
			}
		}
		if details.Known && details.Quantized && details.BlockSize > 0 && len(tensor.Shape) > 0 && tensor.Shape[0] > 0 && tensor.Shape[0]%uint64(details.BlockSize) != 0 {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "tensor_shape_not_block_aligned",
				Message:  core.Sprintf("tensor first dimension %d is not divisible by GGML block size %d", tensor.Shape[0], details.BlockSize),
				Tensor:   tensor.Name,
			})
		}
	}
	return infos, issues
}

func ggufTensorElements(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	total := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0
		}
		total *= dim
	}
	return total
}

func inferGGUFQuantization(metadata map[string]any, tensors []TensorInfo) QuantizationInfo {
	tensorTypes := summarizeGGUFTensorTypes(tensors)
	fileType, fileTypePresent := metadataIntIfPresent(metadata, "general.file_type")
	var fileTypeName string
	var fileTypeBits int
	if fileTypePresent {
		fileTypeName, fileTypeBits = ggufFileTypeQuantization(fileType)
	}
	explicitType := NormalizeQuantType(firstNonEmpty(
		metadataString(metadata["general.quantization_type"]),
		metadataString(metadata["quantization.type"]),
		metadataString(metadata["quantization.name"]),
		metadataString(metadata["general.quantization"]),
	))
	majorityType, majorityBits, majorityGroup := majorityGGUFQuantizedTensorType(tensorTypes)
	quantType := firstNonEmpty(explicitType, fileTypeName, majorityType)
	bits := firstPositive(quantBitsFromTypeName(quantType), fileTypeBits, majorityBits)
	family := quantFamilyForType(quantType)
	if family == "" && majorityType != "" {
		family = quantFamilyForType(majorityType)
	}
	group := firstPositive(metadataInt(metadata["quantization.group_size"]), metadataInt(metadata["general.quantization_group_size"]), majorityGroup)
	return QuantizationInfo{
		Type:         quantType,
		Family:       family,
		Bits:         bits,
		GroupSize:    group,
		FileType:     fileType,
		FileTypeName: fileTypeName,
		Version:      metadataInt(metadata["general.quantization_version"]),
		Mixed:        ggufQuantizationIsMixed(quantType, tensorTypes),
		TensorTypes:  tensorTypes,
	}
}

func metadataIntIfPresent(metadata map[string]any, key string) (int, bool) {
	value, ok := metadata[key]
	if !ok {
		return 0, false
	}
	return metadataInt(value), true
}

func summarizeGGUFTensorTypes(tensors []TensorInfo) []TensorTypeSummary {
	type summaryKey struct {
		typ  uint32
		name string
	}
	byType := map[summaryKey]TensorTypeSummary{}
	for _, tensor := range tensors {
		key := summaryKey{typ: tensor.Type, name: tensor.TypeName}
		summary := byType[key]
		if summary.Count == 0 {
			summary = TensorTypeSummary{
				Type:      tensor.Type,
				Name:      tensor.TypeName,
				DType:     tensor.DType,
				Bits:      tensor.Bits,
				BlockSize: tensor.BlockSize,
				Quantized: tensor.Quantized,
			}
		}
		summary.Count++
		byType[key] = summary
	}
	out := make([]TensorTypeSummary, 0, len(byType))
	for _, summary := range byType {
		out = append(out, summary)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func majorityGGUFQuantizedTensorType(summaries []TensorTypeSummary) (string, int, int) {
	var best TensorTypeSummary
	for _, summary := range summaries {
		if !summary.Quantized {
			continue
		}
		if summary.Count > best.Count || (summary.Count == best.Count && summary.Bits > best.Bits) {
			best = summary
		}
	}
	return best.Name, best.Bits, best.BlockSize
}

func quantizationGroupFromTensorTypes(summaries []TensorTypeSummary) int {
	_, _, group := majorityGGUFQuantizedTensorType(summaries)
	return group
}

func ggufFileTypeQuantization(fileType int) (string, int) {
	switch fileType {
	case 0:
		return "f32", 32
	case 1:
		return "f16", 16
	case 2:
		return "q4_0", 4
	case 3:
		return "q4_1", 4
	case 4:
		return "q4_1_some_f16", 4
	case 7:
		return "q8_0", 8
	case 8:
		return "q5_0", 5
	case 9:
		return "q5_1", 5
	case 10:
		return "q2_k", 2
	case 11:
		return "q3_k_s", 3
	case 12:
		return "q3_k_m", 3
	case 13:
		return "q3_k_l", 3
	case 14:
		return "q4_k_s", 4
	case 15:
		return "q4_k_m", 4
	case 16:
		return "q5_k_s", 5
	case 17:
		return "q5_k_m", 5
	case 18:
		return "q6_k", 6
	case 19:
		return "iq2_xxs", 2
	case 20:
		return "iq2_xs", 2
	case 21:
		return "q2_k_s", 2
	case 22:
		return "iq3_xs", 3
	case 23:
		return "iq3_xxs", 3
	case 24:
		return "iq1_s", 1
	case 25:
		return "iq4_nl", 4
	case 26:
		return "iq3_s", 3
	case 27:
		return "iq3_m", 3
	case 28:
		return "iq2_s", 2
	case 29:
		return "iq2_m", 2
	case 30:
		return "iq4_xs", 4
	case 31:
		return "iq1_m", 1
	case 32:
		return "bf16", 16
	case 33:
		return "q4_0_4_4", 4
	case 34:
		return "q4_0_4_8", 4
	case 35:
		return "q4_0_8_8", 4
	case 36:
		return "tq1_0", 1
	case 37:
		return "tq2_0", 2
	case 38:
		return "mxfp4", 4
	case 39:
		return "nvfp4", 4
	default:
		return "", 0
	}
}

func NormalizeQuantType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, " ", "_")
	return value
}

func quantBitsFromTypeName(name string) int {
	name = NormalizeQuantType(name)
	switch {
	case name == "":
		return 0
	case core.Contains(name, "bf16") || core.Contains(name, "f16"):
		return 16
	case core.Contains(name, "f32"):
		return 32
	case core.Contains(name, "f64"):
		return 64
	case core.Contains(name, "nvfp4") || core.Contains(name, "mxfp4") || core.Contains(name, "iq4") || core.Contains(name, "q4"):
		return 4
	case core.Contains(name, "iq5") || core.Contains(name, "q5"):
		return 5
	case core.Contains(name, "iq8") || core.Contains(name, "q8"):
		return 8
	case core.Contains(name, "iq6") || core.Contains(name, "q6"):
		return 6
	case core.Contains(name, "iq3") || core.Contains(name, "q3"):
		return 3
	case core.Contains(name, "iq2") || core.Contains(name, "q2"):
		return 2
	case core.Contains(name, "iq1") || core.Contains(name, "tq1"):
		return 1
	default:
		return 0
	}
}

func quantFamilyForType(name string) string {
	name = NormalizeQuantType(name)
	switch {
	case name == "":
		return ""
	case core.HasPrefix(name, "iq"):
		return "iq"
	case core.HasPrefix(name, "mxfp"):
		return "mxfp"
	case core.HasPrefix(name, "nvfp"):
		return "nvfp"
	case core.Contains(name, "_k"):
		return "qk"
	case core.HasPrefix(name, "q8"):
		return "q8"
	case core.HasPrefix(name, "q5"):
		return "q5"
	case core.HasPrefix(name, "q4"):
		return "q4"
	case core.HasPrefix(name, "q3"):
		return "q3"
	case core.HasPrefix(name, "q2"):
		return "q2"
	case core.HasPrefix(name, "tq"):
		return "tq"
	case name == "f16" || name == "f32" || name == "bf16" || name == "f64":
		return "dense"
	default:
		return ""
	}
}

func ggufQuantizationIsMixed(quantType string, summaries []TensorTypeSummary) bool {
	quantType = NormalizeQuantType(quantType)
	if core.HasSuffix(quantType, "_m") || core.Contains(quantType, "some_f16") {
		return true
	}
	seen := map[string]bool{}
	for _, summary := range summaries {
		if summary.Quantized && summary.Name != "" {
			seen[summary.Name] = true
		}
	}
	return len(seen) > 1
}

func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
