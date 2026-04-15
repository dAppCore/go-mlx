// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	ggufValueTypeUint8   = 0
	ggufValueTypeInt8    = 1
	ggufValueTypeUint16  = 2
	ggufValueTypeInt16   = 3
	ggufValueTypeUint32  = 4
	ggufValueTypeInt32   = 5
	ggufValueTypeFloat32 = 6
	ggufValueTypeBool    = 7
	ggufValueTypeString  = 8
	ggufValueTypeArray   = 9
	ggufValueTypeUint64  = 10
	ggufValueTypeInt64   = 11
	ggufValueTypeFloat64 = 12
)

const (
	ggufTensorTypeF32    = 0
	ggufTensorTypeF16    = 1
	ggufTensorTypeQ4_0   = 2
	ggufTensorTypeQ4_1   = 3
	ggufTensorTypeQ5_0   = 6
	ggufTensorTypeQ5_1   = 7
	ggufTensorTypeQ8_0   = 8
	ggufTensorTypeQ8_1   = 9
	ggufTensorTypeQ2K    = 10
	ggufTensorTypeQ3K    = 11
	ggufTensorTypeQ4K    = 12
	ggufTensorTypeQ5K    = 13
	ggufTensorTypeQ6K    = 14
	ggufTensorTypeQ8K    = 15
	ggufTensorTypeIQ2XXS = 16
	ggufTensorTypeIQ2XS  = 17
	ggufTensorTypeIQ3XXS = 18
	ggufTensorTypeIQ1S   = 19
	ggufTensorTypeIQ4NL  = 20
	ggufTensorTypeIQ3S   = 21
	ggufTensorTypeIQ2S   = 22
	ggufTensorTypeIQ4XS  = 23
	ggufTensorTypeI8     = 24
	ggufTensorTypeI16    = 25
	ggufTensorTypeI32    = 26
	ggufTensorTypeI64    = 27
	ggufTensorTypeF64    = 28
	ggufTensorTypeIQ1M   = 29
	ggufTensorTypeBF16   = 30
)

// GGUFInfo summarises the metadata of a GGUF checkpoint.
type GGUFInfo struct {
	Path          string
	Architecture  string
	VocabSize     int
	HiddenSize    int
	NumLayers     int
	ContextLength int
	QuantBits     int
	QuantGroup    int
	TensorCount   int
	MetadataCount int
}

// DiscoveredModel is a loadable model discovered on disk.
type DiscoveredModel struct {
	Path       string
	ModelType  string
	QuantBits  int
	QuantGroup int
	NumFiles   int
	Format     string
}

type ggufTensorInfo struct {
	Name string
	Type uint32
}

type modelConfigProbe struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
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

// ReadGGUFInfo reads GGUF metadata without loading model weights into MLX.
func ReadGGUFInfo(modelPath string) (GGUFInfo, error) {
	ggufPath, err := resolveGGUFFile(modelPath)
	if err != nil {
		return GGUFInfo{}, err
	}

	metadata, tensors, err := parseGGUF(ggufPath)
	if err != nil {
		return GGUFInfo{}, err
	}

	absolutePath := ggufPath
	if abs, err := filepath.Abs(ggufPath); err == nil {
		absolutePath = abs
	}

	config, _ := readModelConfig(filepath.Dir(ggufPath))
	architecture := firstNonEmpty(
		metadataString(metadata["general.architecture"]),
		config.architecture(),
	)
	quantBits := config.quantBits()
	if quantBits == 0 {
		quantBits = inferQuantBits(tensors)
	}

	info := GGUFInfo{
		Path:          absolutePath,
		Architecture:  architecture,
		VocabSize:     config.vocabSize(),
		HiddenSize:    config.hiddenSize(),
		NumLayers:     config.numLayers(),
		ContextLength: config.contextLength(),
		QuantBits:     quantBits,
		QuantGroup:    config.quantGroup(),
		TensorCount:   len(tensors),
		MetadataCount: len(metadata),
	}
	if info.NumLayers == 0 {
		info.NumLayers = inferLayerCount(metadata, tensors, info.Architecture)
	}

	return info, nil
}

// DiscoverModels returns loadable safetensors and GGUF models beneath basePath.
func DiscoverModels(basePath string) []DiscoveredModel {
	resolvedPath := basePath
	if abs, err := filepath.Abs(basePath); err == nil {
		resolvedPath = abs
	}

	if info, err := os.Stat(resolvedPath); err == nil && !info.IsDir() {
		if strings.HasSuffix(strings.ToLower(resolvedPath), ".gguf") {
			ggufInfo, err := ReadGGUFInfo(resolvedPath)
			if err == nil {
				return []DiscoveredModel{{
					Path:       ggufInfo.Path,
					ModelType:  ggufInfo.Architecture,
					QuantBits:  ggufInfo.QuantBits,
					QuantGroup: ggufInfo.QuantGroup,
					NumFiles:   1,
					Format:     "gguf",
				}}
			}
		}
		return nil
	}

	var models []DiscoveredModel
	_ = filepath.WalkDir(resolvedPath, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || !d.IsDir() {
			return nil
		}
		if model, ok := probeDiscoveredModel(path); ok {
			models = append(models, model)
		}
		return nil
	})

	sort.Slice(models, func(i, j int) bool {
		return models[i].Path < models[j].Path
	})
	return models
}

func probeDiscoveredModel(dir string) (DiscoveredModel, bool) {
	config, configErr := readModelConfig(dir)

	safetensors, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
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

	ggufs, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if len(ggufs) != 1 {
		return DiscoveredModel{}, false
	}

	info, err := ReadGGUFInfo(ggufs[0])
	if err != nil {
		return DiscoveredModel{}, false
	}
	modelType := info.Architecture
	if modelType == "" && configErr == nil {
		modelType = config.architecture()
	}
	return DiscoveredModel{
		Path:       info.Path,
		ModelType:  modelType,
		QuantBits:  info.QuantBits,
		QuantGroup: info.QuantGroup,
		NumFiles:   1,
		Format:     "gguf",
	}, true
}

func resolveGGUFFile(modelPath string) (string, error) {
	if strings.HasSuffix(strings.ToLower(modelPath), ".gguf") {
		return modelPath, nil
	}

	ggufs, err := filepath.Glob(filepath.Join(modelPath, "*.gguf"))
	if err != nil {
		return "", fmt.Errorf("mlx: scan gguf files: %w", err)
	}
	switch len(ggufs) {
	case 0:
		return "", errors.New("mlx: no .gguf file found")
	case 1:
		return ggufs[0], nil
	default:
		return "", errors.New("mlx: multiple .gguf files found")
	}
}

func parseGGUF(path string) (map[string]any, []ggufTensorInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("mlx: open gguf: %w", err)
	}
	defer file.Close()

	var magic [4]byte
	if _, err := io.ReadFull(file, magic[:]); err != nil {
		return nil, nil, fmt.Errorf("mlx: read gguf magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, nil, errors.New("mlx: invalid gguf magic")
	}

	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, nil, fmt.Errorf("mlx: read gguf version: %w", err)
	}
	if version < 2 {
		return nil, nil, fmt.Errorf("mlx: unsupported gguf version %d", version)
	}

	var tensorCount uint64
	if err := binary.Read(file, binary.LittleEndian, &tensorCount); err != nil {
		return nil, nil, fmt.Errorf("mlx: read gguf tensor count: %w", err)
	}
	var metadataCount uint64
	if err := binary.Read(file, binary.LittleEndian, &metadataCount); err != nil {
		return nil, nil, fmt.Errorf("mlx: read gguf metadata count: %w", err)
	}

	metadata := make(map[string]any, metadataCount)
	for range metadataCount {
		key, err := readGGUFString(file)
		if err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf metadata key: %w", err)
		}
		var valueType uint32
		if err := binary.Read(file, binary.LittleEndian, &valueType); err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf metadata type: %w", err)
		}
		value, err := readGGUFValue(file, valueType)
		if err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf metadata value for %q: %w", key, err)
		}
		metadata[key] = value
	}

	tensors := make([]ggufTensorInfo, 0, tensorCount)
	for range tensorCount {
		name, err := readGGUFString(file)
		if err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf tensor name: %w", err)
		}
		var ndim uint32
		if err := binary.Read(file, binary.LittleEndian, &ndim); err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf tensor ndim: %w", err)
		}
		for range ndim {
			var dim uint64
			if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
				return nil, nil, fmt.Errorf("mlx: read gguf tensor dimension: %w", err)
			}
		}
		var tensorType uint32
		if err := binary.Read(file, binary.LittleEndian, &tensorType); err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf tensor type: %w", err)
		}
		var offset uint64
		if err := binary.Read(file, binary.LittleEndian, &offset); err != nil {
			return nil, nil, fmt.Errorf("mlx: read gguf tensor offset: %w", err)
		}
		tensors = append(tensors, ggufTensorInfo{Name: name, Type: tensorType})
	}

	return metadata, tensors, nil
}

func readGGUFString(reader io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 16<<20 {
		return "", errors.New("gguf string is unreasonably large")
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
	case ggufValueTypeUint32:
		return readGGUFBinary[uint32](reader)
	case ggufValueTypeInt32:
		return readGGUFBinary[int32](reader)
	case ggufValueTypeFloat32:
		return readGGUFBinary[float32](reader)
	case ggufValueTypeBool:
		value, err := readGGUFBinary[uint8](reader)
		return value != 0, err
	case ggufValueTypeString:
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
		values := make([]any, 0, length)
		for range length {
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
		return nil, fmt.Errorf("unsupported gguf metadata type %d", valueType)
	}
}

func readGGUFBinary[T any](reader io.Reader) (T, error) {
	var value T
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readModelConfig(dir string) (*modelConfigProbe, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var config modelConfigProbe
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return &config, nil
}

func (probe *modelConfigProbe) architecture() string {
	if probe == nil {
		return ""
	}
	if probe.ModelType != "" {
		return probe.ModelType
	}
	if probe.TextConfig.ModelType != "" {
		return probe.TextConfig.ModelType
	}
	for _, architecture := range probe.Architectures {
		switch {
		case strings.Contains(architecture, "Gemma4"):
			return "gemma4_text"
		case strings.Contains(architecture, "Gemma3"):
			return "gemma3"
		case strings.Contains(architecture, "Gemma2"):
			return "gemma2"
		case strings.Contains(architecture, "Qwen3"):
			return "qwen3"
		case strings.Contains(architecture, "Qwen2"):
			return "qwen2"
		case strings.Contains(architecture, "Llama"):
			return "llama"
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
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
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
		index := strings.Index(name, marker)
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
	switch tensorType {
	case ggufTensorTypeQ4_0, ggufTensorTypeQ4_1, ggufTensorTypeQ4K, ggufTensorTypeIQ4NL, ggufTensorTypeIQ4XS:
		return 4
	case ggufTensorTypeQ5_0, ggufTensorTypeQ5_1, ggufTensorTypeQ5K:
		return 5
	case ggufTensorTypeQ6K:
		return 6
	case ggufTensorTypeQ8_0, ggufTensorTypeQ8_1, ggufTensorTypeQ8K:
		return 8
	case ggufTensorTypeQ2K, ggufTensorTypeIQ2XXS, ggufTensorTypeIQ2XS, ggufTensorTypeIQ2S:
		return 2
	case ggufTensorTypeQ3K, ggufTensorTypeIQ3XXS, ggufTensorTypeIQ3S:
		return 3
	case ggufTensorTypeIQ1S, ggufTensorTypeIQ1M:
		return 1
	default:
		return 0
	}
}
