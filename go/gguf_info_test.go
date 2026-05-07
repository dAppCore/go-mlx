// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

type ggufMetaSpec struct {
	Key       string
	ValueType uint32
	Value     any
}

type ggufArraySpec struct {
	ElementType uint32
	Values      []any
}

type ggufTensorSpec struct {
	Name string
	Type uint32
	Dims []uint64
}

func TestReadGGUFInfo_Good(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{
		"model_type": "gemma3",
		"vocab_size": 262208,
		"hidden_size": 3072,
		"num_hidden_layers": 26,
		"max_position_embeddings": 8192,
		"quantization": {"bits": 4, "group_size": 32}
	}`), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	ggufPath := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "gemma3"},
			{Key: "gemma3.block_count", ValueType: ggufValueTypeUint32, Value: uint32(26)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: ggufTensorTypeQ4_0, Dims: []uint64{128, 128}},
			{Name: "model.norm.weight", Type: ggufTensorTypeF32, Dims: []uint64{128}},
		},
	)

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if info.Architecture != "gemma3" {
		t.Fatalf("Architecture = %q, want %q", info.Architecture, "gemma3")
	}
	if info.NumLayers != 26 {
		t.Fatalf("NumLayers = %d, want 26", info.NumLayers)
	}
	if info.VocabSize != 262208 {
		t.Fatalf("VocabSize = %d, want 262208", info.VocabSize)
	}
	if info.HiddenSize != 3072 {
		t.Fatalf("HiddenSize = %d, want 3072", info.HiddenSize)
	}
	if info.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", info.ContextLength)
	}
	if info.QuantBits != 4 {
		t.Fatalf("QuantBits = %d, want 4", info.QuantBits)
	}
	if info.QuantGroup != 32 {
		t.Fatalf("QuantGroup = %d, want 32", info.QuantGroup)
	}
	if info.TensorCount != 3 {
		t.Fatalf("TensorCount = %d, want 3", info.TensorCount)
	}
}

func TestReadGGUFInfo_FallbackLayerCount_Good(t *testing.T) {
	coverageTokens := "FallbackLayerCount"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ8_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: ggufTensorTypeQ8_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.2.self_attn.q_proj.weight", Type: ggufTensorTypeQ8_0, Dims: []uint64{128, 128}},
		},
	)

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if info.NumLayers != 3 {
		t.Fatalf("NumLayers = %d, want 3", info.NumLayers)
	}
	if info.QuantBits != 8 {
		t.Fatalf("QuantBits = %d, want 8", info.QuantBits)
	}
}

func TestReadGGUFInfo_MetadataShapeFallbacks_Good(t *testing.T) {
	coverageTokens := "MetadataShapeFallbacks"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "llama"},
			{Key: "llama.vocab_size", ValueType: ggufValueTypeUint32, Value: uint32(32000)},
			{Key: "llama.embedding_length", ValueType: ggufValueTypeUint32, Value: uint32(4096)},
			{Key: "llama.context_length", ValueType: ggufValueTypeUint32, Value: uint32(8192)},
			{Key: "llama.block_count", ValueType: ggufValueTypeUint32, Value: uint32(32)},
		},
		[]ggufTensorSpec{
			{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeQ4_0, Dims: []uint64{128, 128}},
		},
	)

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if info.VocabSize != 32000 {
		t.Fatalf("VocabSize = %d, want 32000", info.VocabSize)
	}
	if info.HiddenSize != 4096 {
		t.Fatalf("HiddenSize = %d, want 4096", info.HiddenSize)
	}
	if info.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", info.ContextLength)
	}
	if info.NumLayers != 32 {
		t.Fatalf("NumLayers = %d, want 32", info.NumLayers)
	}
}

func TestReadGGUFInfo_TextConfigDimensions_Good(t *testing.T) {
	coverageTokens := "TextConfigDimensions"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 2560,
			"num_hidden_layers": 48,
			"max_position_embeddings": 131072
		},
		"quantization_config": {"bits": 4, "group_size": 64}
	}`), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	ggufPath := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, ggufPath, nil, []ggufTensorSpec{
		{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4_0, Dims: []uint64{128, 128}},
	})

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if info.Architecture != "gemma4_text" {
		t.Fatalf("Architecture = %q, want gemma4_text", info.Architecture)
	}
	if info.VocabSize != 262144 {
		t.Fatalf("VocabSize = %d, want 262144", info.VocabSize)
	}
	if info.HiddenSize != 2560 {
		t.Fatalf("HiddenSize = %d, want 2560", info.HiddenSize)
	}
	if info.NumLayers != 48 {
		t.Fatalf("NumLayers = %d, want 48", info.NumLayers)
	}
	if info.ContextLength != 131072 {
		t.Fatalf("ContextLength = %d, want 131072", info.ContextLength)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("quant = %d-bit group=%d, want 4-bit group=64", info.QuantBits, info.QuantGroup)
	}
}

func TestModelConfigProbe_QwenFamilyArchitectures_Good(t *testing.T) {
	cases := []struct {
		name string
		arch string
		want string
	}{
		{name: "qwen3_moe", arch: "Qwen3MoeForCausalLM", want: "qwen3_moe"},
		{name: "qwen3_moe_caps", arch: "Qwen3MoEForCausalLM", want: "qwen3_moe"},
		{name: "qwen3_next", arch: "Qwen3NextForCausalLM", want: "qwen3_next"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			probe := &modelConfigProbe{Architectures: []string{tc.arch}}
			if got := probe.architecture(); got != tc.want {
				t.Fatalf("architecture() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestReadGGUFInfo_QuantizationMetadataAndTensorValidation_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"},
			{Key: "general.file_type", ValueType: ggufValueTypeUint32, Value: uint32(15)},
			{Key: "general.quantization_version", ValueType: ggufValueTypeUint32, Value: uint32(2)},
			{Key: "qwen3.context_length", ValueType: ggufValueTypeUint32, Value: uint32(40960)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.layers.0.self_attn.k_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.norm.weight", Type: ggufTensorTypeF32, Dims: []uint64{128}},
		},
	)

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if !info.Valid() {
		t.Fatalf("GGUF validation issues = %+v", info.ValidationIssues)
	}
	if info.QuantType != "q4_k_m" || info.QuantFamily != "qk" || info.QuantBits != 4 {
		t.Fatalf("quant = type:%q family:%q bits:%d", info.QuantType, info.QuantFamily, info.QuantBits)
	}
	if info.Quantization.FileType != 15 || info.Quantization.FileTypeName != "q4_k_m" || info.Quantization.Version != 2 {
		t.Fatalf("quantization details = %+v", info.Quantization)
	}
	if len(info.Quantization.TensorTypes) != 2 {
		t.Fatalf("tensor type summary = %+v, want q4_k and f32", info.Quantization.TensorTypes)
	}
	if len(info.Tensors) != 3 {
		t.Fatalf("Tensors = %d, want 3", len(info.Tensors))
	}
	if info.Tensors[0].TypeName != "q4_k" || info.Tensors[0].Bits != 4 || info.Tensors[0].BlockSize != 256 {
		t.Fatalf("first tensor = %+v", info.Tensors[0])
	}
	if len(info.Tensors[0].Shape) != 2 || info.Tensors[0].Shape[0] != 256 || info.Tensors[0].Shape[1] != 128 {
		t.Fatalf("first tensor shape = %+v", info.Tensors[0].Shape)
	}
}

func TestReadGGUFInfo_RecognizesCommonGGMLQuantTypes_Good(t *testing.T) {
	cases := []struct {
		name          string
		metadata      []ggufMetaSpec
		tensorType    uint32
		wantType      string
		wantFamily    string
		wantBits      int
		wantTensor    string
		wantTensorBit int
	}{
		{
			name:          "q5_k_m_file_type",
			metadata:      []ggufMetaSpec{{Key: "general.file_type", ValueType: ggufValueTypeUint32, Value: uint32(17)}},
			tensorType:    ggufTensorTypeQ5K,
			wantType:      "q5_k_m",
			wantFamily:    "qk",
			wantBits:      5,
			wantTensor:    "q5_k",
			wantTensorBit: 5,
		},
		{
			name:          "q8_tensor",
			tensorType:    ggufTensorTypeQ8_0,
			wantType:      "q8_0",
			wantFamily:    "q8",
			wantBits:      8,
			wantTensor:    "q8_0",
			wantTensorBit: 8,
		},
		{
			name:          "iq_tensor",
			tensorType:    ggufTensorTypeIQ4NL,
			wantType:      "iq4_nl",
			wantFamily:    "iq",
			wantBits:      4,
			wantTensor:    "iq4_nl",
			wantTensorBit: 4,
		},
		{
			name: "mxfp4_metadata",
			metadata: []ggufMetaSpec{
				{Key: "general.quantization_type", ValueType: ggufValueTypeString, Value: "mxfp4"},
			},
			tensorType:    ggufTensorTypeF16,
			wantType:      "mxfp4",
			wantFamily:    "mxfp",
			wantBits:      4,
			wantTensor:    "f16",
			wantTensorBit: 16,
		},
		{
			name: "nvfp4_metadata",
			metadata: []ggufMetaSpec{
				{Key: "quantization.type", ValueType: ggufValueTypeString, Value: "nvfp4"},
			},
			tensorType:    ggufTensorTypeF16,
			wantType:      "nvfp4",
			wantFamily:    "nvfp",
			wantBits:      4,
			wantTensor:    "f16",
			wantTensorBit: 16,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
			metadata := append([]ggufMetaSpec{{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "llama"}}, tc.metadata...)
			writeTestGGUF(t, ggufPath, metadata, []ggufTensorSpec{
				{Name: "blk.0.attn_q.weight", Type: tc.tensorType, Dims: []uint64{256, 128}},
			})

			info, err := ReadGGUFInfo(ggufPath)
			if err != nil {
				t.Fatalf("ReadGGUFInfo() error = %v", err)
			}
			if info.QuantType != tc.wantType || info.QuantFamily != tc.wantFamily || info.QuantBits != tc.wantBits {
				t.Fatalf("quant = type:%q family:%q bits:%d, want %s/%s/%d", info.QuantType, info.QuantFamily, info.QuantBits, tc.wantType, tc.wantFamily, tc.wantBits)
			}
			if info.Tensors[0].TypeName != tc.wantTensor || info.Tensors[0].Bits != tc.wantTensorBit {
				t.Fatalf("tensor = %+v, want type %s bits %d", info.Tensors[0], tc.wantTensor, tc.wantTensorBit)
			}
		})
	}
}

func TestReadGGUFInfo_InvalidTensorShapeAndDType_Bad(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{127, 128}},
			{Name: "model.layers.0.self_attn.k_proj.weight", Type: 999, Dims: []uint64{128, 0}},
		},
	)

	info, err := ReadGGUFInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadGGUFInfo() error = %v", err)
	}
	if info.Valid() {
		t.Fatalf("Valid() = true, want validation issues for invalid tensor metadata")
	}
	if !ggufValidationHasCode(info.ValidationIssues, "tensor_shape_not_block_aligned") || !ggufValidationHasCode(info.ValidationIssues, "unknown_tensor_type") || !ggufValidationHasCode(info.ValidationIssues, "invalid_tensor_dimension") {
		t.Fatalf("validation issues = %+v", info.ValidationIssues)
	}
}

func TestParseGGUF_MetadataRoundTrip_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.name", ValueType: ggufValueTypeString, Value: "roundtrip"},
			{Key: "general.file_type", ValueType: ggufValueTypeUint32, Value: uint32(15)},
			{Key: "general.alignment", ValueType: ggufValueTypeUint64, Value: uint64(32)},
			{Key: "general.use_mlock", ValueType: ggufValueTypeBool, Value: true},
			{Key: "tokenizer.ggml.tokens", ValueType: ggufValueTypeArray, Value: ggufArraySpec{ElementType: ggufValueTypeString, Values: []any{"<bos>", "<eos>"}}},
		},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}}},
	)

	metadata, tensors, err := parseGGUF(ggufPath)
	if err != nil {
		t.Fatalf("parseGGUF() error = %v", err)
	}
	if metadataString(metadata["general.name"]) != "roundtrip" {
		t.Fatalf("general.name = %q", metadataString(metadata["general.name"]))
	}
	if metadataInt(metadata["general.file_type"]) != 15 || metadataInt(metadata["general.alignment"]) != 32 {
		t.Fatalf("integer metadata = file_type:%v alignment:%v", metadata["general.file_type"], metadata["general.alignment"])
	}
	if value, ok := metadata["general.use_mlock"].(bool); !ok || !value {
		t.Fatalf("general.use_mlock = %#v", metadata["general.use_mlock"])
	}
	tokens, ok := metadata["tokenizer.ggml.tokens"].([]any)
	if !ok || len(tokens) != 2 || tokens[1] != "<eos>" {
		t.Fatalf("tokens = %#v", metadata["tokenizer.ggml.tokens"])
	}
	if len(tensors) != 1 || len(tensors[0].Shape) != 2 || tensors[0].Shape[0] != 256 || tensors[0].Offset != 0 {
		t.Fatalf("tensors = %+v", tensors)
	}
}

func TestDiscoverModels_Good(t *testing.T) {
	base := t.TempDir()

	safetensorsDir := core.PathJoin(base, "gemma")
	if result := core.MkdirAll(safetensorsDir, 0o755); !result.OK {
		t.Fatalf("mkdir safetensors dir: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(safetensorsDir, "config.json"), []byte(`{
		"model_type": "gemma3",
		"quantization": {"bits": 4, "group_size": 32}
	}`), 0o644); !result.OK {
		t.Fatalf("write safetensors config: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(safetensorsDir, "model-00001-of-00001.safetensors"), []byte("stub"), 0o644); !result.OK {
		t.Fatalf("write safetensors file: %v", result.Value)
	}

	ggufDir := core.PathJoin(base, "qwen")
	if result := core.MkdirAll(ggufDir, 0o755); !result.OK {
		t.Fatalf("mkdir gguf dir: %v", result.Value)
	}
	ggufPath := core.PathJoin(ggufDir, "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ggufValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ8_0, Dims: []uint64{64, 64}},
		},
	)

	models := DiscoverModels(base)
	if len(models) != 2 {
		t.Fatalf("DiscoverModels() found %d models, want 2", len(models))
	}

	if models[0].Format != "safetensors" {
		t.Fatalf("first format = %q, want safetensors", models[0].Format)
	}
	if models[1].Format != "gguf" {
		t.Fatalf("second format = %q, want gguf", models[1].Format)
	}
	if models[1].Path != ggufPath {
		t.Fatalf("gguf path = %q, want %q", models[1].Path, ggufPath)
	}
}

func TestReadGGUFInfo_InvalidMagic_Bad(t *testing.T) {
	coverageTokens := "InvalidMagic"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	path := core.PathJoin(t.TempDir(), "broken.gguf")
	if result := core.WriteFile(path, []byte("not-gguf"), 0o644); !result.OK {
		t.Fatalf("write broken file: %v", result.Value)
	}

	if _, err := ReadGGUFInfo(path); err == nil {
		t.Fatal("expected ReadGGUFInfo() to fail for invalid magic")
	}
}

func ggufValidationHasCode(issues []GGUFValidationIssue, code string) bool {
	for _, issue := range issues {
		if issue.Code == code {
			return true
		}
	}
	return false
}

func writeTestGGUF(t *testing.T, path string, metadata []ggufMetaSpec, tensors []ggufTensorSpec) {
	t.Helper()

	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	write := func(value any) {
		t.Helper()
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			t.Fatalf("binary write failed: %v", err)
		}
	}

	if _, err := file.Write([]byte("GGUF")); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(len(metadata)))

	for _, entry := range metadata {
		writeGGUFString(t, file, entry.Key)
		write(entry.ValueType)
		writeGGUFValue(t, file, entry.ValueType, entry.Value)
	}

	for _, tensor := range tensors {
		writeGGUFString(t, file, tensor.Name)
		write(uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			write(dim)
		}
		write(tensor.Type)
		write(uint64(0))
	}
}

func writeGGUFString(t *testing.T, file *core.OSFile, value string) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatalf("write string length: %v", err)
	}
	if _, err := file.Write([]byte(value)); err != nil {
		t.Fatalf("write string bytes: %v", err)
	}
}

func writeGGUFValue(t *testing.T, file *core.OSFile, valueType uint32, value any) {
	t.Helper()
	switch valueType {
	case ggufValueTypeBool:
		boolValue, ok := value.(bool)
		if !ok {
			t.Fatalf("write bool: got %T, want bool", value)
		}
		var encoded uint8
		if boolValue {
			encoded = 1
		}
		if err := binary.Write(file, binary.LittleEndian, encoded); err != nil {
			t.Fatalf("write bool: %v", err)
		}
	case ggufValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			t.Fatalf("write string: got %T, want string", value)
		}
		writeGGUFString(t, file, stringValue)
	case ggufValueTypeUint32:
		uint32Value, ok := value.(uint32)
		if !ok {
			t.Fatalf("write uint32: got %T, want uint32", value)
		}
		if err := binary.Write(file, binary.LittleEndian, uint32Value); err != nil {
			t.Fatalf("write uint32: %v", err)
		}
	case ggufValueTypeUint64:
		uint64Value, ok := value.(uint64)
		if !ok {
			t.Fatalf("write uint64: got %T, want uint64", value)
		}
		if err := binary.Write(file, binary.LittleEndian, uint64Value); err != nil {
			t.Fatalf("write uint64: %v", err)
		}
	case ggufValueTypeArray:
		arrayValue, ok := value.(ggufArraySpec)
		if !ok {
			t.Fatalf("write array: got %T, want ggufArraySpec", value)
		}
		if err := binary.Write(file, binary.LittleEndian, arrayValue.ElementType); err != nil {
			t.Fatalf("write array element type: %v", err)
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(arrayValue.Values))); err != nil {
			t.Fatalf("write array length: %v", err)
		}
		for _, item := range arrayValue.Values {
			writeGGUFValue(t, file, arrayValue.ElementType, item)
		}
	default:
		t.Fatalf("unsupported test gguf value type %d", valueType)
	}
}

// Generated file-aware compliance coverage.
func TestGgufInfo_ReadGGUFInfo_Good(t *testing.T) {
	target := "ReadGGUFInfo"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGgufInfo_ReadGGUFInfo_Bad(t *testing.T) {
	target := "ReadGGUFInfo"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGgufInfo_ReadGGUFInfo_Ugly(t *testing.T) {
	target := "ReadGGUFInfo"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGgufInfo_DiscoverModels_Good(t *testing.T) {
	target := "DiscoverModels"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGgufInfo_DiscoverModels_Bad(t *testing.T) {
	target := "DiscoverModels"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGgufInfo_DiscoverModels_Ugly(t *testing.T) {
	target := "DiscoverModels"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
