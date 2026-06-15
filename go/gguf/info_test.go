// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// ggufTestScratchSize mirrors the 64-byte scratch buffer parseGGUF stack-
// allocates (info.go: `var scratch [64]byte`) — large enough to decode any
// fixed-width value and short interned strings in one read.
const ggufTestScratchSize = 64

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

func TestInfo_ReadInfo_Good(t *testing.T) {
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
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "gemma3"},
			{Key: "gemma3.block_count", ValueType: ValueTypeUint32, Value: uint32(26)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ4_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: TensorTypeQ4_0, Dims: []uint64{128, 128}},
			{Name: "model.norm.weight", Type: ggufTensorTypeF32, Dims: []uint64{128}},
		},
	)

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
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

func TestInfo_ReadInfo_FallbackLayerCount_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ8_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.1.self_attn.q_proj.weight", Type: TensorTypeQ8_0, Dims: []uint64{128, 128}},
			{Name: "model.layers.2.self_attn.q_proj.weight", Type: TensorTypeQ8_0, Dims: []uint64{128, 128}},
		},
	)

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
	}
	if info.NumLayers != 3 {
		t.Fatalf("NumLayers = %d, want 3", info.NumLayers)
	}
	if info.QuantBits != 8 {
		t.Fatalf("QuantBits = %d, want 8", info.QuantBits)
	}
}

func TestInfo_ReadInfo_MetadataShapeFallbacks_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
			{Key: "llama.vocab_size", ValueType: ValueTypeUint32, Value: uint32(32000)},
			{Key: "llama.embedding_length", ValueType: ValueTypeUint32, Value: uint32(4096)},
			{Key: "llama.context_length", ValueType: ValueTypeUint32, Value: uint32(8192)},
			{Key: "llama.block_count", ValueType: ValueTypeUint32, Value: uint32(32)},
		},
		[]ggufTensorSpec{
			{Name: "blk.0.attn_q.weight", Type: TensorTypeQ4_0, Dims: []uint64{128, 128}},
		},
	)

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
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

func TestInfo_ReadInfo_TextConfigDimensions_Good(t *testing.T) {
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
		{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ4_0, Dims: []uint64{128, 128}},
	})

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
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

func TestInfo_architecture_QwenFamilyArchitectures_Good(t *testing.T) {
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

func TestGGUFMetadataHelpers_Ugly(t *testing.T) {
	intCases := []struct {
		value any
		want  int
	}{
		{value: uint8(1), want: 1},
		{value: int8(-2), want: -2},
		{value: uint16(3), want: 3},
		{value: int16(-4), want: -4},
		{value: uint32(5), want: 5},
		{value: int32(-6), want: -6},
		{value: uint64(7), want: 7},
		{value: int64(-8), want: -8},
		{value: float32(9.9), want: 9},
		{value: float64(-10.2), want: -10},
		{value: "11", want: 0},
	}
	for _, tc := range intCases {
		if got := metadataInt(tc.value); got != tc.want {
			t.Fatalf("metadataInt(%T(%v)) = %d, want %d", tc.value, tc.value, got, tc.want)
		}
	}

	if got := metadataString("q4_k_m"); got != "q4_k_m" {
		t.Fatalf("metadataString(string) = %q", got)
	}
	if got := metadataString(4); got != "" {
		t.Fatalf("metadataString(int) = %q, want blank", got)
	}
	if got := metadataArrayLen([]string{"a", "b"}); got != 2 {
		t.Fatalf("metadataArrayLen([]string) = %d, want 2", got)
	}
	if got := metadataArrayLen([]any{"a", "b", "c"}); got != 3 {
		t.Fatalf("metadataArrayLen([]any) = %d, want 3", got)
	}
	if got := metadataArrayLen(ggufStringArrayLen(5)); got != 5 {
		t.Fatalf("metadataArrayLen(ggufStringArrayLen) = %d, want 5", got)
	}
	if got := metadataArrayLen("nope"); got != 0 {
		t.Fatalf("metadataArrayLen(string) = %d, want 0", got)
	}
}

func boolQuantBits(quantized bool, bits int) int {
	if quantized {
		return bits
	}
	return 0
}

func TestInfo_ReadInfo_QuantizationMetadataAndTensorValidation_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
			{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(15)},
			{Key: "general.quantization_version", ValueType: ValueTypeUint32, Value: uint32(2)},
			{Key: "qwen3.context_length", ValueType: ValueTypeUint32, Value: uint32(40960)},
		},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.layers.0.self_attn.k_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 128}},
			{Name: "model.norm.weight", Type: ggufTensorTypeF32, Dims: []uint64{128}},
		},
	)

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
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

func TestInfo_ReadInfo_RecognizesCommonGGMLQuantTypes_Good(t *testing.T) {
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
			metadata:      []ggufMetaSpec{{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(17)}},
			tensorType:    ggufTensorTypeQ5K,
			wantType:      "q5_k_m",
			wantFamily:    "qk",
			wantBits:      5,
			wantTensor:    "q5_k",
			wantTensorBit: 5,
		},
		{
			name:          "q8_tensor",
			tensorType:    TensorTypeQ8_0,
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
				{Key: "general.quantization_type", ValueType: ValueTypeString, Value: "mxfp4"},
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
				{Key: "quantization.type", ValueType: ValueTypeString, Value: "nvfp4"},
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
			metadata := append([]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"}}, tc.metadata...)
			writeTestGGUF(t, ggufPath, metadata, []ggufTensorSpec{
				{Name: "blk.0.attn_q.weight", Type: tc.tensorType, Dims: []uint64{256, 128}},
			})

			info, err := ReadInfo(ggufPath)
			if err != nil {
				t.Fatalf("ReadInfo() error = %v", err)
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

func TestInfo_ReadInfo_Bad(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{127, 128}},
			{Name: "model.layers.0.self_attn.k_proj.weight", Type: 999, Dims: []uint64{128, 0}},
		},
	)

	info, err := ReadInfo(ggufPath)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
	}
	if info.Valid() {
		t.Fatalf("Valid() = true, want validation issues for invalid tensor metadata")
	}
	if !ggufValidationHasCode(info.ValidationIssues, "tensor_shape_not_block_aligned") || !ggufValidationHasCode(info.ValidationIssues, "unknown_tensor_type") || !ggufValidationHasCode(info.ValidationIssues, "invalid_tensor_dimension") {
		t.Fatalf("validation issues = %+v", info.ValidationIssues)
	}
}

func TestInfo_parseGGUF_MetadataRoundTrip_Good(t *testing.T) {
	ggufPath := core.PathJoin(t.TempDir(), "model.gguf")
	writeTestGGUF(t, ggufPath,
		[]ggufMetaSpec{
			{Key: "general.name", ValueType: ValueTypeString, Value: "roundtrip"},
			{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(15)},
			{Key: "general.alignment", ValueType: ggufValueTypeUint64, Value: uint64(32)},
			{Key: "general.use_mlock", ValueType: ggufValueTypeBool, Value: true},
			{Key: "tokenizer.ggml.tokens", ValueType: ggufValueTypeArray, Value: ggufArraySpec{ElementType: ValueTypeString, Values: []any{"<bos>", "<eos>"}}},
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
	// String-element arrays are parsed for their count only — the elements are
	// skipped (ReadInfo needs vocab size, not the token strings), so the array
	// lands as ggufStringArrayLen and metadataArrayLen reports the count.
	if tokens, ok := metadata["tokenizer.ggml.tokens"].(ggufStringArrayLen); !ok || int(tokens) != 2 {
		t.Fatalf("tokens = %#v, want ggufStringArrayLen(2)", metadata["tokenizer.ggml.tokens"])
	}
	if got := metadataArrayLen(metadata["tokenizer.ggml.tokens"]); got != 2 {
		t.Fatalf("metadataArrayLen(tokens) = %d, want 2", got)
	}
	if len(tensors) != 1 || len(tensors[0].Shape) != 2 || tensors[0].Shape[0] != 256 || tensors[0].Offset != 0 {
		t.Fatalf("tensors = %+v", tensors)
	}
}

func TestInfo_DiscoverModels_Good(t *testing.T) {
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
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{
			{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ8_0, Dims: []uint64{64, 64}},
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

func TestInfo_ReadInfo_InvalidMagic_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.gguf")
	if result := core.WriteFile(path, []byte("not-gguf"), 0o644); !result.OK {
		t.Fatalf("write broken file: %v", result.Value)
	}

	if _, err := ReadInfo(path); err == nil {
		t.Fatal("expected ReadInfo() to fail for invalid magic")
	}
}

// TestInfo_Valid_Good asserts the happy path: an Info whose validation issues
// are all warnings (or absent) is reported Valid. Valid() gates only on
// GGUFValidationError severity, so warnings must not flip it false.
func TestInfo_Valid_Good(t *testing.T) {
	noIssues := Info{Path: "/models/clean.gguf"}
	if !noIssues.Valid() {
		t.Fatalf("Valid() = false for an Info with no issues, want true")
	}

	warningsOnly := Info{
		Path: "/models/warned.gguf",
		ValidationIssues: []ValidationIssue{
			{Severity: GGUFValidationWarning, Code: "missing_alignment"},
			{Severity: GGUFValidationWarning, Code: "unusual_block_size", Tensor: "blk.0.attn_q.weight"},
		},
	}
	if !warningsOnly.Valid() {
		t.Fatalf("Valid() = false with warning-only issues %+v, want true", warningsOnly.ValidationIssues)
	}
}

// TestInfo_Valid_Bad asserts the failure path: a single error-severity issue,
// regardless of how many warnings surround it, makes Valid() false.
func TestInfo_Valid_Bad(t *testing.T) {
	oneError := Info{
		Path: "/models/broken.gguf",
		ValidationIssues: []ValidationIssue{
			{Severity: GGUFValidationError, Code: "unknown_tensor_type", Tensor: "blk.0.attn_k.weight"},
		},
	}
	if oneError.Valid() {
		t.Fatalf("Valid() = true with an error issue %+v, want false", oneError.ValidationIssues)
	}
}

// TestInfo_Valid_Ugly drives the boundary cases: an empty (non-nil) slice, and
// an error buried among leading and trailing warnings — Valid() must scan the
// whole slice, not just the head, so a late error still fails the check.
func TestInfo_Valid_Ugly(t *testing.T) {
	empty := Info{ValidationIssues: []ValidationIssue{}}
	if !empty.Valid() {
		t.Fatalf("Valid() = false for an empty issue slice, want true")
	}

	errorAtTail := Info{
		ValidationIssues: []ValidationIssue{
			{Severity: GGUFValidationWarning, Code: "w1"},
			{Severity: GGUFValidationWarning, Code: "w2"},
			{Severity: GGUFValidationError, Code: "invalid_tensor_dimension", Tensor: "blk.5.ffn_up.weight"},
		},
	}
	if errorAtTail.Valid() {
		t.Fatal("Valid() = true with a trailing error after warnings, want false")
	}

	errorAtHead := Info{
		ValidationIssues: []ValidationIssue{
			{Severity: GGUFValidationError, Code: "tensor_shape_not_block_aligned"},
			{Severity: GGUFValidationWarning, Code: "w3"},
		},
	}
	if errorAtHead.Valid() {
		t.Fatal("Valid() = true with a leading error before warnings, want false")
	}
}

// TestInfo_ReadInfo_Ugly covers ReadInfo's file-resolution boundary arms that
// the Good (valid file) and Bad (invalid magic) cases do not reach: a directory
// containing no .gguf at all, and a directory containing more than one .gguf
// (ambiguous — ReadInfo refuses to guess).
func TestInfo_ReadInfo_Ugly(t *testing.T) {
	t.Run("no_gguf_in_dir", func(t *testing.T) {
		dir := t.TempDir()
		if _, err := ReadInfo(dir); err == nil {
			t.Fatal("ReadInfo(dir with no .gguf) error = nil, want no-file error")
		}
	})

	t.Run("multiple_gguf_ambiguous", func(t *testing.T) {
		dir := t.TempDir()
		for _, name := range []string{"a.gguf", "b.gguf"} {
			writeTestGGUF(t, core.PathJoin(dir, name),
				[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}},
				[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: TensorTypeQ8_0, Dims: []uint64{64, 64}}},
			)
		}
		if _, err := ReadInfo(dir); err == nil {
			t.Fatal("ReadInfo(dir with two .gguf) error = nil, want ambiguity error")
		}
	})
}

// TestInfo_DiscoverModels_Bad asserts DiscoverModels skips non-loadable
// candidates rather than reporting them: a safetensors directory missing its
// config.json, and a directory whose lone .gguf is structurally invalid, both
// yield no discovered model.
func TestInfo_DiscoverModels_Bad(t *testing.T) {
	base := t.TempDir()

	// A *.safetensors with no config.json — probeDiscoveredModel rejects it.
	noConfig := core.PathJoin(base, "no-config")
	if result := core.MkdirAll(noConfig, 0o755); !result.OK {
		t.Fatalf("mkdir no-config: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(noConfig, "model-00001-of-00001.safetensors"), []byte("stub"), 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}

	// A directory whose only .gguf fails to parse (bad magic) — ReadInfo errors,
	// so the candidate is dropped.
	brokenGGUF := core.PathJoin(base, "broken")
	if result := core.MkdirAll(brokenGGUF, 0o755); !result.OK {
		t.Fatalf("mkdir broken: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(brokenGGUF, "model.gguf"), []byte("not-gguf-at-all"), 0o644); !result.OK {
		t.Fatalf("write broken gguf: %v", result.Value)
	}

	models := DiscoverModels(base)
	if len(models) != 0 {
		t.Fatalf("DiscoverModels() = %d models, want 0 (all candidates non-loadable): %+v", len(models), models)
	}
}

// TestInfo_DiscoverModels_Ugly drives the single-file and missing-path boundary
// arms: a direct path to one .gguf file (not a directory) is discovered as a
// one-file gguf model, while a path that does not exist returns nil.
func TestInfo_DiscoverModels_Ugly(t *testing.T) {
	t.Run("direct_gguf_file", func(t *testing.T) {
		ggufPath := core.PathJoin(t.TempDir(), "solo.gguf")
		writeTestGGUF(t, ggufPath,
			[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "gemma3"}},
			[]ggufTensorSpec{{Name: "model.layers.0.self_attn.q_proj.weight", Type: TensorTypeQ4_0, Dims: []uint64{128, 128}}},
		)
		models := DiscoverModels(ggufPath)
		if len(models) != 1 {
			t.Fatalf("DiscoverModels(file) = %d models, want 1", len(models))
		}
		if models[0].Format != "gguf" || models[0].NumFiles != 1 {
			t.Fatalf("DiscoverModels(file)[0] = %+v, want gguf/1-file", models[0])
		}
	})

	t.Run("missing_path", func(t *testing.T) {
		missing := core.PathJoin(t.TempDir(), "nowhere")
		if models := DiscoverModels(missing); models != nil {
			t.Fatalf("DiscoverModels(missing) = %+v, want nil", models)
		}
	})
}

func ggufValidationHasCode(issues []ValidationIssue, code string) bool {
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
	case ggufValueTypeUint8:
		writeGGUFScalar[uint8](t, file, value, "uint8")
	case ggufValueTypeInt8:
		writeGGUFScalar[int8](t, file, value, "int8")
	case ggufValueTypeUint16:
		writeGGUFScalar[uint16](t, file, value, "uint16")
	case ggufValueTypeInt16:
		writeGGUFScalar[int16](t, file, value, "int16")
	case ggufValueTypeInt32:
		writeGGUFScalar[int32](t, file, value, "int32")
	case ggufValueTypeFloat32:
		writeGGUFScalar[float32](t, file, value, "float32")
	case ggufValueTypeInt64:
		writeGGUFScalar[int64](t, file, value, "int64")
	case ggufValueTypeFloat64:
		writeGGUFScalar[float64](t, file, value, "float64")
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
	case ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			t.Fatalf("write string: got %T, want string", value)
		}
		writeGGUFString(t, file, stringValue)
	case ValueTypeUint32:
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

// writeGGUFScalar little-endian-encodes a fixed-width scalar metadata value,
// type-asserting it to T first. Backs the numeric arms of writeGGUFValue so a
// synthetic GGUF can carry every ggufValueType the reader recognises.
func writeGGUFScalar[T any](t *testing.T, file *core.OSFile, value any, name string) {
	t.Helper()
	typed, ok := value.(T)
	if !ok {
		t.Fatalf("write %s: got %T, want %s", name, value, name)
	}
	if err := binary.Write(file, binary.LittleEndian, typed); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

// ggufMetadataValueBytes encodes a single GGUF metadata value (no key, no type
// tag — just the value payload) into a byte slice, so readGGUFValue can be
// driven directly over a bytes.Reader without writing a whole file. Reuses the
// production-mirroring writeGGUFValue helper against an in-memory temp file.
func ggufMetadataValueBytes(t *testing.T, valueType uint32, value any) []byte {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "value.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create value file: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	writeGGUFValue(t, file, valueType, value)
	file.Close()
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("read value file: %v", read.Value)
	}
	return read.Value.([]byte)
}

// TestInfo_readGGUFValue_AllValueTypes_Good drives readGGUFValue over every
// fixed-width ggufValueType the GGUF spec defines. parseGGUF only ever supplies
// a handful in the corpus' files, so the numeric arms (uint8..float64) are
// otherwise unexercised; here each is encoded then read back and asserted
// bit-exact, with a non-nil string arena (the path parseGGUF uses).
func TestInfo_readGGUFValue_AllValueTypes_Good(t *testing.T) {
	cases := []struct {
		name      string
		valueType uint32
		value     any
		want      any
	}{
		{"uint8", ggufValueTypeUint8, uint8(0xAB), uint8(0xAB)},
		{"int8", ggufValueTypeInt8, int8(-12), int8(-12)},
		{"uint16", ggufValueTypeUint16, uint16(0xBEEF), uint16(0xBEEF)},
		{"int16", ggufValueTypeInt16, int16(-2000), int16(-2000)},
		{"uint32", ValueTypeUint32, uint32(0xDEADBEEF), uint32(0xDEADBEEF)},
		{"int32", ggufValueTypeInt32, int32(-123456), int32(-123456)},
		{"float32", ggufValueTypeFloat32, float32(3.5), float32(3.5)},
		{"bool_true", ggufValueTypeBool, true, true},
		{"bool_false", ggufValueTypeBool, false, false},
		{"string", ValueTypeString, "hello-gguf", "hello-gguf"},
		{"uint64", ggufValueTypeUint64, uint64(0x0123456789ABCDEF), uint64(0x0123456789ABCDEF)},
		{"int64", ggufValueTypeInt64, int64(-9000000000), int64(-9000000000)},
		{"float64", ggufValueTypeFloat64, float64(2.718281828), float64(2.718281828)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payload := ggufMetadataValueBytes(t, tc.valueType, tc.value)
			scratch := make([]byte, ggufTestScratchSize)
			arena := make([]byte, 0, 256)
			got, err := readGGUFValue(bytes.NewReader(payload), tc.valueType, scratch, &arena)
			if err != nil {
				t.Fatalf("readGGUFValue(%s) error = %v", tc.name, err)
			}
			if got != tc.want {
				t.Fatalf("readGGUFValue(%s) = %v (%T), want %v (%T)", tc.name, got, got, tc.want, tc.want)
			}
		})
	}
}

// TestInfo_readGGUFValue_StringArray_Good verifies the string-element array
// fast-path: the elements are skipped (never materialised) and the value comes
// back as ggufStringArrayLen carrying just the count — the shape ReadInfo
// relies on to size a vocab without allocating 200k token strings.
func TestInfo_readGGUFValue_StringArray_Good(t *testing.T) {
	payload := ggufMetadataValueBytes(t, ggufValueTypeArray, ggufArraySpec{
		ElementType: ValueTypeString,
		Values:      []any{"alpha", "beta", "gamma"},
	})
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)
	got, err := readGGUFValue(bytes.NewReader(payload), ggufValueTypeArray, scratch, &arena)
	if err != nil {
		t.Fatalf("readGGUFValue(string array) error = %v", err)
	}
	count, ok := got.(ggufStringArrayLen)
	if !ok {
		t.Fatalf("readGGUFValue(string array) = %T, want ggufStringArrayLen", got)
	}
	if int(count) != 3 {
		t.Fatalf("string array length = %d, want 3", count)
	}
}

// TestInfo_readGGUFValue_NumericArray_Good verifies a non-string array is fully
// materialised: every element is decoded (recursive readGGUFValue) and returned
// as []any in order.
func TestInfo_readGGUFValue_NumericArray_Good(t *testing.T) {
	payload := ggufMetadataValueBytes(t, ggufValueTypeArray, ggufArraySpec{
		ElementType: ValueTypeUint32,
		Values:      []any{uint32(10), uint32(20), uint32(30)},
	})
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)
	got, err := readGGUFValue(bytes.NewReader(payload), ggufValueTypeArray, scratch, &arena)
	if err != nil {
		t.Fatalf("readGGUFValue(numeric array) error = %v", err)
	}
	values, ok := got.([]any)
	if !ok || len(values) != 3 {
		t.Fatalf("readGGUFValue(numeric array) = %#v, want []any len 3", got)
	}
	for i, want := range []uint32{10, 20, 30} {
		if values[i] != want {
			t.Fatalf("array[%d] = %v, want %d", i, values[i], want)
		}
	}
}

// TestInfo_readGGUFValue_NilArena_Good covers readGGUFValue's strArena==nil branch,
// which delegates to readGGUFString rather than readStringIntoArena. parseGGUF
// always supplies a non-nil arena, so this branch (and readGGUFString itself)
// is only reachable by a direct caller that opts out of arena pooling — a
// documented part of the helper's contract.
func TestInfo_readGGUFValue_NilArena_Good(t *testing.T) {
	cases := []struct {
		name  string
		value string
	}{
		// Short, non-interned: read into scratch, intern miss, string() copy.
		{"short_uninterned", "no-arena-string"},
		// Interned key: read into scratch, intern HIT, returns the singleton.
		{"interned_singleton", "general.architecture"},
		// Longer than the 64-byte scratch: forces the make([]byte, length) arm.
		{"large_heap", string(bytes.Repeat([]byte("z"), 200))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payload := ggufMetadataValueBytes(t, ValueTypeString, tc.value)
			scratch := make([]byte, ggufTestScratchSize)
			got, err := readGGUFValue(bytes.NewReader(payload), ValueTypeString, scratch, nil)
			if err != nil {
				t.Fatalf("readGGUFValue(string, nil arena) error = %v", err)
			}
			if got != tc.value {
				t.Fatalf("readGGUFValue(string, nil arena) = %q, want %q", got, tc.value)
			}
		})
	}
}

// TestInfo_readGGUFValue_ErrorPaths_Bad covers readGGUFString's guard arms via the
// nil-arena readGGUFValue dispatch: an over-long length prefix (the 16 MiB cap)
// and a header that promises more bytes than the stream carries.
func TestInfo_readGGUFValue_ErrorPaths_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	// Length prefix of 17 MiB exceeds the 16 MiB cap -> errGGUFStringTooLong.
	var tooLong [8]byte
	binary.LittleEndian.PutUint64(tooLong[:], 17<<20)
	if _, err := readGGUFValue(bytes.NewReader(tooLong[:]), ValueTypeString, scratch, nil); err == nil {
		t.Fatal("over-long string length: error = nil, want cap error")
	}
	// Header says 10 bytes, only 3 follow -> short read.
	var shortHdr [11]byte
	binary.LittleEndian.PutUint64(shortHdr[:8], 10)
	if _, err := readGGUFValue(bytes.NewReader(shortHdr[:]), ValueTypeString, scratch, nil); err == nil {
		t.Fatal("truncated string body: error = nil, want short-read error")
	}
	// Fewer than 8 bytes -> the length-prefix read itself fails.
	if _, err := readGGUFValue(bytes.NewReader([]byte{1, 2, 3}), ValueTypeString, scratch, nil); err == nil {
		t.Fatal("truncated length prefix: error = nil, want short-read error")
	}
	// Empty string (length 0) is a valid zero-value, not an error.
	var empty [8]byte
	got, err := readGGUFValue(bytes.NewReader(empty[:]), ValueTypeString, scratch, nil)
	if err != nil || got != "" {
		t.Fatalf("empty string = %q, err = %v; want \"\", nil", got, err)
	}
}

// TestInfo_readGGUFValue_UnsupportedType_Bad covers the default arm: an unknown
// value-type tag must surface a clear error rather than silently returning a
// zero value.
func TestInfo_readGGUFValue_UnsupportedType_Bad(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)
	const unknownType = 99
	if _, err := readGGUFValue(bytes.NewReader([]byte{0, 0, 0, 0}), unknownType, scratch, &arena); err == nil {
		t.Fatal("readGGUFValue(unknown type) error = nil, want unsupported-type error")
	}
}

// TestInfo_readGGUFValue_TruncatedScalar_Ugly covers the short-read arms: a value
// header that promises a wider scalar than the stream actually carries must
// return io's unexpected-EOF rather than a partial decode.
func TestInfo_readGGUFValue_TruncatedScalar_Ugly(t *testing.T) {
	scratch := make([]byte, ggufTestScratchSize)
	arena := make([]byte, 0, 256)
	// float64 wants 8 bytes; supply 3.
	if _, err := readGGUFValue(bytes.NewReader([]byte{1, 2, 3}), ggufValueTypeFloat64, scratch, &arena); err == nil {
		t.Fatal("readGGUFValue(truncated float64) error = nil, want short-read error")
	}
}
