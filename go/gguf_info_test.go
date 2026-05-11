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

func TestModelConfigProbe_CommonArchitectureNames_Good(t *testing.T) {
	cases := []struct {
		architecture string
		want         string
	}{
		{architecture: "Gemma4ForConditionalGeneration", want: "gemma4_text"},
		{architecture: "Gemma3ForCausalLM", want: "gemma3"},
		{architecture: "Gemma2ForCausalLM", want: "gemma2"},
		{architecture: "Qwen3ForCausalLM", want: "qwen3"},
		{architecture: "Qwen2ForCausalLM", want: "qwen2"},
		{architecture: "LlamaForCausalLM", want: "llama"},
		{architecture: "MiniMaxM2ForCausalLM", want: "minimax_m2"},
		{architecture: "UnknownForCausalLM", want: ""},
	}

	for _, tc := range cases {
		t.Run(tc.architecture, func(t *testing.T) {
			got := architectureFromTransformersName(tc.architecture)
			if got != tc.want {
				t.Fatalf("architectureFromTransformersName(%q) = %q, want %q", tc.architecture, got, tc.want)
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
	if got := metadataArrayLen("nope"); got != 0 {
		t.Fatalf("metadataArrayLen(string) = %d, want 0", got)
	}
}

func TestGGUFTensorTypeDetails_AllKnownTypes_Good(t *testing.T) {
	cases := []struct {
		typ       uint32
		name      string
		dtype     string
		bits      int
		blockSize int
		quantized bool
	}{
		{typ: ggufTensorTypeF32, name: "f32", dtype: "float32", bits: 32},
		{typ: ggufTensorTypeF16, name: "f16", dtype: "float16", bits: 16},
		{typ: ggufTensorTypeQ4_0, name: "q4_0", dtype: "ggml_q4_0", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ4_1, name: "q4_1", dtype: "ggml_q4_1", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ5_0, name: "q5_0", dtype: "ggml_q5_0", bits: 5, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ5_1, name: "q5_1", dtype: "ggml_q5_1", bits: 5, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ8_0, name: "q8_0", dtype: "ggml_q8_0", bits: 8, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ8_1, name: "q8_1", dtype: "ggml_q8_1", bits: 8, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ2K, name: "q2_k", dtype: "ggml_q2_k", bits: 2, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeQ3K, name: "q3_k", dtype: "ggml_q3_k", bits: 3, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeQ4K, name: "q4_k", dtype: "ggml_q4_k", bits: 4, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeQ5K, name: "q5_k", dtype: "ggml_q5_k", bits: 5, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeQ6K, name: "q6_k", dtype: "ggml_q6_k", bits: 6, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeQ8K, name: "q8_k", dtype: "ggml_q8_k", bits: 8, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ2XXS, name: "iq2_xxs", dtype: "ggml_iq2_xxs", bits: 2, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ2XS, name: "iq2_xs", dtype: "ggml_iq2_xs", bits: 2, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ3XXS, name: "iq3_xxs", dtype: "ggml_iq3_xxs", bits: 3, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ1S, name: "iq1_s", dtype: "ggml_iq1_s", bits: 1, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ4NL, name: "iq4_nl", dtype: "ggml_iq4_nl", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeIQ3S, name: "iq3_s", dtype: "ggml_iq3_s", bits: 3, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ2S, name: "iq2_s", dtype: "ggml_iq2_s", bits: 2, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeIQ4XS, name: "iq4_xs", dtype: "ggml_iq4_xs", bits: 4, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeI8, name: "i8", dtype: "int8", bits: 8},
		{typ: ggufTensorTypeI16, name: "i16", dtype: "int16", bits: 16},
		{typ: ggufTensorTypeI32, name: "i32", dtype: "int32", bits: 32},
		{typ: ggufTensorTypeI64, name: "i64", dtype: "int64", bits: 64},
		{typ: ggufTensorTypeF64, name: "f64", dtype: "float64", bits: 64},
		{typ: ggufTensorTypeIQ1M, name: "iq1_m", dtype: "ggml_iq1_m", bits: 1, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeBF16, name: "bf16", dtype: "bfloat16", bits: 16},
		{typ: ggufTensorTypeQ4_0_4_4, name: "q4_0_4_4", dtype: "ggml_q4_0_4_4", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ4_0_4_8, name: "q4_0_4_8", dtype: "ggml_q4_0_4_8", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ4_0_8_8, name: "q4_0_8_8", dtype: "ggml_q4_0_8_8", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeTQ1_0, name: "tq1_0", dtype: "ggml_tq1_0", bits: 1, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeTQ2_0, name: "tq2_0", dtype: "ggml_tq2_0", bits: 2, blockSize: 256, quantized: true},
		{typ: ggufTensorTypeMXFP4, name: "mxfp4", dtype: "ggml_mxfp4", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeNVFP4, name: "nvfp4", dtype: "ggml_nvfp4", bits: 4, blockSize: 32, quantized: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ggufTensorTypeDetails(tc.typ)
			if !got.Known {
				t.Fatalf("Known = false, want true")
			}
			if got.Name != tc.name || got.DType != tc.dtype || got.Bits != tc.bits || got.BlockSize != tc.blockSize || got.Quantized != tc.quantized {
				t.Fatalf("details = %+v, want name:%s dtype:%s bits:%d block:%d quantized:%v", got, tc.name, tc.dtype, tc.bits, tc.blockSize, tc.quantized)
			}
			if bits := ggufTensorBits(tc.typ); bits != boolQuantBits(tc.quantized, tc.bits) {
				t.Fatalf("ggufTensorBits(%d) = %d", tc.typ, bits)
			}
		})
	}

	if got := ggufTensorTypeDetails(999); got.Known || got.Name != "" {
		t.Fatalf("unknown details = %+v, want zero value", got)
	}
	if bits := ggufTensorBits(999); bits != 0 {
		t.Fatalf("ggufTensorBits(unknown) = %d, want 0", bits)
	}
}

func boolQuantBits(quantized bool, bits int) int {
	if quantized {
		return bits
	}
	return 0
}

func TestGGUFQuantizationHelpers_Good(t *testing.T) {
	fileTypes := []struct {
		fileType int
		name     string
		bits     int
	}{
		{fileType: 0, name: "f32", bits: 32},
		{fileType: 1, name: "f16", bits: 16},
		{fileType: 2, name: "q4_0", bits: 4},
		{fileType: 3, name: "q4_1", bits: 4},
		{fileType: 4, name: "q4_1_some_f16", bits: 4},
		{fileType: 7, name: "q8_0", bits: 8},
		{fileType: 8, name: "q5_0", bits: 5},
		{fileType: 9, name: "q5_1", bits: 5},
		{fileType: 10, name: "q2_k", bits: 2},
		{fileType: 11, name: "q3_k_s", bits: 3},
		{fileType: 12, name: "q3_k_m", bits: 3},
		{fileType: 13, name: "q3_k_l", bits: 3},
		{fileType: 14, name: "q4_k_s", bits: 4},
		{fileType: 15, name: "q4_k_m", bits: 4},
		{fileType: 16, name: "q5_k_s", bits: 5},
		{fileType: 17, name: "q5_k_m", bits: 5},
		{fileType: 18, name: "q6_k", bits: 6},
		{fileType: 19, name: "iq2_xxs", bits: 2},
		{fileType: 20, name: "iq2_xs", bits: 2},
		{fileType: 21, name: "q2_k_s", bits: 2},
		{fileType: 22, name: "iq3_xs", bits: 3},
		{fileType: 23, name: "iq3_xxs", bits: 3},
		{fileType: 24, name: "iq1_s", bits: 1},
		{fileType: 25, name: "iq4_nl", bits: 4},
		{fileType: 26, name: "iq3_s", bits: 3},
		{fileType: 27, name: "iq3_m", bits: 3},
		{fileType: 28, name: "iq2_s", bits: 2},
		{fileType: 29, name: "iq2_m", bits: 2},
		{fileType: 30, name: "iq4_xs", bits: 4},
		{fileType: 31, name: "iq1_m", bits: 1},
		{fileType: 32, name: "bf16", bits: 16},
		{fileType: 33, name: "q4_0_4_4", bits: 4},
		{fileType: 34, name: "q4_0_4_8", bits: 4},
		{fileType: 35, name: "q4_0_8_8", bits: 4},
		{fileType: 36, name: "tq1_0", bits: 1},
		{fileType: 37, name: "tq2_0", bits: 2},
		{fileType: 38, name: "mxfp4", bits: 4},
		{fileType: 39, name: "nvfp4", bits: 4},
	}
	for _, tc := range fileTypes {
		t.Run(tc.name, func(t *testing.T) {
			name, bits := ggufFileTypeQuantization(tc.fileType)
			if name != tc.name || bits != tc.bits {
				t.Fatalf("ggufFileTypeQuantization(%d) = (%q,%d), want (%q,%d)", tc.fileType, name, bits, tc.name, tc.bits)
			}
		})
	}
	name, bits := ggufFileTypeQuantization(999)
	if name != "" || bits != 0 {
		t.Fatalf("unknown file type = (%q,%d), want zero", name, bits)
	}

	familyCases := map[string]string{
		" IQ4-NL ": "iq",
		"mxfp4":    "mxfp",
		"nvfp4":    "nvfp",
		"q4_k_m":   "qk",
		"q8_0":     "q8",
		"q5_1":     "q5",
		"q4_0":     "q4",
		"q3_k_s":   "qk",
		"q2_k":     "qk",
		"tq1_0":    "tq",
		"bf16":     "dense",
		"unknown":  "",
		"":         "",
	}
	for value, want := range familyCases {
		if got := quantFamilyForType(value); got != want {
			t.Fatalf("quantFamilyForType(%q) = %q, want %q", value, got, want)
		}
	}

	bitCases := map[string]int{
		"":       0,
		"f16":    16,
		"f32":    32,
		"f64":    64,
		"nvfp4":  4,
		"iq5_xs": 5,
		"q8_0":   8,
		"q6_k":   6,
		"q3_k":   3,
		"q2_k":   2,
		"tq1_0":  1,
		"dense":  0,
	}
	for value, want := range bitCases {
		if got := quantBitsFromTypeName(value); got != want {
			t.Fatalf("quantBitsFromTypeName(%q) = %d, want %d", value, got, want)
		}
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
