// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"
)

func TestInfoQuant_ggufTensorTypeDetails_Good(t *testing.T) {
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
		{typ: TensorTypeQ4_0, name: "q4_0", dtype: "ggml_q4_0", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ4_1, name: "q4_1", dtype: "ggml_q4_1", bits: 4, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ5_0, name: "q5_0", dtype: "ggml_q5_0", bits: 5, blockSize: 32, quantized: true},
		{typ: ggufTensorTypeQ5_1, name: "q5_1", dtype: "ggml_q5_1", bits: 5, blockSize: 32, quantized: true},
		{typ: TensorTypeQ8_0, name: "q8_0", dtype: "ggml_q8_0", bits: 8, blockSize: 32, quantized: true},
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

func TestInfoQuant_NormalizeQuantType_Good(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"Q4_K_M", "q4_k_m"},
		{"Q4-K-M", "q4_k_m"},
		{"  q8_0  ", "q8_0"},
		{"BF16", "bf16"},
		{"IQ4 XS", "iq4_xs"},
		{"Q5-K M", "q5_k_m"},
	}
	for _, tc := range cases {
		if got := NormalizeQuantType(tc.in); got != tc.want {
			t.Errorf("NormalizeQuantType(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestInfoQuant_NormalizeQuantType_Bad(t *testing.T) {
	// "Bad" input for a pure normaliser is junk that still has a defined,
	// total result — it lowercases and swaps separators without validating.
	cases := []struct {
		in   string
		want string
	}{
		{"not a real type", "not_a_real_type"},
		{"???", "???"},
		{"MIXED-garbage 99", "mixed_garbage_99"},
	}
	for _, tc := range cases {
		if got := NormalizeQuantType(tc.in); got != tc.want {
			t.Errorf("NormalizeQuantType(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestInfoQuant_NormalizeQuantType_Ugly(t *testing.T) {
	// Boundary inputs: empty, whitespace-only, and separator-only.
	if got := NormalizeQuantType(""); got != "" {
		t.Errorf("NormalizeQuantType(%q) = %q, want empty", "", got)
	}
	if got := NormalizeQuantType("   "); got != "" {
		t.Errorf("NormalizeQuantType(whitespace) = %q, want empty after trim", got)
	}
	if got := NormalizeQuantType(" - "); got != "_" {
		t.Errorf("NormalizeQuantType(%q) = %q, want %q", " - ", got, "_")
	}
}
