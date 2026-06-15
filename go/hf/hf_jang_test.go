// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"testing"
)

// TestInferJANG_BasicProfile_Good drives the public InferJANG over a pack
// whose id carries a "jang_2s" needle but no "jangtq" — the jangBasic branch
// that builds the lowercase haystack, resolves the profile name, and reads
// the group size from the QuantizationConfig. Asserts the inferred profile,
// the bits derived from jang.ProfileBits ("jang_2*" -> 2), and the overridden
// group size (96, not the 64 default).
func TestInferJANG_BasicProfile_Good(t *testing.T) {
	meta := ModelMetadata{
		ID:   "dealignai/Qwen3-JANG_2S",
		Tags: []string{"mlx", "jang"},
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "tokenizer.json"},
		},
		Config: ModelConfig{
			QuantizationConfig: &QuantizationConfig{GroupSize: 96},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a 'jang_2s' pack, want a basic JANG profile")
	}
	if info.Profile != "JANG_2S" {
		t.Fatalf("Profile = %q, want JANG_2S", info.Profile)
	}
	if info.BitsDefault != 2 {
		t.Fatalf("BitsDefault = %d, want 2 (jang_2* -> 2 bits)", info.BitsDefault)
	}
	if info.GroupSize != 96 {
		t.Fatalf("GroupSize = %d, want 96 (read from QuantizationConfig, not the 64 default)", info.GroupSize)
	}
	if info.Packed == nil {
		t.Fatal("Packed profile = nil, want BuildPackedProfile output")
	}
}

// TestInferJANG_NoNeedle_Bad asserts the dominant miss path: metadata with no
// "jang" token anywhere (id/tags/filenames) returns nil with no profile work.
func TestInferJANG_NoNeedle_Bad(t *testing.T) {
	meta := ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}, {Name: "tokenizer.json"}},
	}
	if info := InferJANG(meta); info != nil {
		t.Fatalf("InferJANG = %+v, want nil for a non-JANG pack", info)
	}
}

// TestInferJANG_TQNeedleNoGroupSize_Ugly drives the JANGTQ short-circuit when
// the strongest token is "jangtq" (here only in a filename) and neither quant
// block declares a group size — the helper must fall back to the 64 default
// and stamp the fixed JANGTQ profile/bits without scanning a haystack.
func TestInferJANG_TQNeedleNoGroupSize_Ugly(t *testing.T) {
	meta := ModelMetadata{
		ID: "vendor/model-with-only-a-file-needle",
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "weights.JANGTQ.safetensors"},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a JANGTQ filename, want a JANGTQ profile")
	}
	if info.Profile != "JANGTQ" || info.WeightFormat != "mxtq" {
		t.Fatalf("profile/format = %q/%q, want JANGTQ/mxtq", info.Profile, info.WeightFormat)
	}
	if info.BitsDefault != 2 || info.RoutedExpertBits != 2 {
		t.Fatalf("bits = default:%d routed:%d, want 2/2", info.BitsDefault, info.RoutedExpertBits)
	}
	if info.GroupSize != 64 {
		t.Fatalf("GroupSize = %d, want 64 default (no quant block declared a group size)", info.GroupSize)
	}
}
