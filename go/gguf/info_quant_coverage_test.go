// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"
)

// TestQuantFamilyForType_BareQ3Q2_Good covers the q3/q2 prefix arms of
// quantFamilyForType that the existing cases miss: q3_k_s / q2_k both contain
// "_k" and return "qk" before reaching those arms, so a hypothetical bare
// q3_0 / q2_0 (no "_k") is needed to exercise lines that map a leading q3/q2
// to its own family.
func TestQuantFamilyForType_BareQ3Q2_Good(t *testing.T) {
	cases := map[string]string{
		"q3_0": "q3",
		"q2_0": "q2",
	}
	for in, want := range cases {
		if got := quantFamilyForType(in); got != want {
			t.Fatalf("quantFamilyForType(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestGGUFTensorElements_EmptyShape_Good covers the len(shape)==0 -> 0 arm of
// ggufTensorElements (a tensor with no dimensions has zero elements).
func TestGGUFTensorElements_EmptyShape_Good(t *testing.T) {
	if got := ggufTensorElements(nil); got != 0 {
		t.Fatalf("ggufTensorElements(nil) = %d, want 0", got)
	}
	if got := ggufTensorElements([]uint64{}); got != 0 {
		t.Fatalf("ggufTensorElements(empty) = %d, want 0", got)
	}
	// Sanity: a populated shape multiplies through.
	if got := ggufTensorElements([]uint64{4, 8}); got != 32 {
		t.Fatalf("ggufTensorElements([4,8]) = %d, want 32", got)
	}
}

// TestSummarizeGGUFTensorTypes_Empty_Good covers the len(tensors)==0 -> nil
// early return.
func TestSummarizeGGUFTensorTypes_Empty_Good(t *testing.T) {
	if got := summarizeGGUFTensorTypes(nil); got != nil {
		t.Fatalf("summarizeGGUFTensorTypes(nil) = %v, want nil", got)
	}
}

// TestIndexString_Guards_Good covers indexString's two guard arms: an empty
// substring returns 0, and a substring longer than the string returns -1.
func TestIndexString_Guards_Good(t *testing.T) {
	if got := indexString("blk.0.weight", ""); got != 0 {
		t.Fatalf("indexString(empty substr) = %d, want 0", got)
	}
	if got := indexString("ab", "abcdef"); got != -1 {
		t.Fatalf("indexString(substr longer) = %d, want -1", got)
	}
	// Sanity: a real hit returns the index.
	if got := indexString("xxblk.", "blk."); got != 2 {
		t.Fatalf("indexString(hit) = %d, want 2", got)
	}
}

// TestBuildGGUFTensorInfos_EmptyShapeIssue_Bad covers the
// len(tensor.Shape)==0 validation arm of buildGGUFTensorInfos: a tensor with
// no shape dimensions raises an invalid_tensor_shape issue. A known scalar
// type (f32) is used so the unknown-type arm does not fire instead.
func TestBuildGGUFTensorInfos_EmptyShapeIssue_Bad(t *testing.T) {
	infos, issues := buildGGUFTensorInfos([]ggufTensorInfo{
		{Name: "scalar.weight", Type: ggufTensorTypeF32, Shape: nil},
	})
	if len(infos) != 1 {
		t.Fatalf("buildGGUFTensorInfos returned %d infos, want 1", len(infos))
	}
	found := false
	for _, issue := range issues {
		if issue.Code == "invalid_tensor_shape" && issue.Tensor == "scalar.weight" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected invalid_tensor_shape issue, got %+v", issues)
	}
}

// TestInferGGUFQuantization_FamilyFromMajority_Good drives the
// `family == "" && majorityType != ""` fallback: the explicit quant type is
// empty/unfamilied while the majority tensor type carries a recognisable
// family, so inferGGUFQuantization derives the family from the majority.
func TestInferGGUFQuantization_FamilyFromMajority_Good(t *testing.T) {
	// No quantization.* metadata at all -> explicitType == "". Two q4_k
	// tensors make q4_k the majority (family "qk").
	tensors := []TensorInfo{
		{Type: ggufTensorTypeQ4K, TypeName: "q4_k", DType: "ggml_q4_k", Bits: 4, BlockSize: 256, Quantized: true},
		{Type: ggufTensorTypeQ4K, TypeName: "q4_k", DType: "ggml_q4_k", Bits: 4, BlockSize: 256, Quantized: true},
	}
	q := inferGGUFQuantization(map[string]any{}, tensors)
	if q.Family != "qk" {
		t.Fatalf("inferGGUFQuantization family = %q, want qk (from majority)", q.Family)
	}
}

// TestInferGGUFQuantization_EmptyFamilyFallsBackToMajority_Good drives the
// `family == "" && majorityType != ""` recovery arm: an explicit quant type
// string with no recognisable family (so quantFamilyForType returns "") while
// the majority tensor type DOES have a family, so the family is re-derived
// from the majority.
func TestInferGGUFQuantization_EmptyFamilyFallsBackToMajority_Good(t *testing.T) {
	// general.quantization_type = "custom" -> NormalizeQuantType keeps it,
	// quantFamilyForType("custom") == "". The majority tensor type q6_k has
	// family "qk", so the fallback supplies "qk".
	metadata := map[string]any{"general.quantization_type": "custom"}
	tensors := []TensorInfo{
		{Type: ggufTensorTypeQ6K, TypeName: "q6_k", DType: "ggml_q6_k", Bits: 6, BlockSize: 256, Quantized: true},
		{Type: ggufTensorTypeQ6K, TypeName: "q6_k", DType: "ggml_q6_k", Bits: 6, BlockSize: 256, Quantized: true},
	}
	q := inferGGUFQuantization(metadata, tensors)
	if q.Family != "qk" {
		t.Fatalf("inferGGUFQuantization family = %q, want qk (re-derived from majority)", q.Family)
	}
}

// TestGGUFQuantizationIsMixed_MultipleQuantTypes_Good drives the
// quantisedCount>1 -> true arm: two distinct quantised tensor types in the
// summary mean the file is mixed even when the quant type name has no "_m"
// suffix.
func TestGGUFQuantizationIsMixed_MultipleQuantTypes_Good(t *testing.T) {
	summaries := []TensorTypeSummary{
		{Type: TensorTypeQ8_0, Name: "q8_0", Quantized: true, Count: 3},
		{Type: ggufTensorTypeQ4K, Name: "q4_k", Quantized: true, Count: 2},
	}
	if !ggufQuantizationIsMixed("q8_0", summaries) {
		t.Fatal("ggufQuantizationIsMixed(two quant types) = false, want true")
	}
	// Single quantised type, no _m suffix -> not mixed.
	single := []TensorTypeSummary{{Type: TensorTypeQ8_0, Name: "q8_0", Quantized: true, Count: 3}}
	if ggufQuantizationIsMixed("q8_0", single) {
		t.Fatal("ggufQuantizationIsMixed(single quant type) = true, want false")
	}
}
