// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// pure_helpers_cover_test.go covers the small pure (non-Metal) helpers scattered
// across model.go, pinned_array.go, and dtype.go that the heavier suites step
// over: architecture-name compaction, weight resolution, contiguous strides, and
// the DType JSON unmarshal. No Metal runtime — plain Go data.

func TestCompactArchitectureName_Good(t *testing.T) {
	cases := map[string]string{
		"Gemma-4_Text":      "gemma4text",
		"qwen3.5-moe":       "qwen35moe",
		"ALL-CAPS_NAME":     "allcapsname",
		"already":           "already",
		"":                  "",
	}
	for in, want := range cases {
		if got := CompactArchitectureName(in); got != want {
			t.Errorf("CompactArchitectureName(%q) = %q, want %q", in, got, want)
		}
		// the unexported alias must agree with the exported entry point.
		if got := compactArchitectureName(in); got != want {
			t.Errorf("compactArchitectureName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHasResolvedWeight_Good(t *testing.T) {
	// weightCandidates expands a logical name to its on-disk variants; a key
	// present under any candidate resolves. The values are irrelevant (presence
	// only), so nil arrays are fine.
	weights := map[string]*Array{
		"model.layers.0.self_attn.q_proj.weight": nil,
	}
	if !HasResolvedWeight(weights, "model.layers.0.self_attn.q_proj.weight") {
		t.Error("HasResolvedWeight missed an exact key")
	}
	if HasResolvedWeight(weights, "model.layers.0.self_attn.k_proj.weight") {
		t.Error("HasResolvedWeight matched an absent key")
	}
	if HasResolvedWeight(map[string]*Array{}, "anything") {
		t.Error("HasResolvedWeight matched in an empty map")
	}
	// the unexported alias delegates to the exported function.
	if !hasResolvedWeight(weights, "model.layers.0.self_attn.q_proj.weight") {
		t.Error("hasResolvedWeight alias disagreed with HasResolvedWeight")
	}
}

func TestContiguousStrides_Good(t *testing.T) {
	// Row-major strides for [2,3,4]: innermost = 1, then 4, then 12.
	got := contiguousStrides([]int{2, 3, 4})
	want := []int64{12, 4, 1}
	if len(got) != len(want) {
		t.Fatalf("strides len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("contiguousStrides([2 3 4]) = %v, want %v", got, want)
		}
	}
	// Scalar shape → empty strides.
	if s := contiguousStrides(nil); len(s) != 0 {
		t.Errorf("nil-shape strides = %v, want empty", s)
	}
	// A single dimension has stride 1.
	if s := contiguousStrides([]int{5}); len(s) != 1 || s[0] != 1 {
		t.Errorf("[5] strides = %v, want [1]", s)
	}
}

func TestDType_UnmarshalJSON_Good(t *testing.T) {
	// Known type names decode to their DType.
	cases := map[string]DType{
		`"F32"`:      DTypeFloat32,
		`"BF16"`:     DTypeBFloat16,
		`"int8"`:     DTypeInt8,
		`"bool"`:     DTypeBool,
		`"complex64"`: DTypeComplex64,
	}
	for raw, want := range cases {
		var d DType
		if err := d.UnmarshalJSON([]byte(raw)); err != nil {
			t.Fatalf("UnmarshalJSON(%s) error: %v", raw, err)
		}
		if d != want {
			t.Errorf("UnmarshalJSON(%s) = %v, want %v", raw, d, want)
		}
	}

	// An unknown type name falls back to float32 rather than erroring.
	var d DType
	if err := d.UnmarshalJSON([]byte(`"not-a-dtype"`)); err != nil {
		t.Fatalf("unknown dtype unmarshal error: %v", err)
	}
	if d != DTypeFloat32 {
		t.Errorf("unknown dtype → %v, want float32 fallback", d)
	}

	// Malformed JSON (a number, not a string) returns the decode error.
	var bad DType
	if err := bad.UnmarshalJSON([]byte(`123`)); err == nil {
		t.Error("non-string dtype JSON unmarshal returned nil error")
	}
}
