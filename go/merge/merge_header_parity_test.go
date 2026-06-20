// SPDX-Licence-Identifier: EUPL-1.2

// Byte-parity guard for marshalMergedHeader. The merged model.safetensors
// header MUST be byte-for-byte identical to what core.JSONMarshal produced
// before the hand-rolled emitter replaced it — a parse-back round-trip is
// NOT sufficient (it would pass on format drift that still corrupts byte
// offsets), so this diffs the emitter's raw bytes against core.JSONMarshal
// directly over adversarial inputs.
//
// Run:    go test -run='MarshalMergedHeader' ./go/merge

package merge

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

func TestMarshalMergedHeaderParity(t *testing.T) {
	cases := []struct {
		name   string
		header map[string]safetensors.HeaderEntry
	}{
		{
			// nil map -> null (encoding/json semantics); unreachable from
			// buildMergedHeader but pins the emitter's equivalence.
			name:   "nil_map",
			header: nil,
		},
		{
			name:   "empty",
			header: map[string]safetensors.HeaderEntry{},
		},
		{
			name: "single",
			header: map[string]safetensors.HeaderEntry{
				"model.norm.weight": {DType: "F32", Shape: []int64{4096}, DataOffsets: []int64{0, 16384}},
			},
		},
		{
			// File order (insertion) deliberately not alphabetical — pins the
			// emitter's key sort against encoding/json's sorted map output.
			name: "file_order_not_alphabetical",
			header: map[string]safetensors.HeaderEntry{
				"zeta":  {DType: "F32", Shape: []int64{2}, DataOffsets: []int64{0, 8}},
				"alpha": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{8, 12}},
				"mid":   {DType: "F32", Shape: []int64{3}, DataOffsets: []int64{12, 24}},
			},
		},
		{
			// Multi-dim shapes — qwen3-class attention/MLP weights.
			name: "multi_dim_shapes",
			header: map[string]safetensors.HeaderEntry{
				"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []int64{4096, 4096}, DataOffsets: []int64{0, 67108864}},
				"model.layers.0.mlp.gate_proj.weight":    {DType: "F32", Shape: []int64{14336, 4096, 2}, DataOffsets: []int64{67108864, 67108864 + 469762048}},
			},
		},
		{
			// Scalar (non-nil empty shape) must emit [] not null.
			name: "scalar_empty_shape",
			header: map[string]safetensors.HeaderEntry{
				"scalar": {DType: "F32", Shape: []int64{}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Genuinely nil shape must emit null (json.Marshal semantics).
			name: "nil_shape",
			header: map[string]safetensors.HeaderEntry{
				"nilshape": {DType: "F32", Shape: nil, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// THE discriminator: HTML-meta characters in the name. encoding/json
			// default-escapes < > & to < > & — the emitter must
			// match (this is why it mirrors appendJSONStringHTML, not the
			// non-HTML safetensors appendJSONString).
			name: "html_meta_name",
			header: map[string]safetensors.HeaderEntry{
				"a<b>c&d": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Quote, backslash, and control-byte escapes in a name.
			name: "escape_chars_name",
			header: map[string]safetensors.HeaderEntry{
				"q\"b\\s\tt\n\r\x01\x1f": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Unicode line/paragraph separators (U+2028/U+2029) →  / .
			name: "unicode_separators",
			header: map[string]safetensors.HeaderEntry{
				"a b c": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Multi-byte UTF-8 that passes through unescaped.
			name: "valid_multibyte_utf8",
			header: map[string]safetensors.HeaderEntry{
				"café-日本語-emoji😀": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Invalid UTF-8 byte → � (encoding/json emits the escaped
			// replacement char, not the raw bytes).
			name: "invalid_utf8",
			header: map[string]safetensors.HeaderEntry{
				"bad\xffbyte": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
		{
			// Large int64 offsets (multi-GB merged checkpoint).
			name: "large_offsets",
			header: map[string]safetensors.HeaderEntry{
				"big": {DType: "F32", Shape: []int64{1073741824}, DataOffsets: []int64{0, 4294967296}},
			},
		},
		{
			// Negative offset / dim — must still match strconv base-10 form.
			// (Never produced by buildMergedHeader, but pins the int emitter.)
			name: "negative_ints",
			header: map[string]safetensors.HeaderEntry{
				"neg": {DType: "F32", Shape: []int64{-1, -128}, DataOffsets: []int64{-9223372036854775808, 9223372036854775807}},
			},
		},
		{
			// Empty dtype string.
			name: "empty_dtype",
			header: map[string]safetensors.HeaderEntry{
				"d": {DType: "", Shape: []int64{1}, DataOffsets: []int64{0, 4}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := core.JSONMarshal(tc.header)
			if !want.OK {
				t.Fatalf("core.JSONMarshal failed: %v", want.Value)
			}
			wantBytes := want.Value.([]byte)
			got := marshalMergedHeader(tc.header)
			if !bytes.Equal(got, wantBytes) {
				t.Fatalf("byte mismatch:\n  json: %s\n  hand: %s", wantBytes, got)
			}
		})
	}
}

// TestMarshalMergedHeaderParity_BuildMergedHeader closes the loop end-to-end:
// the exact header buildMergedHeader produces from a synthetic index must
// marshal identically through the emitter and core.JSONMarshal — covering the
// real shape/dtype/offset shapes (non-nil empty shape for a scalar tensor,
// F32 dtype, sequential offsets) rather than only hand-built fixtures.
func TestMarshalMergedHeaderParity_BuildMergedHeader(t *testing.T) {
	idx := safetensors.Index{
		Names: []string{
			"model.layers.1.mlp.down_proj.weight",
			"model.layers.0.self_attn.q_proj.weight",
			"model.norm.weight",
			"lm_head.weight",
		},
		Tensors: map[string]safetensors.TensorRef{
			"model.layers.1.mlp.down_proj.weight":    {Name: "model.layers.1.mlp.down_proj.weight", Shape: []uint64{4096, 14336}, Elements: 4096 * 14336},
			"model.layers.0.self_attn.q_proj.weight": {Name: "model.layers.0.self_attn.q_proj.weight", Shape: []uint64{4096, 4096}, Elements: 4096 * 4096},
			"model.norm.weight":                      {Name: "model.norm.weight", Shape: []uint64{4096}, Elements: 4096},
			"scalar.tensor":                          {Name: "scalar.tensor", Shape: []uint64{}, Elements: 1},
			"lm_head.weight":                         {Name: "lm_head.weight", Shape: []uint64{151936, 4096}, Elements: 151936 * 4096},
		},
	}
	// Include the scalar tensor in Names so buildMergedHeader walks it.
	idx.Names = append(idx.Names, "scalar.tensor")

	header := buildMergedHeader(idx)
	want := core.JSONMarshal(header)
	if !want.OK {
		t.Fatalf("core.JSONMarshal failed: %v", want.Value)
	}
	wantBytes := want.Value.([]byte)
	got := marshalMergedHeader(header)
	if !bytes.Equal(got, wantBytes) {
		t.Fatalf("byte mismatch:\n  json: %s\n  hand: %s", wantBytes, got)
	}
}
