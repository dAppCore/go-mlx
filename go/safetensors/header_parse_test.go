// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// TestParseHeader_Parity_Synthetic asserts the hand-rolled parser
// produces a TensorRef that matches the encoding/json reference
// across a representative spread of dtype/shape/offset shapes — the
// safety net for the W8-I refactor.
func TestParseHeader_Parity_Synthetic(t *testing.T) {
	cases := []struct {
		name    string
		entries map[string]HeaderEntry
	}{
		{
			name: "single_2d_f32",
			entries: map[string]HeaderEntry{
				"weight": {DType: "F32", Shape: []int64{2048, 2048}, DataOffsets: []int64{0, 2048 * 2048 * 4}},
			},
		},
		{
			name: "multi_dim_f16",
			entries: map[string]HeaderEntry{
				"model.layers.0.self_attn.q_proj.weight": {DType: "F16", Shape: []int64{4, 28, 2048, 64}, DataOffsets: []int64{0, 4 * 28 * 2048 * 64 * 2}},
				"model.layers.0.self_attn.k_proj.weight": {DType: "BF16", Shape: []int64{4, 28, 2048, 64}, DataOffsets: []int64{4 * 28 * 2048 * 64 * 2, 2 * 4 * 28 * 2048 * 64 * 2}},
			},
		},
		{
			name: "one_dim_with_metadata",
			entries: map[string]HeaderEntry{
				"bias":       {DType: "F32", Shape: []int64{128}, DataOffsets: []int64{0, 512}},
				"embeddings": {DType: "F32", Shape: []int64{1024, 64}, DataOffsets: []int64{512, 512 + 1024*64*4}},
			},
		},
		{
			name: "many_small_tensors",
			entries: func() map[string]HeaderEntry {
				m := map[string]HeaderEntry{}
				var offset int64
				for i := range 32 {
					n := "model.layers." + stIntStr(i/4) + ".self_attn.q_proj.weight." + stIntStr(i%4)
					m[n] = HeaderEntry{DType: "U8", Shape: []int64{int64(16)}, DataOffsets: []int64{offset, offset + 16}}
					offset += 16
				}
				return m
			}(),
		},
		{
			name: "lowercase_dtype",
			entries: map[string]HeaderEntry{
				"x": {DType: "f32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.JoinPath(t.TempDir(), tc.name+".safetensors")
			writeHeaderOnly(t, path, tc.entries, false)
			got, err := ReadIndex(path)
			if err != nil {
				t.Fatalf("ReadIndex: %v", err)
			}
			assertIndexEntries(t, got, tc.entries, path)
		})
	}
}

// TestParseHeader_MetadataSkipped confirms the __metadata__ entry is
// honoured (not present in Tensors/Names) regardless of its body shape.
func TestParseHeader_MetadataSkipped(t *testing.T) {
	entries := map[string]HeaderEntry{
		"weight": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
	}
	path := core.JoinPath(t.TempDir(), "metadata.safetensors")
	writeHeaderOnly(t, path, entries, true)
	got, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	if _, ok := got.Tensors["__metadata__"]; ok {
		t.Fatalf("__metadata__ leaked into Tensors")
	}
	for _, n := range got.Names {
		if n == "__metadata__" {
			t.Fatalf("__metadata__ leaked into Names")
		}
	}
	if len(got.Names) != 1 || got.Names[0] != "weight" {
		t.Fatalf("Names = %v, want [weight]", got.Names)
	}
}

// TestParseHeader_DuplicateRejected confirms the hand-rolled parser
// surfaces duplicate keys (would-be silent overwrites under the old
// map-keyed json.Unmarshal path).
func TestParseHeader_DuplicateRejected(t *testing.T) {
	// Hand-craft a header with a duplicate key — json.Marshal cannot
	// produce one, so we build the JSON literally.
	headerJSON := []byte(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`)
	out := make([]byte, 8+len(headerJSON)+8)
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerJSON)))
	copy(out[8:], headerJSON)
	path := core.JoinPath(t.TempDir(), "dup.safetensors")
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
	if _, err := ReadIndex(path); err == nil {
		t.Fatalf("ReadIndex(duplicate) error = nil")
	}
}

// TestParseHeader_KeyOrderTolerated confirms inner key order does not
// affect the parsed TensorRef — python's json.dumps and the rust
// safetensors crate emit different orderings.
func TestParseHeader_KeyOrderTolerated(t *testing.T) {
	orderings := []string{
		`{"x":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`,
		`{"x":{"shape":[2,3],"dtype":"F32","data_offsets":[0,24]}}`,
		`{"x":{"data_offsets":[0,24],"shape":[2,3],"dtype":"F32"}}`,
		`{"x":{"data_offsets":[0,24],"dtype":"F32","shape":[2,3]}}`,
	}
	for _, headerJSON := range orderings {
		out := make([]byte, 8+len(headerJSON)+24)
		binary.LittleEndian.PutUint64(out[:8], uint64(len(headerJSON)))
		copy(out[8:], headerJSON)
		path := core.JoinPath(t.TempDir(), "order.safetensors")
		if result := core.WriteFile(path, out, 0o644); !result.OK {
			t.Fatalf("WriteFile: %v", result.Value)
		}
		got, err := ReadIndex(path)
		if err != nil {
			t.Fatalf("ReadIndex(%s): %v", headerJSON, err)
		}
		ref := got.Tensors["x"]
		if ref.DType != "F32" {
			t.Fatalf("DType = %q, want F32", ref.DType)
		}
		if len(ref.Shape) != 2 || ref.Shape[0] != 2 || ref.Shape[1] != 3 {
			t.Fatalf("Shape = %v, want [2 3]", ref.Shape)
		}
		if ref.DataStart != int64(8+len(headerJSON)) || ref.ByteLen != 24 {
			t.Fatalf("DataStart=%d ByteLen=%d, want %d 24", ref.DataStart, ref.ByteLen, 8+len(headerJSON))
		}
		if ref.Elements != 6 {
			t.Fatalf("Elements = %d, want 6", ref.Elements)
		}
	}
}

// TestCountTensorsAndDims_Synthetic stress-tests the cheap first-pass
// counter on the same fixtures used by the parity test.
func TestCountTensorsAndDims_Synthetic(t *testing.T) {
	cases := []struct {
		name     string
		entries  map[string]HeaderEntry
		metadata bool
		tensors  int
		dims     int
	}{
		{"one_tensor", map[string]HeaderEntry{
			"w": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
		}, false, 1, 1},
		{"two_tensors_with_metadata", map[string]HeaderEntry{
			"w": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
			"b": {DType: "F16", Shape: []int64{2, 3}, DataOffsets: []int64{16, 28}},
		}, true, 2, 3},
		{"qwen_shape", func() map[string]HeaderEntry {
			m := map[string]HeaderEntry{}
			var offset int64
			for i := range 200 {
				n := "model.layers." + stIntStr(i/4) + ".self_attn.q_proj.weight." + stIntStr(i%4)
				m[n] = HeaderEntry{DType: "U8", Shape: []int64{16}, DataOffsets: []int64{offset, offset + 16}}
				offset += 16
			}
			return m
		}(), false, 200, 200},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.JoinPath(t.TempDir(), tc.name+".safetensors")
			writeHeaderOnly(t, path, tc.entries, tc.metadata)
			// Read the header bytes back exactly as ReadIndex does.
			opened := core.Open(path)
			if !opened.OK {
				t.Fatalf("Open: %v", opened.Value)
			}
			file := opened.Value.(*core.OSFile)
			defer file.Close()
			var lenBuf [8]byte
			if _, err := file.Read(lenBuf[:]); err != nil {
				t.Fatalf("Read len: %v", err)
			}
			headerLen := binary.LittleEndian.Uint64(lenBuf[:])
			headerBytes := make([]byte, headerLen)
			if _, err := file.Read(headerBytes); err != nil {
				t.Fatalf("Read header: %v", err)
			}
			tensors, dims := countTensorsAndDims(headerBytes)
			if tensors != tc.tensors {
				t.Fatalf("tensors = %d, want %d", tensors, tc.tensors)
			}
			if dims != tc.dims {
				t.Fatalf("dims = %d, want %d", dims, tc.dims)
			}
		})
	}
}

// TestParseString_Direct exercises parseString — the allocate-on-read
// string reader that peekStringSpan (zero-alloc span) superseded in the
// hot path. It is dead in production (no caller) and is asserted here only
// on the paths that are sound: a clean string and the not-a-string /
// unterminated rejections. parseString's escape-delegation arm is NOT
// asserted for a decoded value — it hands a stale p.pos (still at the
// opening quote) to parseStringEscaped and yields "" — a latent quirk that
// is harmless precisely because the function has no caller. The live
// escaped-string path is parseStringEscaped via materialiseString, which
// IS covered (TestSafetensors_WriteSubset_UglyEscapedName round-trips a
// name carrying a quote, backslash, newline and a control byte).
func TestParseString_Direct(t *testing.T) {
	t.Run("plain", func(t *testing.T) {
		p := jsonParser{data: []byte(`"hello" rest`)}
		got, ok := p.parseString()
		if !ok || got != "hello" {
			t.Fatalf("parseString = (%q,%v), want (hello,true)", got, ok)
		}
		// pos advanced past the closing quote, onto the space.
		if p.peek() != ' ' {
			t.Fatalf("pos not advanced past closing quote; peek = %q", p.peek())
		}
	})
	t.Run("not a string", func(t *testing.T) {
		p := jsonParser{data: []byte(`123`)}
		if _, ok := p.parseString(); ok {
			t.Fatal("parseString(non-string) ok = true, want false")
		}
	})
	t.Run("unterminated", func(t *testing.T) {
		p := jsonParser{data: []byte(`"no end`)}
		if _, ok := p.parseString(); ok {
			t.Fatal("parseString(unterminated) ok = true, want false")
		}
	})
}

// TestParseHeader_Malformed drives the header walker's error branches with
// hand-rolled bad header bytes through the public ParseHeaderRefs entry.
// Each case is a real malformed header a corrupt or hostile file could
// carry — reachable without any fault injection. dataStart is a fixed
// placeholder; these all fail before payload offsets matter.
func TestParseHeader_Malformed(t *testing.T) {
	cases := map[string]string{
		"not an object":            `[]`,
		"key not a string":         `{123:{}}`,
		"missing colon":            `{"w" {}}`,
		"tensor not an object":     `{"w":42}`,
		"shape not an array":       `{"w":{"dtype":"F32","shape":7,"data_offsets":[0,4]}}`,
		"shape dim not an integer": `{"w":{"dtype":"F32","shape":["x"],"data_offsets":[0,4]}}`,
		"offsets not an array":     `{"w":{"dtype":"F32","shape":[1],"data_offsets":7}}`,
		"offsets[0] not integer":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":["a",4]}}`,
		"offsets missing comma":    `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0 4]}}`,
		"offsets[1] not integer":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,"b"]}}`,
		"offsets unterminated":     `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4}}`,
		"dtype not a string":       `{"w":{"dtype":7,"shape":[1],"data_offsets":[0,4]}}`,
		"missing required field":   `{"w":{"dtype":"F32","data_offsets":[0,4]}}`,
		"negative offset begin":    `{"w":{"dtype":"F32","shape":[1],"data_offsets":[-1,4]}}`,
		"end before begin":         `{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}`,
		"trailing junk in entry":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4] 9}}`,
		"trailing junk top level":  `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]} 9}`,
		"duplicate tensor":         `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`,
		"unterminated header":      `{"w":{"dtype":"F32","shape":[1]`,
	}
	for name, header := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseHeaderRefs("p", []byte(header), 8); err == nil {
				t.Errorf("ParseHeaderRefs(%q) error = nil, want non-nil", header)
			}
		})
	}
}

// TestInternDType_Canonicalisation tables internDType across the full
// dtype vocabulary: uppercase canonicals (the fast path), the lowercase /
// mixed-case forms older writers emit (single-char normalise back to the
// canonical pointer), and a genuinely-unknown dtype that falls through to
// the core.Upper heap-string default. internDType is unexported so this is
// a white-box test; ReadIndex reaches it transitively but only for the few
// dtypes a fixture conveniently carries — the table is the honest tool for
// the per-byte branch matrix.
func TestInternDType_Canonicalisation(t *testing.T) {
	cases := map[string]string{
		// 2-byte.
		"I8": "I8", "i8": "I8", "U8": "U8", "u8": "U8",
		// 3-byte uppercase canonicals.
		"F16": "F16", "F32": "F32", "F64": "F64",
		"I16": "I16", "I32": "I32", "I64": "I64",
		"U16": "U16", "U32": "U32", "U64": "U64",
		// 3-byte lowercase / mixed — normalise to canonical.
		"f16": "F16", "f32": "F32", "f64": "F64",
		"i16": "I16", "i32": "I32", "i64": "I64",
		"u16": "U16", "u32": "U32", "u64": "U64",
		// 4-byte.
		"BF16": "BF16", "bf16": "BF16", "BOOL": "BOOL", "bool": "BOOL",
		// 7- and 9-byte float8 families, mixed case.
		"F8_E5M2": "F8_E5M2", "f8_e5m2": "F8_E5M2",
		"F8_E4M3FN": "F8_E4M3FN", "f8_e4m3fn": "F8_E4M3FN",
		// Unknown dtype → upper-cased heap string (the default arm).
		"complex64": "COMPLEX64", "weird": "WEIRD",
	}
	for in, want := range cases {
		if got := internDType([]byte(in)); got != want {
			t.Errorf("internDType(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestParseHeader_UnknownKeysSkipped confirms a tensor entry tolerates
// forward-compat keys it does not recognise: the walker skips the unknown
// value (here a nested array and an object) and still resolves the three
// required fields. This drives parseTensorEntry's default skipValue arm.
func TestParseHeader_UnknownKeysSkipped(t *testing.T) {
	header := `{"w":{"dtype":"F32","extra":[1,2,3],"shape":[2],"future":{"k":true},"data_offsets":[0,8]}}`
	index, err := ParseHeaderRefs("p", []byte(header), 8)
	if err != nil {
		t.Fatalf("ParseHeaderRefs: %v", err)
	}
	ref, ok := index.Tensors["w"]
	if !ok {
		t.Fatalf("tensor w missing; names = %v", index.Names)
	}
	if ref.DType != "F32" || ref.Elements != 2 || ref.ByteLen != 8 {
		t.Fatalf("ref = %+v, want F32/2 elements/8 bytes", ref)
	}
}

func assertIndexEntries(t *testing.T, got Index, expected map[string]HeaderEntry, path string) {
	t.Helper()
	if got.Path != path {
		t.Fatalf("Path = %q, want %q", got.Path, path)
	}
	wantCount := 0
	for k := range expected {
		if k != "__metadata__" {
			wantCount++
		}
	}
	if len(got.Tensors) != wantCount {
		t.Fatalf("len(Tensors) = %d, want %d", len(got.Tensors), wantCount)
	}
	if len(got.Names) != wantCount {
		t.Fatalf("len(Names) = %d, want %d", len(got.Names), wantCount)
	}
	for k, want := range expected {
		if k == "__metadata__" {
			continue
		}
		ref, ok := got.Tensors[k]
		if !ok {
			t.Fatalf("missing tensor %q", k)
		}
		if ref.Name != k {
			t.Fatalf("Name = %q, want %q", ref.Name, k)
		}
		if ref.Path != path {
			t.Fatalf("ref.Path = %q, want %q", ref.Path, path)
		}
		if ref.DType != core.Upper(want.DType) {
			t.Fatalf("DType = %q, want %q", ref.DType, core.Upper(want.DType))
		}
		if len(ref.Shape) != len(want.Shape) {
			t.Fatalf("len(Shape) = %d, want %d", len(ref.Shape), len(want.Shape))
		}
		for i, d := range want.Shape {
			if ref.Shape[i] != uint64(d) {
				t.Fatalf("Shape[%d] = %d, want %d", i, ref.Shape[i], d)
			}
		}
		elements := 1
		for _, d := range want.Shape {
			elements *= int(d)
		}
		if ref.Elements != elements {
			t.Fatalf("Elements = %d, want %d", ref.Elements, elements)
		}
		// DataStart = 8 + headerLen + want.DataOffsets[0]
		// ByteLen   = want.DataOffsets[1] - want.DataOffsets[0]
		if ref.ByteLen != want.DataOffsets[1]-want.DataOffsets[0] {
			t.Fatalf("ByteLen = %d, want %d", ref.ByteLen, want.DataOffsets[1]-want.DataOffsets[0])
		}
	}
}

// writeHeaderOnly lays down a synthetic safetensors file containing
// header + zero-byte payload region. Sized payloads are not needed —
// the parity test only inspects index output, not tensor bytes.
func writeHeaderOnly(t *testing.T, path string, entries map[string]HeaderEntry, includeMetadata bool) {
	t.Helper()
	header := map[string]any{}
	maxOffset := int64(0)
	for k, v := range entries {
		header[k] = map[string]any{
			"dtype":        v.DType,
			"shape":        v.Shape,
			"data_offsets": v.DataOffsets,
		}
		if v.DataOffsets[1] > maxOffset {
			maxOffset = v.DataOffsets[1]
		}
	}
	if includeMetadata {
		header["__metadata__"] = map[string]any{
			"format":  "pt",
			"version": "1",
			"extra":   "value with \"escapes\" and {braces} inside",
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+int(maxOffset))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}
