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
				for i := 0; i < 32; i++ {
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
			for i := 0; i < 200; i++ {
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
