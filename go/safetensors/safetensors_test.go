// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestWriteSubset_Good(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "attention.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{
		"model.embed_tokens.weight":                  {1, 2, 3, 4},
		"model.layers.0.self_attn.q_proj.weight":     {5, 6, 7, 8},
		"model.layers.0.mlp.down_proj.weight":        {9, 10, 11, 12},
		"model.layers.0.self_attn.q_proj.weight.idx": {13, 14, 15, 16},
	})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}

	err = WriteSubset(context.Background(), target, []TensorRef{
		index.Tensors["model.embed_tokens.weight"],
		index.Tensors["model.layers.0.self_attn.q_proj.weight"],
	})
	if err != nil {
		t.Fatalf("WriteSubset: %v", err)
	}

	got, err := ReadIndex(target)
	if err != nil {
		t.Fatalf("ReadIndex(target): %v", err)
	}
	if len(got.Names) != 2 {
		t.Fatalf("names = %v, want two tensors", got.Names)
	}
	if _, ok := got.Tensors["model.layers.0.mlp.down_proj.weight"]; ok {
		t.Fatalf("target contains excluded MLP tensor: %v", got.Names)
	}
	assertRawTensorEqual(t, index.Tensors["model.embed_tokens.weight"], got.Tensors["model.embed_tokens.weight"])
	assertRawTensorEqual(t, index.Tensors["model.layers.0.self_attn.q_proj.weight"], got.Tensors["model.layers.0.self_attn.q_proj.weight"])
}

func TestWriteSubset_BadEmpty(t *testing.T) {
	err := WriteSubset(context.Background(), core.PathJoin(t.TempDir(), "empty.safetensors"), nil)

	if err == nil {
		t.Fatal("WriteSubset(nil) error = nil")
	}
}

func TestWriteSubset_UglyContextCancelled(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "cancelled.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = WriteSubset(ctx, target, []TensorRef{index.Tensors["x"]})

	if err == nil {
		t.Fatal("WriteSubset(cancelled) error = nil")
	}
}

func assertRawTensorEqual(t *testing.T, want, got TensorRef) {
	t.Helper()
	wantRaw, err := ReadRefRaw(want)
	if err != nil {
		t.Fatalf("ReadRefRaw(want): %v", err)
	}
	gotRaw, err := ReadRefRaw(got)
	if err != nil {
		t.Fatalf("ReadRefRaw(got): %v", err)
	}
	if string(wantRaw) != string(gotRaw) {
		t.Fatalf("raw tensor mismatch: want %v got %v", wantRaw, gotRaw)
	}
}

// TestSubsetHeaderEncoded_ParityWithJSONMarshal anchors the hand-rolled
// JSON encoder against the reflection-driven core.JSONMarshal form. The
// W10-R refactor of subsetHeader → subsetHeaderEncoded swapped a
// map[string]HeaderEntry + JSONMarshal pipeline for a single byte
// append. This test fixes that "bit-exact" claim — any structural drift
// (key order, integer width, dtype canonicalisation, string escapes)
// would break model-extract round-trips and pack-time golden files.
func TestSubsetHeaderEncoded_ParityWithJSONMarshal(t *testing.T) {
	cases := []struct {
		name string
		refs []TensorRef
	}{
		{
			name: "single_2d_f32",
			refs: []TensorRef{
				{Name: "weight", DType: "F32", Shape: []uint64{2048, 2048}, ByteLen: 2048 * 2048 * 4},
			},
		},
		{
			name: "multi_dim_mix",
			refs: []TensorRef{
				{Name: "model.layers.0.self_attn.q_proj.weight", DType: "F16", Shape: []uint64{4, 28, 2048, 64}, ByteLen: 4 * 28 * 2048 * 64 * 2},
				{Name: "model.layers.0.self_attn.k_proj.weight", DType: "BF16", Shape: []uint64{4, 28, 2048, 64}, ByteLen: 4 * 28 * 2048 * 64 * 2},
				{Name: "alpha", DType: "U8", Shape: []uint64{16}, ByteLen: 16},
			},
		},
		{
			name: "lowercase_dtype_canonicalised",
			refs: []TensorRef{
				{Name: "x", DType: "f32", Shape: []uint64{4}, ByteLen: 16},
			},
		},
		{
			name: "single_one_dim",
			refs: []TensorRef{
				{Name: "bias", DType: "F32", Shape: []uint64{128}, ByteLen: 512},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, got, err := subsetHeaderEncoded(tc.refs)
			if err != nil {
				t.Fatalf("subsetHeaderEncoded: %v", err)
			}
			// Reference: build the same map[string]HeaderEntry the old
			// subsetHeader produced, then JSONMarshal it.
			byName := map[string]TensorRef{}
			names := make([]string, 0, len(tc.refs))
			for _, ref := range tc.refs {
				byName[ref.Name] = ref
				names = append(names, ref.Name)
			}
			core.SliceSort(names)
			header := make(map[string]HeaderEntry, len(names))
			var offset int64
			for _, name := range names {
				ref := byName[name]
				shape := make([]int64, len(ref.Shape))
				for i, d := range ref.Shape {
					shape[i] = int64(d)
				}
				header[name] = HeaderEntry{
					DType:       core.Upper(ref.DType),
					Shape:       shape,
					DataOffsets: []int64{offset, offset + ref.ByteLen},
				}
				offset += ref.ByteLen
			}
			encoded := core.JSONMarshal(header)
			if !encoded.OK {
				t.Fatalf("JSONMarshal reference: %v", encoded.Value)
			}
			want := encoded.Value.([]byte)
			if string(got) != string(want) {
				t.Fatalf("encoder drift:\n got=%s\nwant=%s", got, want)
			}
		})
	}
}

// --- DTypeByteSize ---

func TestSafetensors_DTypeByteSize_Good(t *testing.T) {
	cases := []struct {
		dtype string
		want  int
	}{
		// Canonical fast-path matches.
		{"F16", 2}, {"BF16", 2}, {"F32", 4}, {"F64", 8},
		// Lowercase / mixed-case branch (the non-canonical path that
		// avoids core.Upper).
		{"f16", 2}, {"bf16", 2}, {"f32", 4}, {"f64", 8}, {"Bf16", 2},
	}
	for _, tc := range cases {
		got, err := DTypeByteSize(tc.dtype)
		if err != nil {
			t.Errorf("DTypeByteSize(%q) error = %v", tc.dtype, err)
			continue
		}
		if got != tc.want {
			t.Errorf("DTypeByteSize(%q) = %d, want %d", tc.dtype, got, tc.want)
		}
	}
}

func TestSafetensors_DTypeByteSize_Bad(t *testing.T) {
	// Unsupported but plausible dtypes the dense decoder cannot handle.
	for _, dtype := range []string{"I8", "U8", "I32", "C64", "FP8"} {
		if _, err := DTypeByteSize(dtype); err == nil {
			t.Errorf("DTypeByteSize(%q) error = nil, want unsupported", dtype)
		}
	}
}

func TestSafetensors_DTypeByteSize_Ugly(t *testing.T) {
	// Empty string and length-edge inputs that fall through every
	// length-branched byte compare without matching.
	for _, dtype := range []string{"", "F", "F1", "F128", "BF8", "XXXX"} {
		if _, err := DTypeByteSize(dtype); err == nil {
			t.Errorf("DTypeByteSize(%q) error = nil, want unsupported", dtype)
		}
	}
}

// --- DecodeFloatData (in-memory, per flag #1: dtype matrix needs no file
// and only F32 round-trips bit-exact for arbitrary values; F16/BF16/F64
// are checked against known bit patterns / the Float16ToFloat32 oracle) ---

func TestSafetensors_DecodeFloatData_Good(t *testing.T) {
	t.Run("F32_bit_exact", func(t *testing.T) {
		want := []float32{0, 1, -1, 3.14159, math.MaxFloat32, -math.SmallestNonzeroFloat32}
		raw := make([]byte, len(want)*4)
		for i, v := range want {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		got, err := DecodeFloatData("F32", raw, len(want))
		if err != nil {
			t.Fatalf("DecodeFloatData(F32): %v", err)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Errorf("F32[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})
	t.Run("F16_oracle", func(t *testing.T) {
		// 0x3c00=1.0, 0xc000=-2.0, 0x0000=+0.0, 0x7c00=+Inf.
		bits := []uint16{0x3c00, 0xc000, 0x0000, 0x7c00}
		raw := make([]byte, len(bits)*2)
		for i, b := range bits {
			binary.LittleEndian.PutUint16(raw[i*2:], b)
		}
		got, err := DecodeFloatData("F16", raw, len(bits))
		if err != nil {
			t.Fatalf("DecodeFloatData(F16): %v", err)
		}
		for i, b := range bits {
			want := Float16ToFloat32(b)
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Errorf("F16[%d]=0x%04x decoded %v, oracle %v", i, b, got[i], want)
			}
		}
	})
	t.Run("BF16_high_half", func(t *testing.T) {
		// BF16 is the high 16 bits of an IEEE-754 float32. 0x3f80=1.0,
		// 0xc000=-2.0, 0x4049≈3.140625.
		bits := []uint16{0x3f80, 0xc000, 0x4049}
		raw := make([]byte, len(bits)*2)
		for i, b := range bits {
			binary.LittleEndian.PutUint16(raw[i*2:], b)
		}
		got, err := DecodeFloatData("BF16", raw, len(bits))
		if err != nil {
			t.Fatalf("DecodeFloatData(BF16): %v", err)
		}
		for i, b := range bits {
			want := math.Float32frombits(uint32(b) << 16)
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Errorf("BF16[%d]=0x%04x decoded %v, want %v", i, b, got[i], want)
			}
		}
	})
	t.Run("F64_downcast", func(t *testing.T) {
		src := []float64{1.0, -2.5, 1e300}
		raw := make([]byte, len(src)*8)
		for i, v := range src {
			binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
		}
		got, err := DecodeFloatData("F64", raw, len(src))
		if err != nil {
			t.Fatalf("DecodeFloatData(F64): %v", err)
		}
		for i, v := range src {
			if math.Float32bits(got[i]) != math.Float32bits(float32(v)) {
				t.Errorf("F64[%d] = %v, want %v", i, got[i], float32(v))
			}
		}
	})
}

func TestSafetensors_DecodeFloatData_Bad(t *testing.T) {
	// Each dtype validates payload length == elements * width; a short
	// payload must surface the typed sentinel for that dtype.
	cases := []struct {
		dtype    string
		raw      []byte
		elements int
		sentinel error
	}{
		{"F32", make([]byte, 7), 2, errF32PayloadMismatch},   // want 8
		{"F16", make([]byte, 3), 2, errF16PayloadMismatch},   // want 4
		{"BF16", make([]byte, 3), 2, errBF16PayloadMatch},    // want 4
		{"F64", make([]byte, 15), 2, errF64PayloadMismatch},  // want 16
	}
	for _, tc := range cases {
		_, err := DecodeFloatData(tc.dtype, tc.raw, tc.elements)
		if !errors.Is(err, tc.sentinel) {
			t.Errorf("DecodeFloatData(%s, len=%d, n=%d) err = %v, want %v",
				tc.dtype, len(tc.raw), tc.elements, err, tc.sentinel)
		}
	}
	// Unsupported dtype hits the default branch.
	if _, err := DecodeFloatData("I32", []byte{0, 0, 0, 0}, 1); err == nil {
		t.Error("DecodeFloatData(I32) error = nil, want unsupported")
	}
}

func TestSafetensors_DecodeFloatData_Ugly(t *testing.T) {
	// Boundary element counts. elements=0 yields an empty slice with no
	// error (zero-length payload matches). elements=1 is the smallest
	// real decode. Negative element counts never occur in real refs
	// (Elements is derived from positive shape dims) and would panic on
	// the scratch[:elements] slice, so they are out of contract.
	t.Run("empty", func(t *testing.T) {
		got, err := DecodeFloatData("F32", nil, 0)
		if err != nil {
			t.Fatalf("DecodeFloatData(F32, n=0): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("DecodeFloatData(F32, n=0) len = %d, want 0", len(got))
		}
	})
	t.Run("single", func(t *testing.T) {
		raw := make([]byte, 4)
		binary.LittleEndian.PutUint32(raw, math.Float32bits(42.0))
		got, err := DecodeFloatData("F32", raw, 1)
		if err != nil {
			t.Fatalf("DecodeFloatData(F32, n=1): %v", err)
		}
		if len(got) != 1 || got[0] != 42.0 {
			t.Fatalf("DecodeFloatData(F32, n=1) = %v, want [42]", got)
		}
	})
}

// --- RefFromHeader ---

func TestSafetensors_RefFromHeader_Good(t *testing.T) {
	cases := []struct {
		name      string
		entry     HeaderEntry
		dataStart int64
		wantElems int
		wantStart int64
		wantLen   int64
	}{
		{
			name:      "2d",
			entry:     HeaderEntry{DType: "f32", Shape: []int64{2, 3}, DataOffsets: []int64{0, 24}},
			dataStart: 100,
			wantElems: 6,
			wantStart: 100,
			wantLen:   24,
		},
		{
			name:      "4d_offset",
			entry:     HeaderEntry{DType: "F16", Shape: []int64{2, 2, 2, 2}, DataOffsets: []int64{32, 64}},
			dataStart: 8,
			wantElems: 16,
			wantStart: 40, // dataStart + begin
			wantLen:   32,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref, err := RefFromHeader("model.safetensors", tc.name, tc.entry, tc.dataStart)
			if err != nil {
				t.Fatalf("RefFromHeader: %v", err)
			}
			if ref.Elements != tc.wantElems {
				t.Errorf("Elements = %d, want %d", ref.Elements, tc.wantElems)
			}
			if ref.DataStart != tc.wantStart {
				t.Errorf("DataStart = %d, want %d", ref.DataStart, tc.wantStart)
			}
			if ref.ByteLen != tc.wantLen {
				t.Errorf("ByteLen = %d, want %d", ref.ByteLen, tc.wantLen)
			}
			// DType is normalised to upper-case.
			if ref.DType != core.Upper(tc.entry.DType) {
				t.Errorf("DType = %q, want %q", ref.DType, core.Upper(tc.entry.DType))
			}
		})
	}
}

func TestSafetensors_RefFromHeader_Bad(t *testing.T) {
	cases := []struct {
		name  string
		entry HeaderEntry
	}{
		{"one_offset", HeaderEntry{DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0}}},
		{"three_offsets", HeaderEntry{DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 4, 8}}},
		{"negative_begin", HeaderEntry{DType: "F32", Shape: []int64{4}, DataOffsets: []int64{-1, 16}}},
		{"end_before_begin", HeaderEntry{DType: "F32", Shape: []int64{4}, DataOffsets: []int64{16, 8}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := RefFromHeader("p", tc.name, tc.entry, 0); err == nil {
				t.Errorf("RefFromHeader(%s) error = nil, want invalid offsets", tc.name)
			}
		})
	}
}

func TestSafetensors_RefFromHeader_Ugly(t *testing.T) {
	// Non-positive shape dimensions are rejected — a zero or negative dim
	// has no valid element count.
	cases := []struct {
		name  string
		shape []int64
	}{
		{"zero_dim", []int64{2, 0, 3}},
		{"negative_dim", []int64{-1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			entry := HeaderEntry{DType: "F32", Shape: tc.shape, DataOffsets: []int64{0, 16}}
			if _, err := RefFromHeader("p", tc.name, entry, 0); err == nil {
				t.Errorf("RefFromHeader(%v) error = nil, want invalid shape", tc.shape)
			}
		})
	}
}

// --- ReadRefRaw / ReadRefValues (file-based; F32-only per flag #1) ---

func TestSafetensors_ReadRefValues_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	want := []float32{1, 2, 3, 4, -5.5, 1024.25}
	writeF32Safetensors(t, path, map[string][]float32{"weight": want})

	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	got, err := ReadRefValues(index.Tensors["weight"])
	if err != nil {
		t.Fatalf("ReadRefValues: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("value[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSafetensors_ReadRefValues_Bad(t *testing.T) {
	// Ref pointing at a path that does not exist.
	ref := TensorRef{Name: "x", Path: core.PathJoin(t.TempDir(), "missing.safetensors"), DType: "F32", Elements: 1, ByteLen: 4}
	if _, err := ReadRefValues(ref); err == nil {
		t.Fatal("ReadRefValues(missing file) error = nil")
	}
}

func TestSafetensors_ReadRefValues_Ugly(t *testing.T) {
	// Single-element tensor — smallest non-empty payload.
	dir := t.TempDir()
	path := core.PathJoin(dir, "single.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"scalar": {3.5}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	got, err := ReadRefValues(index.Tensors["scalar"])
	if err != nil {
		t.Fatalf("ReadRefValues: %v", err)
	}
	if len(got) != 1 || got[0] != 3.5 {
		t.Fatalf("ReadRefValues = %v, want [3.5]", got)
	}
}

func TestSafetensors_ReadRefRaw_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1, 2}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	raw, err := ReadRefRaw(ref)
	if err != nil {
		t.Fatalf("ReadRefRaw: %v", err)
	}
	if int64(len(raw)) != ref.ByteLen {
		t.Fatalf("len(raw) = %d, want %d", len(raw), ref.ByteLen)
	}
	// First float32 little-endian should be 1.0.
	if got := math.Float32frombits(binary.LittleEndian.Uint32(raw[:4])); got != 1.0 {
		t.Fatalf("raw[0] = %v, want 1.0", got)
	}
}

func TestSafetensors_ReadRefRaw_Bad(t *testing.T) {
	// Negative ByteLen is rejected before any file open.
	ref := TensorRef{Name: "x", Path: "irrelevant", ByteLen: -1}
	if _, err := ReadRefRaw(ref); err == nil {
		t.Fatal("ReadRefRaw(negative ByteLen) error = nil")
	}
}

func TestSafetensors_ReadRefRaw_Ugly(t *testing.T) {
	// Ref claims more bytes than the file holds → truncated payload.
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1, 2}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	ref.ByteLen += 1024 // demand past EOF
	if _, err := ReadRefRaw(ref); err == nil {
		t.Fatal("ReadRefRaw(truncated) error = nil")
	}
}

// TestSafetensors_ReadIndex_Bad drives ReadIndex's structural failure
// paths with real files (not fault injection): a path that does not
// exist, a file too short to hold the 8-byte header length, and a file
// whose declared header length runs past EOF. Each is a corruption a
// real on-disk safetensors could carry.
func TestSafetensors_ReadIndex_Bad(t *testing.T) {
	dir := t.TempDir()

	t.Run("missing file", func(t *testing.T) {
		if _, err := ReadIndex(core.PathJoin(dir, "nope.safetensors")); err == nil {
			t.Fatal("ReadIndex(missing) error = nil")
		}
	})

	t.Run("shorter than header length", func(t *testing.T) {
		// Only 4 bytes — the 8-byte little-endian header length can't be
		// read in full.
		path := core.PathJoin(dir, "short.safetensors")
		if result := core.WriteFile(path, []byte{1, 2, 3, 4}, 0o644); !result.OK {
			t.Fatalf("WriteFile: %v", result.Value)
		}
		if _, err := ReadIndex(path); err == nil {
			t.Fatal("ReadIndex(short) error = nil")
		}
	})

	t.Run("header length past EOF", func(t *testing.T) {
		// Declared header length is 1000 but only a few header bytes
		// follow — the header read can't be satisfied.
		path := core.PathJoin(dir, "overclaim.safetensors")
		buf := make([]byte, 8+4)
		binary.LittleEndian.PutUint64(buf[:8], 1000)
		copy(buf[8:], []byte(`{"x"`))
		if result := core.WriteFile(path, buf, 0o644); !result.OK {
			t.Fatalf("WriteFile: %v", result.Value)
		}
		if _, err := ReadIndex(path); err == nil {
			t.Fatal("ReadIndex(overclaim) error = nil")
		}
	})
}

// --- ReadRefFloat32Chunk / TensorReader.ReadFloat32Chunk(Into) ---

func TestSafetensors_ReadRefFloat32Chunk_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	full := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	writeF32Safetensors(t, path, map[string][]float32{"weight": full})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]

	// A middle chunk must equal the corresponding slice of the full tensor.
	chunk, err := ReadRefFloat32Chunk(ref, 2, 3)
	if err != nil {
		t.Fatalf("ReadRefFloat32Chunk: %v", err)
	}
	want := full[2:5]
	if len(chunk) != len(want) {
		t.Fatalf("len = %d, want %d", len(chunk), len(want))
	}
	for i := range want {
		if chunk[i] != want[i] {
			t.Errorf("chunk[%d] = %v, want %v", i, chunk[i], want[i])
		}
	}
}

func TestSafetensors_ReadRefFloat32Chunk_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1, 2, 3, 4}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]

	// offset+count past the element bound → typed out-of-bounds sentinel.
	if _, err := ReadRefFloat32Chunk(ref, 2, 5); !errors.Is(err, errChunkOutOfBounds) {
		t.Fatalf("ReadRefFloat32Chunk(oob) err = %v, want errChunkOutOfBounds", err)
	}
	// Negative offset is also out of bounds.
	if _, err := ReadRefFloat32Chunk(ref, -1, 1); !errors.Is(err, errChunkOutOfBounds) {
		t.Fatalf("ReadRefFloat32Chunk(neg) err = %v, want errChunkOutOfBounds", err)
	}
}

func TestSafetensors_ReadRefFloat32Chunk_Ugly(t *testing.T) {
	// Zero-count chunk at a valid offset returns an empty slice, no error.
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1, 2, 3}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	got, err := ReadRefFloat32Chunk(index.Tensors["weight"], 1, 0)
	if err != nil {
		t.Fatalf("ReadRefFloat32Chunk(count=0): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("len = %d, want 0", len(got))
	}
}

func TestSafetensors_ReadFloat32ChunkInto_Good(t *testing.T) {
	// The scratch-aware variant must decode byte-identically to the
	// allocating ReadFloat32Chunk, and reuse the returned buffers across
	// calls (per the documented chunked-loop contract).
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	full := []float32{10, 11, 12, 13, 14, 15}
	writeF32Safetensors(t, path, map[string][]float32{"weight": full})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	reader, err := OpenReader(index.Tensors["weight"])
	if err != nil {
		t.Fatalf("OpenReader: %v", err)
	}
	defer reader.Close()

	var rawScratch []byte
	var valScratch []float32
	for offset := 0; offset < len(full); offset += 2 {
		rawScratch, valScratch, values, err := reader.ReadFloat32ChunkInto(offset, 2, rawScratch, valScratch)
		if err != nil {
			t.Fatalf("ReadFloat32ChunkInto(%d): %v", offset, err)
		}
		_ = rawScratch
		_ = valScratch
		want := full[offset : offset+2]
		for i := range want {
			if values[i] != want[i] {
				t.Errorf("offset %d values[%d] = %v, want %v", offset, i, values[i], want[i])
			}
		}
	}
}

// --- OpenReader / OpenReaders / NewFileReader / Close / CloseReaders ---

func TestSafetensors_OpenReader_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {7, 8, 9}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	reader, err := OpenReader(index.Tensors["weight"])
	if err != nil {
		t.Fatalf("OpenReader: %v", err)
	}
	defer reader.Close()
	got, err := reader.ReadFloat32Chunk(0, 3)
	if err != nil {
		t.Fatalf("ReadFloat32Chunk: %v", err)
	}
	if len(got) != 3 || got[0] != 7 || got[2] != 9 {
		t.Fatalf("ReadFloat32Chunk = %v, want [7 8 9]", got)
	}
}

func TestSafetensors_OpenReader_Bad(t *testing.T) {
	t.Run("unsupported_dtype", func(t *testing.T) {
		ref := TensorRef{Name: "x", Path: "irrelevant", DType: "I32", Elements: 1, ByteLen: 4}
		if _, err := OpenReader(ref); err == nil {
			t.Fatal("OpenReader(unsupported dtype) error = nil")
		}
	})
	t.Run("missing_file", func(t *testing.T) {
		ref := TensorRef{Name: "x", Path: core.PathJoin(t.TempDir(), "missing.safetensors"), DType: "F32", Elements: 1, ByteLen: 4}
		if _, err := OpenReader(ref); err == nil {
			t.Fatal("OpenReader(missing file) error = nil")
		}
	})
}

func TestSafetensors_OpenReaders_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{
		"a": {1, 2},
		"b": {3, 4, 5},
	})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	refs := []TensorRef{index.Tensors["a"], index.Tensors["b"]}
	readers, err := OpenReaders(refs)
	if err != nil {
		t.Fatalf("OpenReaders: %v", err)
	}
	defer CloseReaders(readers)
	if len(readers) != 2 {
		t.Fatalf("len(readers) = %d, want 2", len(readers))
	}
	vals, err := readers[1].ReadFloat32Chunk(0, 3)
	if err != nil {
		t.Fatalf("ReadFloat32Chunk: %v", err)
	}
	if len(vals) != 3 || vals[2] != 5 {
		t.Fatalf("ReadFloat32Chunk = %v, want [3 4 5]", vals)
	}
}

func TestSafetensors_OpenReaders_Bad(t *testing.T) {
	// A bad ref mid-list must close everything already opened and error.
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"good": {1, 2}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	bad := TensorRef{Name: "bad", Path: path, DType: "I32", Elements: 1, ByteLen: 4}
	if _, err := OpenReaders([]TensorRef{index.Tensors["good"], bad}); err == nil {
		t.Fatal("OpenReaders(with bad ref) error = nil")
	}
}

// TestSafetensors_NewFileReader_Good exercises the documented borrow
// pattern: one open *core.OSFile is bound to a ref via NewFileReader and
// the caller owns the handle's lifetime. NOTE — the doc comment says the
// reader "does NOT own the file", yet TensorReader.Close() unconditionally
// closes r.file. This is the deliberate one-type-for-owned-and-borrowed
// shape; the documented usage (caller closes the handle once, does not
// call reader.Close) is what this test drives.
func TestSafetensors_NewFileReader_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {2, 4, 6, 8}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]

	opened := core.Open(path)
	if !opened.OK {
		t.Fatalf("core.Open: %v", opened.Value)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close() // caller owns the handle

	reader, err := NewFileReader(file, ref)
	if err != nil {
		t.Fatalf("NewFileReader: %v", err)
	}
	got, err := reader.ReadFloat32Chunk(1, 2)
	if err != nil {
		t.Fatalf("ReadFloat32Chunk: %v", err)
	}
	if len(got) != 2 || got[0] != 4 || got[1] != 6 {
		t.Fatalf("ReadFloat32Chunk = %v, want [4 6]", got)
	}
}

func TestSafetensors_NewFileReader_Bad(t *testing.T) {
	// Unsupported dtype is rejected before binding the handle.
	dir := t.TempDir()
	path := core.PathJoin(dir, "dense.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1}})
	opened := core.Open(path)
	if !opened.OK {
		t.Fatalf("core.Open: %v", opened.Value)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()
	if _, err := NewFileReader(file, TensorRef{Name: "x", DType: "I32", Elements: 1, ByteLen: 4}); err == nil {
		t.Fatal("NewFileReader(unsupported dtype) error = nil")
	}
}

// --- IndexFiles ---

func TestSafetensors_IndexFiles_Good(t *testing.T) {
	dir := t.TempDir()
	p1 := core.PathJoin(dir, "shard-1.safetensors")
	p2 := core.PathJoin(dir, "shard-2.safetensors")
	writeRawSafetensors(t, p1, map[string][]byte{"a": {1}, "b": {2}})
	writeRawSafetensors(t, p2, map[string][]byte{"c": {3}, "d": {4}})

	index, err := IndexFiles([]string{p1, p2})
	if err != nil {
		t.Fatalf("IndexFiles: %v", err)
	}
	if len(index.Names) != 4 {
		t.Fatalf("Names = %v, want 4 tensors", index.Names)
	}
	// Names are sorted across the merged shards.
	for i := 1; i < len(index.Names); i++ {
		if index.Names[i-1] > index.Names[i] {
			t.Fatalf("Names not sorted: %v", index.Names)
		}
	}
	// Each tensor resolves to its source shard.
	if index.Tensors["a"].Path != p1 || index.Tensors["c"].Path != p2 {
		t.Fatalf("tensor paths not preserved: a=%s c=%s", index.Tensors["a"].Path, index.Tensors["c"].Path)
	}
}

func TestSafetensors_IndexFiles_Bad(t *testing.T) {
	// A tensor name present in two shards is a duplicate and must error.
	dir := t.TempDir()
	p1 := core.PathJoin(dir, "shard-1.safetensors")
	p2 := core.PathJoin(dir, "shard-2.safetensors")
	writeRawSafetensors(t, p1, map[string][]byte{"dup": {1}})
	writeRawSafetensors(t, p2, map[string][]byte{"dup": {2}})
	if _, err := IndexFiles([]string{p1, p2}); err == nil {
		t.Fatal("IndexFiles(duplicate tensor) error = nil")
	}
}

func TestSafetensors_IndexFiles_Ugly(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		// No paths → empty, non-nil index, no error.
		index, err := IndexFiles(nil)
		if err != nil {
			t.Fatalf("IndexFiles(nil): %v", err)
		}
		if index.Tensors == nil || len(index.Names) != 0 {
			t.Fatalf("IndexFiles(nil) = %+v, want empty index", index)
		}
	})
	t.Run("single_path", func(t *testing.T) {
		// One path takes the single-shard fast path (Path cleared).
		dir := t.TempDir()
		p := core.PathJoin(dir, "one.safetensors")
		writeRawSafetensors(t, p, map[string][]byte{"x": {1}, "y": {2}})
		index, err := IndexFiles([]string{p})
		if err != nil {
			t.Fatalf("IndexFiles(single): %v", err)
		}
		if len(index.Names) != 2 {
			t.Fatalf("Names = %v, want 2", index.Names)
		}
	})
}

// --- WriteRefFloat32Chunks (per flag #2: writes headerless raw float32-LE,
// NOT a safetensors container — verify by reading the raw bytes back) ---

func TestSafetensors_WriteRefFloat32Chunks_Good(t *testing.T) {
	dir := t.TempDir()
	src := core.PathJoin(dir, "src.safetensors")
	want := []float32{1, 2, 3, 4, 5, 6, 7}
	writeF32Safetensors(t, src, map[string][]float32{"weight": want})
	index, err := ReadIndex(src)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]

	dst := core.PathJoin(dir, "raw.f32")
	created := core.OpenFile(dst, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile(dst): %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	// chunkElements=2 forces multiple chunked writes over the 7 elements.
	if err := WriteRefFloat32Chunks(context.Background(), out, ref, 2); err != nil {
		out.Close()
		t.Fatalf("WriteRefFloat32Chunks: %v", err)
	}
	out.Close()

	// The destination is a bare float32-LE blob, no 8-byte header.
	read := core.ReadFile(dst)
	if !read.OK {
		t.Fatalf("ReadFile(dst): %v", read.Value)
	}
	raw := read.Value.([]byte)
	if len(raw) != len(want)*4 {
		t.Fatalf("dst len = %d, want %d", len(raw), len(want)*4)
	}
	for i, w := range want {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if got != w {
			t.Errorf("raw[%d] = %v, want %v", i, got, w)
		}
	}
}

func TestSafetensors_WriteRefFloat32Chunks_Bad(t *testing.T) {
	// A cancelled context aborts before/within the chunk loop.
	dir := t.TempDir()
	src := core.PathJoin(dir, "src.safetensors")
	writeF32Safetensors(t, src, map[string][]float32{"weight": {1, 2, 3, 4}})
	index, err := ReadIndex(src)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	dst := core.PathJoin(dir, "raw.f32")
	created := core.OpenFile(dst, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile(dst): %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	defer out.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := WriteRefFloat32Chunks(ctx, out, index.Tensors["weight"], 1); err == nil {
		t.Fatal("WriteRefFloat32Chunks(cancelled) error = nil")
	}
}

// --- WriteSubset escape path (bonus flag #6: a tensor name with a quote +
// newline drives appendJSONString's escape branch, hexNibble, and the
// header parser's parseStringEscaped through one real round-trip) ---

func TestSafetensors_WriteSubset_UglyEscapedName(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "escaped.safetensors")
	// Name contains a double-quote, backslash, newline and a low control
	// byte (0x01 →  via hexNibble).
	weird := "weird\"name\\with\nctrl\x01"
	writeRawSafetensors(t, source, map[string][]byte{weird: {9, 8, 7, 6}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex(source): %v", err)
	}

	if err := WriteSubset(context.Background(), target, []TensorRef{index.Tensors[weird]}); err != nil {
		t.Fatalf("WriteSubset: %v", err)
	}
	got, err := ReadIndex(target)
	if err != nil {
		t.Fatalf("ReadIndex(target): %v", err)
	}
	if _, ok := got.Tensors[weird]; !ok {
		t.Fatalf("escaped name not round-tripped; got names = %v", got.Names)
	}
	assertRawTensorEqual(t, index.Tensors[weird], got.Tensors[weird])
}

// --- Residual coverage: thin non-scratch wrappers + resultError ---
//
// refFromHeaderSlab, writeFloat32Values and writeRefRawChunks are the
// allocate-fresh entry points that the *Scratch / inline-emit variants
// superseded on the alloc-reduction passes; they retain no production
// caller (the merge + read paths all use the scratch forms). They remain
// as the documented standalone form, so these white-box tests exercise
// their contract directly rather than leaving them dark. resultError's
// non-error Value branch is likewise only reachable by handing it a
// failed Result whose Value is not an error.

func TestSafetensors_RefFromHeaderSlab_Good(t *testing.T) {
	// refFromHeaderSlab carves the Shape out of a shared slab. Pre-size
	// the slab so the slice is taken in-place (the production contract:
	// callers guarantee capacity from the prior header scan).
	slab := make([]uint64, 0, 2)
	entry := HeaderEntry{DType: "f32", Shape: []int64{2, 3}, DataOffsets: []int64{0, 24}}
	ref, err := refFromHeaderSlab("model.safetensors", "weight", entry, 128, &slab)
	if err != nil {
		t.Fatalf("refFromHeaderSlab: %v", err)
	}
	// DType is canonicalised to upper-case; element count is the shape
	// product; DataStart folds in the slab-relative begin offset.
	if ref.DType != "F32" {
		t.Errorf("DType = %q, want F32", ref.DType)
	}
	if ref.Elements != 6 {
		t.Errorf("Elements = %d, want 6", ref.Elements)
	}
	if ref.DataStart != 128 || ref.ByteLen != 24 {
		t.Errorf("DataStart/ByteLen = %d/%d, want 128/24", ref.DataStart, ref.ByteLen)
	}
	if len(ref.Shape) != 2 || ref.Shape[0] != 2 || ref.Shape[1] != 3 {
		t.Errorf("Shape = %v, want [2 3]", ref.Shape)
	}
	// The slab grew to hold both dims.
	if len(slab) != 2 {
		t.Errorf("slab len = %d, want 2", len(slab))
	}
}

func TestSafetensors_RefFromHeaderSlab_Bad(t *testing.T) {
	slab := make([]uint64, 0, 4)
	cases := map[string]HeaderEntry{
		"wrong offsets count": {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{0}},
		"negative begin":      {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{-1, 4}},
		"end before begin":    {DType: "F32", Shape: []int64{1}, DataOffsets: []int64{8, 4}},
		"zero dim":            {DType: "F32", Shape: []int64{0}, DataOffsets: []int64{0, 0}},
		"negative dim":        {DType: "F32", Shape: []int64{-2}, DataOffsets: []int64{0, 8}},
	}
	for name, entry := range cases {
		if _, err := refFromHeaderSlab("p", "t", entry, 0, &slab); err == nil {
			t.Errorf("%s: error = nil, want non-nil", name)
		}
	}
}

func TestSafetensors_WriteFloat32Values_Good(t *testing.T) {
	// writeFloat32Values is the nil-scratch wrapper over the scratch form.
	// Write a known slice, read the file back, assert bit-exact LE bytes.
	dir := t.TempDir()
	path := core.PathJoin(dir, "vals.f32")
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile: %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	want := []float32{1.5, -2, 3.25}
	if err := writeFloat32Values(out, want); err != nil {
		out.Close()
		t.Fatalf("writeFloat32Values: %v", err)
	}
	out.Close()

	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile: %v", read.Value)
	}
	raw := read.Value.([]byte)
	if len(raw) != len(want)*4 {
		t.Fatalf("len = %d, want %d", len(raw), len(want)*4)
	}
	for i, w := range want {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.Float32bits(got) != math.Float32bits(w) {
			t.Errorf("value[%d] = %v, want %v", i, got, w)
		}
	}
}

func TestSafetensors_WriteRefRawChunks_Good(t *testing.T) {
	// writeRefRawChunks is the nil-scratch wrapper WriteSubset bypasses in
	// favour of the scratch form. Drive it directly: copy a tensor's raw
	// payload into a fresh file and confirm the bytes match the source.
	dir := t.TempDir()
	src := core.PathJoin(dir, "src.safetensors")
	writeRawSafetensors(t, src, map[string][]byte{"blob": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}})
	index, err := ReadIndex(src)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["blob"]

	dst := core.PathJoin(dir, "blob.raw")
	created := core.OpenFile(dst, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile(dst): %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	// chunkBytes=3 forces several chunked reads over the 10-byte payload.
	if err := writeRefRawChunks(context.Background(), out, ref, 3); err != nil {
		out.Close()
		t.Fatalf("writeRefRawChunks: %v", err)
	}
	out.Close()

	read := core.ReadFile(dst)
	if !read.OK {
		t.Fatalf("ReadFile(dst): %v", read.Value)
	}
	if got := read.Value.([]byte); len(got) != 10 || got[0] != 1 || got[9] != 10 {
		t.Fatalf("dst bytes = %v, want 1..10", got)
	}
}

func TestSafetensors_WriteRefRawChunks_Bad(t *testing.T) {
	// A cancelled context is observed before the first chunk copy.
	dir := t.TempDir()
	src := core.PathJoin(dir, "src.safetensors")
	writeRawSafetensors(t, src, map[string][]byte{"blob": {1, 2, 3, 4}})
	index, err := ReadIndex(src)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	dst := core.PathJoin(dir, "blob.raw")
	created := core.OpenFile(dst, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile(dst): %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	defer out.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := writeRefRawChunks(ctx, out, index.Tensors["blob"], 1); err == nil {
		t.Fatal("writeRefRawChunks(cancelled) error = nil")
	}
}

func TestSafetensors_ResultError(t *testing.T) {
	// OK result → nil error.
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Errorf("resultError(OK) = %v, want nil", err)
	}
	// Failed result carrying an error Value → that error is returned.
	sentinel := core.NewError("boom")
	if err := resultError(core.Result{OK: false, Value: sentinel}); err != sentinel {
		t.Errorf("resultError(err Value) = %v, want the sentinel", err)
	}
	// Failed result whose Value is NOT an error → the generic fallback.
	if err := resultError(core.Result{OK: false, Value: "not an error"}); err != errCoreResultFailed {
		t.Errorf("resultError(non-error Value) = %v, want errCoreResultFailed", err)
	}
}

// writeF32Safetensors lays down a real safetensors file whose tensors carry
// F32 payloads (little-endian float32). Unlike writeRawSafetensors (which
// tags every tensor U8 for header-shape tests), this produces decodable
// dense tensors so the float read paths can be round-tripped bit-exactly
// — F32 is the only dtype that survives write→read without representation
// loss (see DecodeFloatData flag #1).
func writeF32Safetensors(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		vals := tensors[name]
		raw := make([]byte, len(vals)*4)
		for i, v := range vals {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		header[name] = HeaderEntry{
			DType:       "F32",
			Shape:       []int64{int64(len(vals))},
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

func writeRawSafetensors(t *testing.T, path string, tensors map[string][]byte) {
	t.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	core.SliceSort(names)
	var offset int64
	payload := []byte{}
	for _, name := range names {
		raw := tensors[name]
		header[name] = HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(len(raw))},
			DataOffsets: []int64{offset, offset + int64(len(raw))},
		}
		payload = append(payload, raw...)
		offset += int64(len(raw))
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}
