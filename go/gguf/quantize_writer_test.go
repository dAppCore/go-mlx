// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

func TestGGUFQuantize_WriteStreamedGGUF_Good(t *testing.T) {
	source := core.PathJoin(t.TempDir(), "source.safetensors")
	writeTestSafetensorsF32(t, source, []safetensorTestTensor{
		{Name: "model.layers.0.self_attn.k_proj.weight", Shape: []int{32, 2}, Data: ascendingFloat32s(64)},
	})
	index, err := safetensors.IndexFiles([]string{source})
	if err != nil {
		t.Fatalf("index safetensors: %v", err)
	}
	tensors, refs, err := buildStreamingGGUFQuantizedTensors(index, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("build streaming tensors: %v", err)
	}
	if len(tensors) != 1 || len(refs) != 1 {
		t.Fatalf("stream tensor counts = %d/%d, want 1/1", len(tensors), len(refs))
	}

	output := core.PathJoin(t.TempDir(), "streamed.gguf")
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	if err := writeQuantizedGGUFStream(context.Background(), output, metadata, tensors, refs, QuantizeQ8_0, 32); err != nil {
		t.Fatalf("writeQuantizedGGUFStream() error = %v", err)
	}

	info, err := ReadInfo(output)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
	}
	if !info.Valid() || info.TensorCount != 1 || info.Tensors[0].TypeName != "q8_0" {
		t.Fatalf("streamed info = %+v", info)
	}
}

func TestGGUFQuantize_WriteBufferedGGUF_Good(t *testing.T) {
	output := core.PathJoin(t.TempDir(), "buffered.gguf")
	values := ascendingFloat32s(32)
	data := quantizeQ8_0(values)
	tensors := []ggufQuantizedTensor{{
		Name:  "model.norm.weight",
		Type:  TensorTypeQ8_0,
		Shape: []uint64{32},
		Data:  data,
	}}
	metadata := ggufQuantizeMetadata(mp.ModelPack{Architecture: "qwen3"}, QuantizeQ8_0, nil)
	if err := writeQuantizedGGUF(output, metadata, tensors); err != nil {
		t.Fatalf("writeQuantizedGGUF() error = %v", err)
	}
	info, err := ReadInfo(output)
	if err != nil {
		t.Fatalf("ReadInfo() error = %v", err)
	}
	if !info.Valid() || info.TensorCount != 1 || info.Tensors[0].TypeName != "q8_0" {
		t.Fatalf("buffered info = %+v", info)
	}
	if got := ggufQuantizedTensorDataSize(ggufQuantizedTensor{Size: 12, Data: data}); got != 12 {
		t.Fatalf("ggufQuantizedTensorDataSize(Size) = %d, want 12", got)
	}
}

func TestGGUFQuantize_StreamErrorPaths_Bad(t *testing.T) {
	if _, _, err := buildStreamingGGUFQuantizedTensors(safetensors.Index{
		Names: []string{"bad.weight"},
		Tensors: map[string]safetensors.TensorRef{
			"bad.weight": {Name: "bad.weight", DType: "I32", Shape: []uint64{32}, Elements: 32},
		},
	}, QuantizeQ8_0); err == nil {
		t.Fatal("expected unsupported dtype error")
	}
	if _, _, err := buildStreamingGGUFQuantizedTensors(safetensors.Index{
		Names: []string{"bad.weight"},
		Tensors: map[string]safetensors.TensorRef{
			"bad.weight": {Name: "bad.weight", DType: "F32", Shape: []uint64{32}, Elements: 31},
		},
	}, QuantizeQ8_0); err == nil {
		t.Fatal("expected block alignment error")
	}
	if err := writeQuantizedGGUFStream(context.Background(), core.PathJoin(t.TempDir(), "bad.gguf"), nil, []ggufQuantizedTensor{{}}, nil, QuantizeQ8_0, 32); err == nil {
		t.Fatal("expected tensor/ref alignment error")
	}
}

func TestGGUFQuantizeMetadata_LabelsAndDenseFloats_Ugly(t *testing.T) {
	source := mp.ModelPack{Architecture: "qwen3", VocabSize: 10, HiddenSize: 20, NumLayers: 2, ContextLength: 128}
	metadata := ggufQuantizeMetadata(source, QuantizeQ4_0, map[string]string{"z": "last", "a": "first"})
	if len(metadata) != 11 {
		t.Fatalf("metadata entries = %d, want 11", len(metadata))
	}
	if metadata[len(metadata)-2].Key != "go_mlx.label.a" || metadata[len(metadata)-1].Key != "go_mlx.label.z" {
		t.Fatalf("labels were not sorted: %+v", metadata[len(metadata)-2:])
	}

	floatCases := []float32{0, 1, -2, float32(math.Inf(1)), float32(math.NaN())}
	for _, value := range floatCases {
		half := float32ToFloat16(value)
		roundTrip := safetensors.Float16ToFloat32(half)
		if math.IsNaN(float64(value)) {
			if !math.IsNaN(float64(roundTrip)) {
				t.Fatalf("NaN roundtrip = %v", roundTrip)
			}
			continue
		}
		if math.IsInf(float64(value), 0) {
			if !math.IsInf(float64(roundTrip), 0) {
				t.Fatalf("Inf roundtrip = %v", roundTrip)
			}
			continue
		}
		if value != 0 && roundTrip == 0 {
			t.Fatalf("float16 roundtrip of %v underflowed unexpectedly", value)
		}
	}
}

// TestWriteGGUFMetadataValue_Good covers both supported metadata write types
// and, crucially, the int->uint32 coercion arm that the streaming header writer
// relies on (alignment / counts are often plain ints at the call site). Bytes
// are read back and decoded to confirm the little-endian encoding.
func TestWriteGGUFMetadataValue_Good(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
			return writeGGUFMetadataValue(f, ValueTypeString, "gemma4")
		})
		// uint64 length prefix + bytes.
		wantLen := uint64(len("gemma4"))
		if binary.LittleEndian.Uint64(got[:8]) != wantLen || string(got[8:]) != "gemma4" {
			t.Fatalf("string value bytes = %v", got)
		}
	})
	t.Run("uint32_native", func(t *testing.T) {
		got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
			return writeGGUFMetadataValue(f, ValueTypeUint32, uint32(0xDEADBEEF))
		})
		if binary.LittleEndian.Uint32(got) != 0xDEADBEEF {
			t.Fatalf("uint32 value = %#x", binary.LittleEndian.Uint32(got))
		}
	})
	t.Run("uint32_from_int", func(t *testing.T) {
		// The int->uint32 arm: a plain int (e.g. alignment=32) is accepted.
		got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
			return writeGGUFMetadataValue(f, ValueTypeUint32, 32)
		})
		if binary.LittleEndian.Uint32(got) != 32 {
			t.Fatalf("uint32 from int = %d, want 32", binary.LittleEndian.Uint32(got))
		}
	})
}

// TestWriteGGUFMetadataValue_Bad covers the type-mismatch and unsupported-type
// error arms: a non-string for a string slot, a non-numeric for a uint32 slot,
// and a value-type the writer does not implement.
func TestWriteGGUFMetadataValue_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "bad.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create temp file: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	if err := writeGGUFMetadataValue(file, ValueTypeString, 123); err == nil {
		t.Fatal("string slot with int value: error = nil, want type error")
	}
	if err := writeGGUFMetadataValue(file, ValueTypeUint32, "not-a-number"); err == nil {
		t.Fatal("uint32 slot with string value: error = nil, want type error")
	}
	// ggufValueTypeFloat32 (6) has no writer arm — unsupported on the write side.
	if err := writeGGUFMetadataValue(file, ggufValueTypeFloat32, float32(1.5)); err == nil {
		t.Fatal("unsupported write type: error = nil, want unsupported-type error")
	}
}

// TestWriteGGUFMetadataEntry_Good covers the full entry writer (key + type tag
// + value) and asserts the three sections decode back in order.
func TestWriteGGUFMetadataEntry_Good(t *testing.T) {
	got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
		return writeGGUFMetadataEntry(f, ggufMetadataEntry{
			Key:       "general.architecture",
			ValueType: ValueTypeString,
			Value:     "gemma4",
		})
	})
	keyLen := binary.LittleEndian.Uint64(got[:8])
	key := string(got[8 : 8+keyLen])
	if key != "general.architecture" {
		t.Fatalf("entry key = %q", key)
	}
	rest := got[8+keyLen:]
	valueType := binary.LittleEndian.Uint32(rest[:4])
	if valueType != ValueTypeString {
		t.Fatalf("entry value type = %d, want %d", valueType, ValueTypeString)
	}
	valLen := binary.LittleEndian.Uint64(rest[4:12])
	if string(rest[12:12+valLen]) != "gemma4" {
		t.Fatalf("entry value = %q", string(rest[12:12+valLen]))
	}
}

// TestWriteGGUFStringValue_Ugly exercises both branches of the string writer:
// the small-string stacked single-write path AND the >248-byte heap fallback
// (a value too large for the 256-byte stack buffer). Both must produce an
// identical [uint64 length][bytes] layout.
func TestWriteGGUFStringValue_Ugly(t *testing.T) {
	cases := []struct {
		name   string
		length int
	}{
		{"small_stacked", 16},           // fits the 256-byte stack buffer
		{"boundary_just_fits", 256 - 8}, // exactly at the +8 length-prefix limit
		{"large_heap_fallback", 4096},   // forces the separate-write fallback
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := string(bytes.Repeat([]byte("x"), tc.length))
			got := writeGGUFToTempFile(t, func(f *core.OSFile) error {
				return writeGGUFStringValue(f, value)
			})
			if len(got) != 8+tc.length {
				t.Fatalf("total bytes = %d, want %d", len(got), 8+tc.length)
			}
			if binary.LittleEndian.Uint64(got[:8]) != uint64(tc.length) {
				t.Fatalf("length prefix = %d, want %d", binary.LittleEndian.Uint64(got[:8]), tc.length)
			}
			if string(got[8:]) != value {
				t.Fatal("value bytes mismatch")
			}
		})
	}
}

// TestWritePadding_Good covers the multi-chunk padding writer, including a
// length larger than the 32 KiB zero buffer (forcing the loop to iterate) and
// the n==0 fast exit.
func TestWritePadding_Good(t *testing.T) {
	t.Run("zero", func(t *testing.T) {
		got := writeGGUFToTempFile(t, func(f *core.OSFile) error { return writePadding(f, 0) })
		if len(got) != 0 {
			t.Fatalf("writePadding(0) wrote %d bytes, want 0", len(got))
		}
	})
	t.Run("multi_chunk", func(t *testing.T) {
		const n = 32*1024 + 17 // one full buffer chunk + a remainder
		got := writeGGUFToTempFile(t, func(f *core.OSFile) error { return writePadding(f, n) })
		if len(got) != n {
			t.Fatalf("writePadding(%d) wrote %d bytes", n, len(got))
		}
		for i, b := range got {
			if b != 0 {
				t.Fatalf("padding byte %d = %d, want 0", i, b)
			}
		}
	})
}
