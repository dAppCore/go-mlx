// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// exampleFs is the filesystem handle the examples use for temp dirs.
// Examples cannot take a *testing.T, so they cannot call t.TempDir();
// core's Fs.TempDir is the established example idiom (see core's own
// ExampleData_New).
var exampleFs = (&core.Fs{}).New("/")

// mkTempDir returns a fresh temp directory for an example fixture. The
// second return is the cleanup closure callers defer.
func mkTempDir() (string, func()) {
	dir := exampleFs.TempDir("safetensors-example")
	return dir, func() { exampleFs.DeleteAll(dir) }
}

// buildSafetensors assembles a complete safetensors file on disk from a
// raw JSON header string and a tensor payload blob. It is the in-situ
// fixture builder the round-trip examples below use: the 8-byte
// little-endian header length, then the header JSON, then the payload —
// exactly the on-disk layout ReadIndex expects. Hand-rolling the header
// (rather than marshalling a map[string]HeaderEntry) lets the examples
// exercise headers a real writer emits but the test marshaller cannot —
// a __metadata__ block and the full dtype vocabulary.
func buildSafetensors(path, header string, payload []byte) error {
	out := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(header)))
	copy(out[8:], header)
	copy(out[8+len(header):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		if err, ok := result.Value.(error); ok {
			return err
		}
		return core.NewError("write failed")
	}
	return nil
}

// ExampleDTypeByteSize shows how to resolve the on-disk byte width of a
// supported dense safetensors dtype. The dtype string is matched
// case-insensitively, so a header carrying "bf16" resolves the same as
// the canonical "BF16".
func ExampleDTypeByteSize() {
	for _, dtype := range []string{"F16", "BF16", "F32", "F64", "bf16"} {
		size, err := safetensors.DTypeByteSize(dtype)
		if err != nil {
			fmt.Printf("%s: %v\n", dtype, err)
			continue
		}
		fmt.Printf("%s -> %d bytes/element\n", dtype, size)
	}
	// Output:
	// F16 -> 2 bytes/element
	// BF16 -> 2 bytes/element
	// F32 -> 4 bytes/element
	// F64 -> 8 bytes/element
	// bf16 -> 2 bytes/element
}

// ExampleDecodeFloatData decodes a little-endian F32 payload (the raw
// bytes as they appear in a safetensors tensor blob) into a []float32.
// The element count is supplied by the caller from the tensor's shape;
// DecodeFloatData validates that the payload length matches.
func ExampleDecodeFloatData() {
	// Three F32 values: 1.0, 2.5, -3.0 laid out little-endian.
	raw := make([]byte, 3*4)
	binary.LittleEndian.PutUint32(raw[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(raw[4:], math.Float32bits(2.5))
	binary.LittleEndian.PutUint32(raw[8:], math.Float32bits(-3.0))

	values, err := safetensors.DecodeFloatData("F32", raw, 3)
	if err != nil {
		fmt.Println("decode:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2.5 -3]
}

// ExampleFloat16ToFloat32 converts a single IEEE-754 half-precision bit
// pattern to float32. 0x3c00 is the half-precision encoding of 1.0 and
// 0xc000 is -2.0.
func ExampleFloat16ToFloat32() {
	fmt.Println(safetensors.Float16ToFloat32(0x3c00))
	fmt.Println(safetensors.Float16ToFloat32(0xc000))
	// Output:
	// 1
	// -2
}

// ExampleRefFromHeader builds a TensorRef from a parsed header entry. The
// returned ref carries the absolute byte range of the tensor payload in
// the source file (DataStart is dataStart + the entry's begin offset) and
// the element count derived from the shape.
func ExampleRefFromHeader() {
	entry := safetensors.HeaderEntry{
		DType:       "F32",
		Shape:       []int64{2, 3},
		DataOffsets: []int64{0, 24},
	}
	ref, err := safetensors.RefFromHeader("model.safetensors", "weight", entry, 128)
	if err != nil {
		fmt.Println("ref:", err)
		return
	}
	fmt.Printf("name=%s dtype=%s elements=%d start=%d len=%d\n",
		ref.Name, ref.DType, ref.Elements, ref.DataStart, ref.ByteLen)
	// Output:
	// name=weight dtype=F32 elements=6 start=128 len=24
}

// Example_writeReadRoundTrip is the end-to-end read/write usage: build an
// index over a source file, write a subset of its tensors to a fresh
// safetensors file with WriteSubset, then read that file back and confirm
// the float values survive bit-exact. This is how the merge / shard tools
// stream tensors between files without loading a whole model — the values
// are copied through bounded chunks, never fully materialised.
func Example_writeReadRoundTrip() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	src := core.PathJoin(dir, "src.safetensors")
	dst := core.PathJoin(dir, "subset.safetensors")

	// Source file: two F32 tensors laid out little-endian, header sorted
	// by name (alpha < beta) exactly as a real writer emits.
	alpha := []float32{1, 2, 3}
	beta := []float32{-4.5, 1024.25}
	payload := make([]byte, 0, (len(alpha)+len(beta))*4)
	for _, v := range alpha {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	for _, v := range beta {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := `{"alpha":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},` +
		`"beta":{"dtype":"F32","shape":[2],"data_offsets":[12,20]}}`
	if err := buildSafetensors(src, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(src)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}

	// Write just "alpha" to a new file, then read it straight back.
	if err := safetensors.WriteSubset(context.Background(), dst, []safetensors.TensorRef{index.Tensors["alpha"]}); err != nil {
		fmt.Println("write subset:", err)
		return
	}
	back, err := safetensors.ReadIndex(dst)
	if err != nil {
		fmt.Println("read back:", err)
		return
	}
	values, err := safetensors.ReadRefValues(back.Tensors["alpha"])
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Printf("subset tensors: %v\n", back.Names)
	fmt.Printf("alpha values: %v\n", values)
	// Output:
	// subset tensors: [alpha]
	// alpha values: [1 2 3]
}

// Example_readChunked shows reading a tensor in element-range chunks via a
// TensorReader, the pattern the chunked-write and pack-compare paths use to
// bound peak memory. OpenReader binds the ref to its file; each
// ReadFloat32Chunk decodes only the requested element window.
func Example_readChunked() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "chunked.safetensors")

	want := []float32{10, 11, 12, 13, 14, 15}
	payload := make([]byte, 0, len(want)*4)
	for _, v := range want {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := fmt.Sprintf(`{"vec":{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, len(want), len(payload))
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["vec"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()

	// Read elements [2,5) — a middle window, not the whole tensor.
	chunk, err := reader.ReadFloat32Chunk(2, 3)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Printf("elements[2:5] = %v\n", chunk)
	// Output:
	// elements[2:5] = [12 13 14]
}

// ExampleReadIndex_metadata shows that the __metadata__ entry safetensors
// writers prepend (a free-form JSON object of housekeeping fields) is
// indexed past and dropped — only real tensors appear in the returned
// Index. The metadata object here deliberately carries the full spread of
// JSON value kinds (string, array, bool, null, nested object) that the
// header walker must skip over to reach the tensor that follows it.
func ExampleReadIndex_metadata() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "with_meta.safetensors")

	// One F32 scalar payload (value 1.0) preceded by a rich metadata block.
	payload := binary.LittleEndian.AppendUint32(nil, math.Float32bits(1.0))
	header := `{"__metadata__":{` +
		`"format":"pt",` + // string value
		`"shape_hint":[1,2,3],` + // array value (skipArray)
		`"trained":true,` + // bool literal (skipLiteral)
		`"notes":null,` + // null literal (skipLiteral)
		`"extra":{"nested":"object","deep":[false]}` + // nested object (skipObject)
		`},"scalar":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	// Print via the sorted Names slice — ranging the Tensors map directly
	// would make the output order non-deterministic.
	fmt.Printf("tensors: %v\n", index.Names)
	fmt.Printf("scalar dtype: %s\n", index.Tensors["scalar"].DType)
	// Output:
	// tensors: [scalar]
	// scalar dtype: F32
}

// ExampleReadIndex_dtypes shows that ReadIndex catalogues a tensor of any
// declared dtype — indexing reads the header only, so dtypes go-mlx cannot
// yet decode (the integer, boolean and 8-bit-float families) still produce
// valid refs. The header carries the dtype string case the parser canon-
// icalises (lowercase "f32" → "F32"), demonstrating the case-insensitive
// dtype interning real-world headers from older writers rely on.
func ExampleReadIndex_dtypes() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "dtypes.safetensors")

	// Each tensor is a single element; byte widths differ by dtype so the
	// payload offsets advance accordingly. Names are pre-sorted.
	type entry struct {
		name  string
		dtype string
		bytes int
	}
	entries := []entry{
		{"a_bool", "BOOL", 1},
		{"b_f8e4", "F8_E4M3FN", 1},
		{"c_f8e5", "F8_E5M2", 1},
		{"d_i16", "I16", 2},
		{"e_i8", "I8", 1},
		{"f_lower", "f32", 4}, // lowercase — interned to F32
		{"g_u8", "U8", 1},
	}
	header := "{"
	payload := []byte{}
	offset := 0
	for i, e := range entries {
		if i > 0 {
			header += ","
		}
		header += fmt.Sprintf(`%q:{"dtype":%q,"shape":[1],"data_offsets":[%d,%d]}`,
			e.name, e.dtype, offset, offset+e.bytes)
		payload = append(payload, make([]byte, e.bytes)...)
		offset += e.bytes
	}
	header += "}"
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	for _, name := range index.Names {
		fmt.Printf("%s -> %s\n", name, index.Tensors[name].DType)
	}
	// Output:
	// a_bool -> BOOL
	// b_f8e4 -> F8_E4M3FN
	// c_f8e5 -> F8_E5M2
	// d_i16 -> I16
	// e_i8 -> I8
	// f_lower -> F32
	// g_u8 -> U8
}
