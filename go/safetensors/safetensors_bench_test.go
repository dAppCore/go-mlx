// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the safetensors header parse + subset write paths.
// Per AX-11 — ReadIndex fires once per shard on every model load; a
// Gemma-class model with 28 layers has ~200+ tensor refs. RefFromHeader,
// DecodeFloatData and WriteSubset are the inner loops both load and
// model-extract pipelines hit.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/safetensors

package safetensors

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	stSinkIndex   Index
	stSinkRef     TensorRef
	stSinkFloats  []float32
	stSinkBytes   []byte
	stSinkErr     error
)

// writeBenchSafetensors writes a synthetic safetensors file with
// tensorCount U8 tensors of payloadBytes each. U8 is used so the parser
// path mirrors what the IndexFiles bench would see on a real model
// without forcing actual quant payloads. Header build mirrors the
// production writeRawSafetensors test helper.
func writeBenchSafetensors(b *testing.B, path string, tensorCount, payloadBytes int) {
	b.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, tensorCount)
	for i := 0; i < tensorCount; i++ {
		names = append(names, "model.layers."+stIntStr(i/4)+".self_attn.q_proj.weight."+stIntStr(i%4))
	}
	core.SliceSort(names)
	var offset int64
	for _, name := range names {
		header[name] = HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(payloadBytes)},
			DataOffsets: []int64{offset, offset + int64(payloadBytes)},
		}
		offset += int64(payloadBytes)
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+int(offset))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	// Payload bytes left zero — the parser does not interpret U8 payloads
	// while building the index, so the cost we want to measure is header
	// parse + tensor-ref construction.
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		b.Fatalf("WriteFile: %v", result.Value)
	}
}

// writeBenchDenseF32Safetensors lays down a single F32 tensor of the
// requested element count, used for the decode/raw-read benches.
func writeBenchDenseF32Safetensors(b *testing.B, path string, elements int) {
	b.Helper()
	payload := make([]byte, elements*4)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(float32(i)*0.001))
	}
	header := map[string]HeaderEntry{
		"weight": {
			DType:       "F32",
			Shape:       []int64{int64(elements)},
			DataOffsets: []int64{0, int64(len(payload))},
		},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		b.Fatalf("WriteFile: %v", result.Value)
	}
}

// stIntStr — small integer-to-string helper to avoid pulling strconv
// or fmt into the bench file's import block.
func stIntStr(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// --- ReadIndex — header parse + per-tensor ref build ---

func BenchmarkSafetensors_ReadIndex_Small(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "small.safetensors")
	writeBenchSafetensors(b, path, 16, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ReadIndex(path)
	}
}

func BenchmarkSafetensors_ReadIndex_Typical(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "typical.safetensors")
	// 28 layers × 7 tensors/layer ≈ qwen3 shape.
	writeBenchSafetensors(b, path, 200, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ReadIndex(path)
	}
}

// --- IndexFiles — multi-shard merge ---

func BenchmarkSafetensors_IndexFiles_TwoShards(b *testing.B) {
	dir := b.TempDir()
	path1 := core.JoinPath(dir, "shard-1.safetensors")
	path2 := core.JoinPath(dir, "shard-2.safetensors")
	writeBenchSafetensors(b, path1, 100, 16)
	writeBenchSafetensorsOffset(b, path2, 100, 16, 100)
	paths := []string{path1, path2}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = IndexFiles(paths)
	}
}

// writeBenchSafetensorsOffset is a writeBenchSafetensors variant that
// shifts each tensor name by a constant offset so two shards generated
// at the same call site do not produce duplicate names (IndexFiles
// errors on duplicate keys).
func writeBenchSafetensorsOffset(b *testing.B, path string, tensorCount, payloadBytes, nameOffset int) {
	b.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, tensorCount)
	for i := 0; i < tensorCount; i++ {
		idx := i + nameOffset
		names = append(names, "model.layers."+stIntStr(idx/4)+".self_attn.q_proj.weight."+stIntStr(idx%4))
	}
	core.SliceSort(names)
	var offset int64
	for _, name := range names {
		header[name] = HeaderEntry{
			DType:       "U8",
			Shape:       []int64{int64(payloadBytes)},
			DataOffsets: []int64{offset, offset + int64(payloadBytes)},
		}
		offset += int64(payloadBytes)
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+int(offset))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		b.Fatalf("WriteFile: %v", result.Value)
	}
}

// --- RefFromHeader — inner loop of ReadIndex ---

func BenchmarkSafetensors_RefFromHeader_2D(b *testing.B) {
	entry := HeaderEntry{
		DType:       "F32",
		Shape:       []int64{2048, 2048},
		DataOffsets: []int64{0, 2048 * 2048 * 4},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkRef, stSinkErr = RefFromHeader("/tmp/x.safetensors", "model.layers.0.self_attn.q_proj.weight", entry, 1024)
	}
}

func BenchmarkSafetensors_RefFromHeader_4D(b *testing.B) {
	entry := HeaderEntry{
		DType:       "F16",
		Shape:       []int64{4, 28, 2048, 64},
		DataOffsets: []int64{0, 4 * 28 * 2048 * 64 * 2},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkRef, stSinkErr = RefFromHeader("/tmp/x.safetensors", "model.layers.0.self_attn.q_proj.weight", entry, 1024)
	}
}

// --- DTypeByteSize — per-tensor when opening readers ---

func BenchmarkSafetensors_DTypeByteSize_F16(b *testing.B) {
	dtype := "F16"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := DTypeByteSize(dtype)
		stSinkErr = err
		_ = n
	}
}

func BenchmarkSafetensors_DTypeByteSize_BF16(b *testing.B) {
	dtype := "bf16"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := DTypeByteSize(dtype)
		stSinkErr = err
		_ = n
	}
}

// --- Float16ToFloat32 — bit-twiddle hot path inside DecodeFloatData(F16) ---

func BenchmarkSafetensors_Float16ToFloat32_Normal(b *testing.B) {
	// 0x3c00 = 1.0 in fp16 (normal range).
	value := uint16(0x3c00)
	var sink float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Float16ToFloat32(value)
	}
	_ = sink
}

func BenchmarkSafetensors_Float16ToFloat32_Subnormal(b *testing.B) {
	// Subnormal triggers the in-loop renormalisation branch.
	value := uint16(0x0200)
	var sink float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Float16ToFloat32(value)
	}
	_ = sink
}

// --- DecodeFloatData — F32 / F16 / BF16 / F64 conversion paths ---

func BenchmarkSafetensors_DecodeFloatData_F32_512(b *testing.B) {
	elements := 512
	raw := make([]byte, elements*4)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)*0.001))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = DecodeFloatData("F32", raw, elements)
	}
}

func BenchmarkSafetensors_DecodeFloatData_F32_2048(b *testing.B) {
	elements := 2048
	raw := make([]byte, elements*4)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)*0.001))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = DecodeFloatData("F32", raw, elements)
	}
}

func BenchmarkSafetensors_DecodeFloatData_F16_2048(b *testing.B) {
	elements := 2048
	raw := make([]byte, elements*2)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint16(raw[i*2:], 0x3c00)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = DecodeFloatData("F16", raw, elements)
	}
}

func BenchmarkSafetensors_DecodeFloatData_BF16_2048(b *testing.B) {
	elements := 2048
	raw := make([]byte, elements*2)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint16(raw[i*2:], 0x3f80)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = DecodeFloatData("BF16", raw, elements)
	}
}

func BenchmarkSafetensors_DecodeFloatData_F64_2048(b *testing.B) {
	elements := 2048
	raw := make([]byte, elements*8)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(float64(i)*0.001))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = DecodeFloatData("F64", raw, elements)
	}
}

// --- Full read paths against a real (temp) file ---

func BenchmarkSafetensors_ReadRefRaw_2048F32(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "dense.safetensors")
	writeBenchDenseF32Safetensors(b, path, 2048)
	index, err := ReadIndex(path)
	if err != nil {
		b.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkBytes, stSinkErr = ReadRefRaw(ref)
	}
}

func BenchmarkSafetensors_ReadRefValues_2048F32(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "dense.safetensors")
	writeBenchDenseF32Safetensors(b, path, 2048)
	index, err := ReadIndex(path)
	if err != nil {
		b.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = ReadRefValues(ref)
	}
}

func BenchmarkSafetensors_ReadRefFloat32Chunk_512(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "dense.safetensors")
	writeBenchDenseF32Safetensors(b, path, 4096)
	index, err := ReadIndex(path)
	if err != nil {
		b.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkFloats, stSinkErr = ReadRefFloat32Chunk(ref, 0, 512)
	}
}

// --- WriteSubset roundtrip — model-extract path used by lora/serve ---

func BenchmarkSafetensors_WriteSubset_TwoTensors(b *testing.B) {
	dir := b.TempDir()
	source := core.JoinPath(dir, "source.safetensors")
	writeBenchSafetensors(b, source, 4, 64)
	index, err := ReadIndex(source)
	if err != nil {
		b.Fatalf("ReadIndex: %v", err)
	}
	refs := []TensorRef{
		index.Tensors[index.Names[0]],
		index.Tensors[index.Names[1]],
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkErr = WriteSubset(ctx, core.JoinPath(dir, "subset.safetensors"), refs)
	}
}
