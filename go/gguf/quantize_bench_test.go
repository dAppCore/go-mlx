// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the dense-safetensors header parse path in the GGUF
// quantizer. Per AX-11 — readDenseSafetensors runs once per shard on
// every quantize pass; the header walk is the alloc-heavy stage where
// the reflection-based json.Unmarshal previously dominated. These
// benches measure the header parse + per-tensor TensorRef construction
// in isolation (small F32 payloads) so the header walker cost is the
// signal — payload decode is exercised separately by the safetensors
// DecodeFloatData benches.
//
// Run:    go test -bench='BenchmarkReadDenseSafetensors' -benchmem -run='^$' ./go/gguf

package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

// Sinks defeat compiler DCE.
var (
	rdsSinkTensors []denseSafetensor
	rdsSinkErr     error
)

// writeBenchDenseSafetensors lays down a synthetic safetensors file
// with tensorCount F32 tensors, each carrying elements F32 values. The
// header is built via the public json marshal path (same shape as the
// production writer) so the readDenseSafetensors walker sees a
// realistic on-disk header layout.
func writeBenchDenseSafetensors(b *testing.B, path string, tensorCount, elements int) {
	b.Helper()
	header := map[string]safetensors.HeaderEntry{}
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+rdsIntStr(i/4)+".self_attn.q_proj.weight."+rdsIntStr(i%4))
	}
	core.SliceSort(names)
	var offset int64
	payloadStride := int64(elements * 4)
	for _, name := range names {
		header[name] = safetensors.HeaderEntry{
			DType:       "F32",
			Shape:       []int64{int64(elements)},
			DataOffsets: []int64{offset, offset + payloadStride},
		}
		offset += payloadStride
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+int(offset))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	// Payload is filled with deterministic non-zero F32 values so the
	// DecodeFloatData path inside readDenseSafetensors runs on real
	// data rather than zeros (which would short-circuit denormal paths
	// in some codecs).
	payload := out[8+len(headerBytes):]
	for i := 0; i < tensorCount*elements; i++ {
		binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(float32(i)*0.001))
	}
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		b.Fatalf("WriteFile: %v", result.Value)
	}
}

// rdsIntStr — small integer-to-string helper to avoid pulling strconv
// or fmt into the bench file's import block (mirrors the helper used
// by the safetensors package bench file).
func rdsIntStr(n int) string {
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

// BenchmarkReadDenseSafetensors_Small — 16 small tensors, the floor
// case. Header parse cost dominates over payload decode at this size.
func BenchmarkReadDenseSafetensors_Small(b *testing.B) {
	path := core.PathJoin(b.TempDir(), "small.safetensors")
	writeBenchDenseSafetensors(b, path, 16, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rdsSinkTensors, rdsSinkErr = readDenseSafetensors(path)
	}
}

// BenchmarkReadDenseSafetensors_Typical — 200 tensors × 8 elements,
// shaped like a qwen3-class shard (28 layers × ~7 tensors/layer). This
// is the headline case: the header walk runs on a realistic name +
// shape distribution.
func BenchmarkReadDenseSafetensors_Typical(b *testing.B) {
	path := core.PathJoin(b.TempDir(), "typical.safetensors")
	writeBenchDenseSafetensors(b, path, 200, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rdsSinkTensors, rdsSinkErr = readDenseSafetensors(path)
	}
}

// BenchmarkReadDenseSafetensors_LargePayload — 32 tensors × 65536 F32
// elements (256 KiB each, ~8 MiB file). The small/typical cases above
// use 8-element tensors, where payload bytes are negligible and the
// header walk dominates; this case exposes the PAYLOAD byte cost, which
// is the production reality (real shards are GB-scale, MB-scale tensors).
// Per AX-11 this guards the streaming read: readDenseSafetensors must
// NOT read the whole file into one buffer (filesize + Σdecoded B/op) —
// it reads the header only, then ReadAts each tensor into one reused
// scratch sized to the largest tensor (max_tensor_bytes + Σdecoded). On
// this fixture that is B/op 16.8MB → 8.67MB. A regression to a whole-file
// read shows here as B/op jumping back toward ~2× the file size.
func BenchmarkReadDenseSafetensors_LargePayload(b *testing.B) {
	path := core.PathJoin(b.TempDir(), "large.safetensors")
	writeBenchDenseSafetensors(b, path, 32, 65536)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rdsSinkTensors, rdsSinkErr = readDenseSafetensors(path)
	}
}
