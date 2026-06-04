// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Benchmarks for the minimax_m2 safetensors header parse path. The
// MiniMax M2 staged loader hits this path once per shard on every model
// load — a large MoE pack with 32+ experts × 3 projections per layer
// produces hundreds of tensor entries per shard. Mirror of the
// safetensors_bench_test.go shape so we can compare alloc counts
// directly against the safetensors package baseline.
//
// Run: go test -bench='Minimax' -benchmem -run='^$' -benchtime=200ms ./go/pkg/metal/...

package minimaxm2

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	mm2SinkTensors map[string]miniMaxM2SafetensorTensorRef
	mm2SinkErr     error
)

// writeMiniMaxM2BenchSafetensors writes a synthetic safetensors file
// with tensorCount U8 tensors of payloadBytes each. Mirrors the shape
// used in safetensors/safetensors_bench_test.go so per-tensor cost is
// directly comparable across the two parse paths.
func writeMiniMaxM2BenchSafetensors(b *testing.B, path string, tensorCount, payloadBytes int) {
	b.Helper()
	type entry struct {
		DType       string  `json:"dtype"`
		Shape       []int64 `json:"shape"`
		DataOffsets []int64 `json:"data_offsets"`
	}
	header := map[string]entry{}
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+mm2IntStr(i/4)+".self_attn.q_proj.weight."+mm2IntStr(i%4))
	}
	core.SliceSort(names)
	var offset int64
	for _, name := range names {
		header[name] = entry{
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

// mm2IntStr — small integer-to-string helper to avoid pulling strconv
// or fmt into the bench file's import block.
func mm2IntStr(n int) string {
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

// BenchmarkMinimaxM2_ReadHeader_Small exercises the safetensors-header
// parse path for a tiny shard. Counterpart to safetensors
// BenchmarkSafetensors_ReadIndex_Small.
func BenchmarkMinimaxM2_ReadHeader_Small(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "small.safetensors")
	writeMiniMaxM2BenchSafetensors(b, path, 16, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mm2SinkTensors, mm2SinkErr = readMiniMaxM2SafetensorHeaderRefs(path)
	}
}

// BenchmarkMinimaxM2_ReadHeader_Typical exercises the path at a
// MiniMax-M2 shard scale — 200 tensors per shard is representative of
// a single shard out of a 32-expert pack.
func BenchmarkMinimaxM2_ReadHeader_Typical(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "typical.safetensors")
	writeMiniMaxM2BenchSafetensors(b, path, 200, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mm2SinkTensors, mm2SinkErr = readMiniMaxM2SafetensorHeaderRefs(path)
	}
}

// BenchmarkMinimaxM2_ReadHeader_Large stretches the parser at a larger
// expert-pack scale (500 tensors — a wider MoE pack).
func BenchmarkMinimaxM2_ReadHeader_Large(b *testing.B) {
	path := core.JoinPath(b.TempDir(), "large.safetensors")
	writeMiniMaxM2BenchSafetensors(b, path, 500, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mm2SinkTensors, mm2SinkErr = readMiniMaxM2SafetensorHeaderRefs(path)
	}
}
