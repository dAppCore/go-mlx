// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the GGUF header reader.
// Per AX-11 — ReadInfo is called once per model load. Cost scales
// with metadata-entry count + tensor count. Real models have ~30
// architecture/quant config entries + 100s-1000s of tensors + (on
// tokenisers that embed the vocab) 100k+ token strings.
//
// Run:    go test -bench='BenchmarkInfo' -benchmem -run='^$' ./go/gguf

package gguf

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// writeTestGGUFForBench is a *testing.B-compatible twin of
// writeTestGGUF (which takes *testing.T). Same wire format the
// production parser reads; this writes the synthetic file to a temp
// path so the bench harness can re-open it on every iteration.
func writeTestGGUFForBench(b *testing.B, path string, metadata []ggufMetaSpec, tensors []ggufTensorSpec) {
	b.Helper()
	created := core.Create(path)
	if !created.OK {
		b.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	write := func(value any) {
		b.Helper()
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			b.Fatalf("binary write failed: %v", err)
		}
	}
	writeStr := func(value string) {
		b.Helper()
		if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
			b.Fatalf("write string length: %v", err)
		}
		if _, err := file.Write([]byte(value)); err != nil {
			b.Fatalf("write string bytes: %v", err)
		}
	}

	if _, err := file.Write([]byte("GGUF")); err != nil {
		b.Fatalf("write magic: %v", err)
	}
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(len(metadata)))

	for _, entry := range metadata {
		writeStr(entry.Key)
		write(entry.ValueType)
		switch typed := entry.Value.(type) {
		case string:
			writeStr(typed)
		case uint32:
			write(typed)
		default:
			b.Fatalf("unsupported value type %T", entry.Value)
		}
	}
	for _, tensor := range tensors {
		writeStr(tensor.Name)
		write(uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			write(dim)
		}
		write(tensor.Type)
		write(uint64(0))
	}
}

// Sinks defeat compiler DCE.
var (
	benchSinkInfo Info
	benchSinkErr  error
)

func benchMetadata(extraStrings int) []ggufMetaSpec {
	base := []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(15)},
		{Key: "qwen3.block_count", ValueType: ValueTypeUint32, Value: uint32(28)},
		{Key: "qwen3.context_length", ValueType: ValueTypeUint32, Value: uint32(40960)},
		{Key: "qwen3.embedding_length", ValueType: ValueTypeUint32, Value: uint32(2048)},
		{Key: "qwen3.attention.head_count", ValueType: ValueTypeUint32, Value: uint32(16)},
		{Key: "qwen3.attention.head_count_kv", ValueType: ValueTypeUint32, Value: uint32(8)},
	}
	for i := 0; i < extraStrings; i++ {
		base = append(base, ggufMetaSpec{
			Key:       "synthetic.entry." + intStr(i),
			ValueType: ValueTypeString,
			Value:     "value-payload-of-modest-length-" + intStr(i),
		})
	}
	return base
}

func benchTensors(count int) []ggufTensorSpec {
	out := make([]ggufTensorSpec, 0, count)
	for i := 0; i < count; i++ {
		out = append(out, ggufTensorSpec{
			Name: "blk." + intStr(i/4) + ".weight." + intStr(i%4),
			Type: TensorTypeQ4_0,
			Dims: []uint64{4096, 4096},
		})
	}
	return out
}

// intStr — small inline integer-to-string helper. Avoids importing
// strconv at the top of the bench file.
func intStr(n int) string {
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

// --- ReadInfo at varying header shapes ---

func BenchmarkInfo_ReadInfo_Minimal(b *testing.B) {
	tmp := b.TempDir() + "/model.gguf"
	writeTestGGUFForBench(b, tmp, benchMetadata(0), nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInfo, benchSinkErr = ReadInfo(tmp)
	}
}

func BenchmarkInfo_ReadInfo_TypicalLayers(b *testing.B) {
	tmp := b.TempDir() + "/model.gguf"
	// 28 layers × 7 tensors = ~200 tensor descriptors, mirroring a
	// qwen3-class model's tensor manifest size.
	writeTestGGUFForBench(b, tmp, benchMetadata(20), benchTensors(200))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInfo, benchSinkErr = ReadInfo(tmp)
	}
}

func BenchmarkInfo_ReadInfo_VocabHeavy(b *testing.B) {
	tmp := b.TempDir() + "/model.gguf"
	// 200 extra string-typed metadata entries — proxy for tokeniser
	// configuration that surfaces hundreds of string fields beyond
	// the architecture-shape entries. Real Gemma 4 tokenisers push
	// past 256k vocab entries — this bench is a conservative floor.
	writeTestGGUFForBench(b, tmp, benchMetadata(200), benchTensors(50))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInfo, benchSinkErr = ReadInfo(tmp)
	}
}

// quantize.go hot-loop benches. Per AX-11 — the inner block loop runs
// once per 32 float32s; a 7B-parameter tensor takes ~200M iterations.
// Cost shape is dominated by the per-block math (scale + per-element
// quantise) so measuring at 8192 values (256 blocks) gives a stable
// per-iteration cost without dwarfing the warm-up.

var benchSinkBytes []byte

func benchQuantizeValues(n int) []float32 {
	out := make([]float32, n)
	// Deterministic-but-non-trivial input: sine-modulated so block
	// max-abs varies across blocks (forces the scale + invScale path
	// to actually execute, vs constant-zero input which would short-
	// circuit the inner loop).
	for i := range out {
		// Map i into a small float range with sign flips. Pure-Go math
		// to keep the bench file free of imports it doesn't already use.
		x := float32(i%256) - 128
		out[i] = x / 64
	}
	return out
}

func BenchmarkQuantize_Q8_0(b *testing.B) {
	values := benchQuantizeValues(8192)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ8_0(values)
	}
}

func BenchmarkQuantize_Q4_0(b *testing.B) {
	values := benchQuantizeValues(8192)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ4_0(values)
	}
}

func BenchmarkQuantize_MaxAbs(b *testing.B) {
	values := benchQuantizeValues(8192)
	var sink float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = maxAbsFloat32(values)
	}
	_ = sink
}
