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
		case ggufArraySpec:
			// Tokeniser-embedded vocab arrays — element type + length
			// header, then each element framed as a GGUF value. Bench
			// harness only needs the string-element path today (vocab),
			// so other element types fail loudly rather than silently
			// emit an under-cooked fixture.
			write(typed.ElementType)
			write(uint64(len(typed.Values)))
			for _, item := range typed.Values {
				switch elem := item.(type) {
				case string:
					if typed.ElementType != ValueTypeString {
						b.Fatalf("bench fixture: string element with non-string element type %d", typed.ElementType)
					}
					writeStr(elem)
				default:
					b.Fatalf("bench fixture: unsupported array element type %T", item)
				}
			}
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
	for i := range extraStrings {
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
	for i := range count {
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

// vocabTokens — generate N synthetic tokens with the shape of a real
// BPE/SentencePiece vocab: most entries are 1-6 ASCII bytes, a
// minority push past 16 bytes (Unicode-merged tokens). The point is
// not byte-exact realism — it's giving the reader something that
// stresses the per-element string-box / arena path the way a real
// tokenizer.ggml.tokens array does.
func vocabTokens(n int) []any {
	out := make([]any, n)
	for i := range n {
		switch i % 7 {
		case 0:
			out[i] = "the"
		case 1:
			out[i] = "ing"
		case 2:
			out[i] = " a"
		case 3:
			out[i] = " the"
		case 4:
			out[i] = "Ġmodel"
		case 5:
			out[i] = "tion"
		default:
			// Slightly longer tail entry to push the average byte-length
			// past the trivial-case so allocators don't all fall into
			// the same size class.
			out[i] = "▁synthetic_vocab_entry_" + intStr(i)
		}
	}
	return out
}

func benchMetadataWithVocab(n int) []ggufMetaSpec {
	base := benchMetadata(20)
	return append(base, ggufMetaSpec{
		Key:       "tokenizer.ggml.tokens",
		ValueType: ggufValueTypeArray,
		Value: ggufArraySpec{
			ElementType: ValueTypeString,
			Values:      vocabTokens(n),
		},
	})
}

// BenchmarkInfo_ReadInfo_TokeniserVocab — the W10-T target shape:
// tokenizer-embedded gguf where the vocab array dominates header
// parse cost. N=10000 covers smaller models; N=200000 covers the
// Gemma 4 / Llama 4 class with 256k vocab. Pre-specialisation
// baseline is dominated by the per-element `string` box into a
// `[]any` slice — the specialisation returns `[]string` directly.
func BenchmarkInfo_ReadInfo_TokeniserVocab_10k(b *testing.B) {
	tmp := b.TempDir() + "/model.gguf"
	writeTestGGUFForBench(b, tmp, benchMetadataWithVocab(10000), benchTensors(50))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInfo, benchSinkErr = ReadInfo(tmp)
	}
}

func BenchmarkInfo_ReadInfo_TokeniserVocab_200k(b *testing.B) {
	tmp := b.TempDir() + "/model.gguf"
	writeTestGGUFForBench(b, tmp, benchMetadataWithVocab(200000), benchTensors(50))
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
