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
	"testing"
)

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

func BenchmarkQuantize_Q5_0(b *testing.B) {
	values := benchQuantizeValues(8192)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ5_0(values)
	}
}

func BenchmarkQuantize_Q4_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ4_K(values)
	}
}

func BenchmarkQuantize_Q5_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ5_K(values)
	}
}

func BenchmarkQuantize_Q6_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ6_K(values)
	}
}

func BenchmarkQuantize_Q8_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ8_K(values)
	}
}

func BenchmarkQuantize_Q3_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ3_K(values)
	}
}

func BenchmarkQuantize_Q2_K(b *testing.B) {
	values := benchQuantizeValues(qkBlockSize * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = quantizeQ2_K(values)
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
