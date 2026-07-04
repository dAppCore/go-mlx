// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU-only jang shape utilities. The Dequantize /
// Project paths require Metal — not benchable in CI — but the shape
// helpers fire per tensor on the same hot loops covered by the root
// model_slice benches, so the per-call cost matters.
//
// Run:    go test -bench='BenchmarkJang' -benchmem -run='^$' ./go/quant/jang

package jang

import "testing"

var (
	jangBenchInt32 []int32
	jangBenchInt   []int
	jangBenchN     int
	jangBenchErr   error
)

// --- MetalShape — uint64 → int32 with bound check ---

func BenchmarkJang_MetalShape_2D(b *testing.B) {
	shape := []uint64{2048, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchInt32, jangBenchErr = MetalShape(shape)
	}
}

func BenchmarkJang_MetalShape_4D(b *testing.B) {
	shape := []uint64{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchInt32, jangBenchErr = MetalShape(shape)
	}
}

// --- ShapeElements — overflow-checked product ---

func BenchmarkJang_ShapeElements_2D(b *testing.B) {
	shape := []int32{2048, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchN, jangBenchErr = ShapeElements(shape)
	}
}

func BenchmarkJang_ShapeElements_4D(b *testing.B) {
	shape := []int32{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchN, jangBenchErr = ShapeElements(shape)
	}
}

// --- Int32SliceToInts — pure conversion, used on every metal handoff ---

func BenchmarkJang_Int32SliceToInts_2D(b *testing.B) {
	in := []int32{2048, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchInt = Int32SliceToInts(in)
	}
}

func BenchmarkJang_Int32SliceToInts_4D(b *testing.B) {
	in := []int32{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangBenchInt = Int32SliceToInts(in)
	}
}
