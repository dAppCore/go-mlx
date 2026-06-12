// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

func BenchmarkNormalizeRootShapeArgs_Int32Slice(b *testing.B) {
	dims := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	args := []any{dims}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeRootShapeArgs(args)
	}
}

func BenchmarkNormalizeRootShapeArgs_IntSlice(b *testing.B) {
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8}
	args := []any{dims}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeRootShapeArgs(args)
	}
}

func BenchmarkNormalizeRootShapeArgs_PlainArgs(b *testing.B) {
	args := []any{int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeRootShapeArgs(args)
	}
}

func BenchmarkNormalizeRootInt32Arg(b *testing.B) {
	b.Run("int", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = normalizeRootInt32Arg("shape", 42)
		}
	})
	b.Run("int64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = normalizeRootInt32Arg("shape", int64(42))
		}
	})
	b.Run("uint64", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = normalizeRootInt32Arg("shape", uint64(42))
		}
	})
}

// --- merged from root_bench_test.go (orphan sweep: shape.go argument-normalisation benches) ---
// Sinks defeat compiler DCE.
var (
	rootBenchShape []int32
	rootBenchInt32 int32
	rootBenchBool  bool
)

// --- Shape normalisation (shape.go) ---

func BenchmarkShape_NormalizeShapeArgs_Empty(b *testing.B) {
	args := []any{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeShapeArgs_IntSlice4D(b *testing.B) {
	args := []any{[]int{4, 28, 2048, 64}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

// 4D variadic (the common per-tensor call shape).
func BenchmarkShape_NormalizeShapeArgs_Variadic4D(b *testing.B) {
	args := []any{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeShapeArgs_Int32SliceFastPath(b *testing.B) {
	dims := []int32{4, 28, 2048, 64}
	args := []any{dims}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeInt32Arg_Int(b *testing.B) {
	value := any(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchInt32 = normalizeRootInt32Arg("shape", value)
	}
}

func BenchmarkShape_NormalizeInt32Arg_Int64(b *testing.B) {
	value := any(int64(2048))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchInt32 = normalizeRootInt32Arg("shape", value)
	}
}

// --- Tensor-name classifiers (model_slice.go) ---
// Fired per tensor ref during SliceModel + inspection. With 1000+ refs
// per model the per-call substring scan adds up.

// Names representative of the qwen3/gemma-class checkpoint layout.
