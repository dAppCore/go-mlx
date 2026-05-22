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
