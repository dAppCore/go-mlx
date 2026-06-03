// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkFromValues_Int32_1(b *testing.B) {
	values := []int32{42}
	b.ReportAllocs()
	for b.Loop() {
		array := FromValues(values, 1)
		Free(array)
	}
}

func BenchmarkFromValues_Int32_1Literal(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		array := FromValues([]int32{42}, 1)
		Free(array)
	}
}

func BenchmarkFromSingleInt32(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		array := fromSingleInt32(42)
		Free(array)
	}
}

func BenchmarkFromSingleInt32_Reshape2_1x1(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		array := fromSingleInt32(42)
		matrix := Reshape2(array, 1, 1)
		Free(array, matrix)
	}
}

func BenchmarkFromSingleInt32Matrix(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		array := fromSingleInt32Matrix(42)
		Free(array)
	}
}

func BenchmarkFromValues_Int32_512(b *testing.B) {
	values := make([]int32, 512)
	for i := range values {
		values[i] = int32(i)
	}
	b.ReportAllocs()
	for b.Loop() {
		array := FromValues(values, 512)
		Free(array)
	}
}

func BenchmarkFromValues_Float32_2048(b *testing.B) {
	values := make([]float32, 2048)
	for i := range values {
		values[i] = float32(i) * 0.5
	}
	b.ReportAllocs()
	for b.Loop() {
		array := FromValues(values, 2048)
		Free(array)
	}
}

func BenchmarkSuppressTokenArray_64(b *testing.B) {
	ids := make([]int32, 64)
	for i := range ids {
		ids[i] = int32(i)
	}
	b.ReportAllocs()
	for b.Loop() {
		array := suppressTokenArray(ids)
		Free(array)
	}
}
