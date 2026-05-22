// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Benchmarks for the per-token, per-layer cgo-int slice allocations in
// AsStrided, Reshape, Transpose, BroadcastTo, Slice, and SliceUpdateInplace.
// Each function used to call make([]C.int, len(shape)) on every invocation;
// the W10-A pass replaces those with [8]C.int stack arrays.
//
// Shapes mirror the Gemma 4 / Qwen 3 / Llama 3 transformer attention path:
// 4-D tensors with rank-4 starts/ends/strides for KV-cache slice work, and
// 4-D shape/stride arrays for the per-token Q/K/V AsStrided that produces
// the [B, H, L, D] view from [B*L*H*D] projections.

func BenchmarkAsStrided_4D_PerToken(b *testing.B) {
	// Single-token decode shape: B=1, H=8, L=1, D=128.  L*H*D=1024 elements.
	a := Zeros([]int32{1024}, DTypeFloat32)
	defer Free(a)

	shape := []int32{1, 8, 1, 128}
	strides := []int64{1024, 128, 1024, 1}

	b.ReportAllocs()
	for b.Loop() {
		v := AsStrided(a, shape, strides, 0)
		Free(v)
	}
}

func BenchmarkReshape_2D_PerToken(b *testing.B) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 6)
	defer Free(a)
	shape := []int32{2, 3}

	b.ReportAllocs()
	for b.Loop() {
		r := Reshape(a, shape...)
		Free(r)
	}
}

func BenchmarkReshape_4D_PerToken(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1024)
	defer Free(a)
	shape := []int32{1, 8, 1, 128}

	b.ReportAllocs()
	for b.Loop() {
		r := Reshape(a, shape...)
		Free(r)
	}
}

func BenchmarkTranspose_4D_PerToken(b *testing.B) {
	// [B, L, H, D] -> [B, H, L, D] — the Q/K/V reshape-transpose pattern.
	a := Zeros([]int32{1, 1, 8, 128}, DTypeFloat32)
	defer Free(a)
	axes := []int{0, 2, 1, 3}

	b.ReportAllocs()
	for b.Loop() {
		t := Transpose(a, axes...)
		Free(t)
	}
}

func BenchmarkBroadcastTo_4D_PerToken(b *testing.B) {
	// [1, 1, 1, 128] -> [1, 8, 1, 128] — GQA broadcast.
	a := Zeros([]int32{1, 1, 1, 128}, DTypeFloat32)
	defer Free(a)

	shape := []int32{1, 8, 1, 128}
	b.ReportAllocs()
	for b.Loop() {
		v := BroadcastTo(a, shape)
		Free(v)
	}
}

func BenchmarkSqueeze_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 1, 1, 128}, DTypeFloat32)
	defer Free(a)
	axes := []int{0, 2}

	b.ReportAllocs()
	for b.Loop() {
		s := Squeeze(a, axes...)
		Free(s)
	}
}

func BenchmarkSlice_4D_PerToken(b *testing.B) {
	// KV-cache slice pattern: [B, H, max, D] -> [B, H, offset, D].
	a := Zeros([]int32{1, 8, 64, 128}, DTypeFloat32)
	defer Free(a)

	starts := []int32{0, 0, 0, 0}
	ends := []int32{1, 8, 32, 128}

	b.ReportAllocs()
	for b.Loop() {
		s := Slice(a, starts, ends)
		Free(s)
	}
}

func BenchmarkSliceUpdateInplace_4D_PerToken(b *testing.B) {
	// KV-cache update pattern: a single token written into the cache.
	a := Zeros([]int32{1, 8, 64, 128}, DTypeFloat32)
	defer Free(a)
	upd := Zeros([]int32{1, 8, 1, 128}, DTypeFloat32)
	defer Free(upd)

	starts := []int32{0, 0, 0, 0}
	ends := []int32{1, 8, 1, 128}

	b.ReportAllocs()
	for b.Loop() {
		s := SliceUpdateInplace(a, upd, starts, ends)
		Free(s)
	}
}

func BenchmarkSoftmax_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 32000}, DTypeFloat32)
	defer Free(a)

	b.ReportAllocs()
	for b.Loop() {
		s := Softmax(a)
		Free(s)
	}
}

func BenchmarkSum_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 8, 1, 128}, DTypeFloat32)
	defer Free(a)

	b.ReportAllocs()
	for b.Loop() {
		s := Sum(a, -1, false)
		Free(s)
	}
}

func BenchmarkMean_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 8, 1, 128}, DTypeFloat32)
	defer Free(a)

	b.ReportAllocs()
	for b.Loop() {
		m := Mean(a, -1, false)
		Free(m)
	}
}

func BenchmarkZeros_4D_PerToken(b *testing.B) {
	shape := []int32{1, 8, 64, 128}

	b.ReportAllocs()
	for b.Loop() {
		z := Zeros(shape, DTypeFloat32)
		Free(z)
	}
}
