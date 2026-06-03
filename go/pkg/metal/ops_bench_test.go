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

// BenchmarkAsStrided_4D_PerToken_InlineSliceLiterals mirrors the actual
// gemma3 / gemma4 / qwen3 attention forward pattern: the [B, H, L, D]
// shape and rank-4 strides are constructed as Go slice literals INSIDE
// the per-token call (caller has only the cfg + B + L in scope).  The
// W10-A substrate fix made the AsStrided call itself 0-alloc when the
// caller passes pre-built slices; this benchmark measures the residual
// inline-literal cost that the model files still pay three times per
// layer per token (Q/K/V).
func BenchmarkAsStrided_4D_PerToken_InlineSliceLiterals(b *testing.B) {
	a := Zeros([]int32{1024}, DTypeFloat32)
	defer Free(a)

	// Treat these as if they were cfg fields (loop-hoisted to mirror the
	// model files reading from *TextConfig / *Qwen3Config).
	var (
		B   int32 = 1
		H   int32 = 8
		L   int32 = 1
		D   int32 = 128
		HxD int32 = H * D
	)

	b.ReportAllocs()
	for b.Loop() {
		v := AsStrided(a,
			[]int32{B, H, L, D},
			[]int64{int64(L * HxD), int64(D), int64(HxD), 1},
			0,
		)
		Free(v)
	}
}

// BenchmarkReshape_4D_PerToken_VariadicArgs mirrors the gemma3 / qwen3
// attention forward call site `Reshape(transposed, B, L, H*D)` — the
// variadic slice escapes to the heap because the substrate dereferences
// &shape[0] for the cgo inline call.  Documents the residual per-layer
// alloc the variadic call shape leaves at the model-layer site even
// after W10-A made the substrate Reshape 0-alloc.
func BenchmarkReshape_4D_PerToken_VariadicArgs(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1024)
	defer Free(a)

	var (
		B int32 = 1
		L int32 = 1
		H int32 = 8
		D int32 = 128
	)

	b.ReportAllocs()
	for b.Loop() {
		r := Reshape(a, B, L, H*D)
		Free(r)
	}
}

// BenchmarkTranspose_4D_PerToken_VariadicArgs mirrors the gemma3 /
// qwen3 / gemma4 attention forward call site `Transpose(out, 0, 2, 1,
// 3)`.  The variadic []int axes argument escapes to the heap because
// the substrate takes &axes[0] for the cgo inline call.
func BenchmarkTranspose_4D_PerToken_VariadicArgs(b *testing.B) {
	a := Zeros([]int32{1, 1, 8, 128}, DTypeFloat32)
	defer Free(a)

	b.ReportAllocs()
	for b.Loop() {
		t := Transpose(a, 0, 2, 1, 3)
		Free(t)
	}
}

// BenchmarkSqueeze_PerToken_VariadicArgs mirrors the gemma4
// splitPerLayerInputTensor inner-loop call `Squeeze(sliced, 2)` — one
// per layer, per forward.  The variadic []int axes escapes to the heap
// because the substrate takes &axes[0] for the cgo inline call.
func BenchmarkSqueeze_PerToken_VariadicArgs(b *testing.B) {
	a := Zeros([]int32{1, 1, 1, 128}, DTypeFloat32)
	defer Free(a)

	b.ReportAllocs()
	for b.Loop() {
		s := Squeeze(a, 2)
		Free(s)
	}
}

// BenchmarkMulScalar_PerToken / BenchmarkAddScalar_PerToken target the
// W11-F inline-C bridge.  The legacy AddScalar / MulScalar implementation
// is FromValue(s) + Add/Mul(a, scalar) + Free(scalar) — 3 cgo crossings
// plus a Go-side *Array wrapper for the scalar.  The W11-F bridge
// (mlx_add_scalar_inline / mlx_multiply_scalar_inline) materialises the
// scalar mlx_array on the C stack, dispatches the op, and frees the
// scalar before returning, collapsing the whole sequence into a single
// cgo crossing.  Sites hit by gemma4 attention scale, embedding scale,
// router scale, softcap, gemma4_vision rescaling, etc.  Per-token shape
// is the embedding row (2048 ≈ Gemma 4 1B).
func BenchmarkMulScalar_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 2048}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		y := MulScalar(a, 2.5)
		Free(y)
	}
}

func BenchmarkAddScalar_PerToken(b *testing.B) {
	a := Zeros([]int32{1, 2048}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		y := AddScalar(a, 0.25)
		Free(y)
	}
}

// BenchmarkMulScalar_1M / BenchmarkAddScalar_1M include a Materialize
// step so the Metal kernel time is part of the measurement — useful when
// reasoning about the relative impact of the bridge vs the kernel cost.
func BenchmarkMulScalar_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		y := MulScalar(a, 2.5)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAddScalar_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		y := AddScalar(a, 0.25)
		Materialize(y)
		Free(y)
	}
}

// BenchmarkReshape1_Variadic / _Scalar measure the per-call alloc cost
// of Reshape(arr, int32(n)) vs the W11-AC Reshape1(arr, n) primitive on
// the rank-1 frontier. packQ4Cached pays this on every Q4 K/V Update
// (Reshape(q, int32(n)) + Reshape(packed2D, int32(pairs))), unpackQ4 on
// every dequant (Reshape(stacked, int32(flatLen))), maxAll on every
// quantise-max boundary (Reshape(a, int32(n))). The variadic form
// escapes the int32 to heap; the scalar form passes the dim in a
// register and materialises the 1-element shape buffer on the C stack.
func BenchmarkReshape1_Variadic(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1, 1024)
	defer Free(a)
	var n int32 = 1024
	b.ReportAllocs()
	for b.Loop() {
		r := Reshape(a, n)
		Free(r)
	}
}

func BenchmarkReshape1_Scalar(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1, 1024)
	defer Free(a)
	var n int32 = 1024
	b.ReportAllocs()
	for b.Loop() {
		r := Reshape1(a, n)
		Free(r)
	}
}

// BenchmarkReshape2_Variadic / _Scalar measure the per-call alloc cost
// of Reshape(arr, int32(h), int32(w)) vs Reshape2(arr, h, w) on the
// rank-2 [pairs, 2] view that packQ4Cached materialises on every Q4 K/V
// Update.
func BenchmarkReshape2_Variadic(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1024)
	defer Free(a)
	var h, w int32 = 512, 2
	b.ReportAllocs()
	for b.Loop() {
		r := Reshape(a, h, w)
		Free(r)
	}
}

func BenchmarkReshape2_Scalar(b *testing.B) {
	data := make([]float32, 1024)
	a := FromValues(data, 1024)
	defer Free(a)
	var h, w int32 = 512, 2
	b.ReportAllocs()
	for b.Loop() {
		r := Reshape2(a, h, w)
		Free(r)
	}
}

// BenchmarkSlice1_Variadic / _Scalar measure the per-call alloc cost of
// Slice(flat, []int32{0}, []int32{n}) vs Slice1(flat, 0, n) on the
// rank-1 frontier — unpackQ4 tail-trim boundary.
func BenchmarkSlice1_Variadic(b *testing.B) {
	a := Zeros([]int32{1024}, DTypeFloat32)
	defer Free(a)
	starts := []int32{0}
	ends := []int32{512}
	b.ReportAllocs()
	for b.Loop() {
		s := Slice(a, starts, ends)
		Free(s)
	}
}

func BenchmarkSlice1_Scalar(b *testing.B) {
	a := Zeros([]int32{1024}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		s := Slice1(a, 0, 512)
		Free(s)
	}
}

// BenchmarkSlice2_SliceAxis / _Variadic / _Scalar — SliceAxis is the
// legacy path used by packQ4Cached (`SliceAxis(paired, 1, 0, 1)` +
// `SliceAxis(paired, 1, 1, 2)` per Q4 K/V Update). SliceAxis allocates
// `make([]int32, ndim)` twice per call so the rank-2 surface pays ~4
// slice heap allocs per pack. Slice2 collapses both starts + ends into
// register-passed scalars.
func BenchmarkSlice2_SliceAxis(b *testing.B) {
	a := Zeros([]int32{512, 2}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		s := SliceAxis(a, 1, 0, 1)
		Free(s)
	}
}

func BenchmarkSlice2_Variadic(b *testing.B) {
	a := Zeros([]int32{512, 2}, DTypeFloat32)
	defer Free(a)
	starts := []int32{0, 0}
	ends := []int32{512, 1}
	b.ReportAllocs()
	for b.Loop() {
		s := Slice(a, starts, ends)
		Free(s)
	}
}

func BenchmarkSlice2_Scalar(b *testing.B) {
	a := Zeros([]int32{512, 2}, DTypeFloat32)
	defer Free(a)
	b.ReportAllocs()
	for b.Loop() {
		s := Slice2(a, 0, 0, 512, 1)
		Free(s)
	}
}

// BenchmarkSliceUpdateInplace2_Variadic / _Scalar mirror the rank-2
// update pair-symmetry with Slice2.
func BenchmarkSliceUpdateInplace2_Variadic(b *testing.B) {
	a := Zeros([]int32{512, 2}, DTypeFloat32)
	defer Free(a)
	upd := Zeros([]int32{1, 2}, DTypeFloat32)
	defer Free(upd)
	starts := []int32{0, 0}
	ends := []int32{1, 2}
	b.ReportAllocs()
	for b.Loop() {
		s := SliceUpdateInplace(a, upd, starts, ends)
		Free(s)
	}
}

func BenchmarkSliceUpdateInplace2_Scalar(b *testing.B) {
	a := Zeros([]int32{512, 2}, DTypeFloat32)
	defer Free(a)
	upd := Zeros([]int32{1, 2}, DTypeFloat32)
	defer Free(upd)
	b.ReportAllocs()
	for b.Loop() {
		s := SliceUpdateInplace2(a, upd, 0, 0, 1, 2)
		Free(s)
	}
}
