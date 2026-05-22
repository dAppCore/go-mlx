// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Pinned-array bench coverage map (W7-E, Wave 7).
//
// Strategic: the Zero-Copy Graph Injection path (`runtime.Pinner` +
// std::mdspan + go_mlx_array_new_pinned_strided_data) is load-bearing for
// the .mp4-as-portable-knowledge thesis. These benches measure:
//
//   1. fromPinnedRawBytes throughput at typical KV cache shapes
//      [B=1, H, L, D] across L = {1, 32, 512, 4096, 16384}.
//   2. Pinned vs FromValues copy path at matched tensor sizes —
//      this is the ratio Snider wants visible.
//   3. fromPinnedRawBytesStrided cost — the mdspan-wrap path that
//      exercises the C++23 view layer.
//   4. PinSlice/Release overhead per-call (Pin scaling is hidden
//      inside fromPinnedRawBytes — these isolate the cgo/PinSlice cost).
//
// All benches that touch MLX use the standard runtime gate via the
// build tag; non-runtime probes (allocations, PinSlice scaling) run
// unconditionally to keep the cgo boundary measurable on CI.

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

// --- Helpers ---

// makePinnedFloat32Bytes returns a heap-allocated little-endian float32
// byte slice of the given element count, suitable for fromPinnedRawBytes.
func makePinnedFloat32Bytes(n int) []byte {
	raw := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)*0.5))
	}
	return raw
}

// kvShapeElements computes total elements for a [B, H, L, D] shape.
func kvShapeElements(B, H, L, D int) int {
	return B * H * L * D
}

// --- fromPinnedRawBytes — typical KV shapes [B=1, H=8, L=*, D=64] ---

func BenchmarkPinnedArray_NewFromGoSlice_KVShape_L1(b *testing.B) {
	const B, H, L, D = 1, 8, 1, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

func BenchmarkPinnedArray_NewFromGoSlice_KVShape_L32(b *testing.B) {
	const B, H, L, D = 1, 8, 32, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

func BenchmarkPinnedArray_NewFromGoSlice_KVShape_L512(b *testing.B) {
	const B, H, L, D = 1, 8, 512, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

func BenchmarkPinnedArray_NewFromGoSlice_KVShape_L4096(b *testing.B) {
	const B, H, L, D = 1, 8, 4096, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

func BenchmarkPinnedArray_NewFromGoSlice_KVShape_L16384(b *testing.B) {
	const B, H, L, D = 1, 8, 16384, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// --- fromPinnedRawBytes_4DKVShape — typical Gemma-4 global head dim ---

// Gemma 4 global attention uses head_dim 256 + a small head count.
// This shape is the realistic .mp4 stride target.
func BenchmarkPinnedArray_NewFromGoSlice_Gemma4Global_L4096(b *testing.B) {
	const B, H, L, D = 1, 4, 4096, 256
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// Gemma 4 local sliding-window attention caps at 512.
func BenchmarkPinnedArray_NewFromGoSlice_Gemma4LocalWindow_L512(b *testing.B) {
	const B, H, L, D = 1, 4, 512, 256
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// --- Pinned-zero-copy vs FromValues copy path: same payload size ---

// The copy path: FromValues materialises C memory via mlx_array_new_data.
// Compare against fromPinnedRawBytes at matched [1, 8, 4096, 64] (4 MiB
// float32). The ratio is the headline number — Snider expects pinned to
// win because it skips the host-side reshuffle and stays Go-resident.
func BenchmarkPinnedArray_VsCopyPath_FromValues_L4096(b *testing.B) {
	const B, H, L, D = 1, 8, 4096, 64
	n := kvShapeElements(B, H, L, D)
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i) * 0.5
	}
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	for b.Loop() {
		arr := FromValues(values, B, H, L, D)
		Free(arr)
	}
}

func BenchmarkPinnedArray_VsCopyPath_PinnedRaw_L4096(b *testing.B) {
	const B, H, L, D = 1, 8, 4096, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// Same comparison at L=1 — single-token decode shape. This is the hot
// path during generation; the per-token cgo boundary cost matters most
// here.
func BenchmarkPinnedArray_VsCopyPath_FromValues_L1(b *testing.B) {
	const B, H, L, D = 1, 8, 1, 64
	n := kvShapeElements(B, H, L, D)
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i) * 0.5
	}
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	for b.Loop() {
		arr := FromValues(values, B, H, L, D)
		Free(arr)
	}
}

func BenchmarkPinnedArray_VsCopyPath_PinnedRaw_L1(b *testing.B) {
	const B, H, L, D = 1, 8, 1, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// L16384 — long-context turn material. The 100k-token .mp4 retained
// state path benchmarks against this shape to confirm pinned cost stays
// O(1) regardless of tensor size.
func BenchmarkPinnedArray_VsCopyPath_FromValues_L16384(b *testing.B) {
	const B, H, L, D = 1, 8, 16384, 64
	n := kvShapeElements(B, H, L, D)
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i) * 0.5
	}
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	for b.Loop() {
		arr := FromValues(values, B, H, L, D)
		Free(arr)
	}
}

func BenchmarkPinnedArray_VsCopyPath_PinnedRaw_L16384(b *testing.B) {
	const B, H, L, D = 1, 8, 16384, 64
	n := kvShapeElements(B, H, L, D)
	raw := makePinnedFloat32Bytes(n)
	shape := []int{B, H, L, D}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytes: %v", err)
		}
		Free(arr)
	}
}

// --- fromPinnedRawBytesStrided — mdspan-wrap path ---

// Strided pinned construction exercises the C++23 std::mdspan layer
// inside pinned_array_bridge.cpp. The strides here mirror the typical
// non-contiguous view onto a larger backing buffer (e.g. taking a
// per-layer slice from a packed KV tape).
func BenchmarkPinnedArray_Strided_Subview_L4096(b *testing.B) {
	const B, H, L, D = 1, 8, 4096, 64
	const storageL = 8192
	storageShape := []int{B, H, storageL, D}
	viewShape := []int{B, H, L, D}
	viewStrides := contiguousStrides(storageShape)
	// Offset to position the view at the middle of the backing tape —
	// this is how a layer slice would be read out of the .mp4 tape.
	viewOffset := (storageL / 4) * H * D

	raw := makePinnedFloat32Bytes(kvShapeElements(B, H, storageL, D))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytesStrided(raw, storageShape, viewShape, viewStrides, viewOffset, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytesStrided: %v", err)
		}
		Free(arr)
	}
}

func BenchmarkPinnedArray_Strided_Subview_L16384(b *testing.B) {
	const B, H, L, D = 1, 8, 16384, 64
	const storageL = 32768
	storageShape := []int{B, H, storageL, D}
	viewShape := []int{B, H, L, D}
	viewStrides := contiguousStrides(storageShape)
	viewOffset := (storageL / 4) * H * D

	raw := makePinnedFloat32Bytes(kvShapeElements(B, H, storageL, D))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		arr, err := fromPinnedRawBytesStrided(raw, storageShape, viewShape, viewStrides, viewOffset, DTypeFloat32)
		if err != nil {
			b.Fatalf("fromPinnedRawBytesStrided: %v", err)
		}
		Free(arr)
	}
}

// --- contiguousStrides — pure CPU stride compute ---

// Stride compute happens on every fromPinnedRawBytes call. Cheap, but
// non-zero — bench it to confirm.
func BenchmarkPinnedArray_ContiguousStrides_4D(b *testing.B) {
	shape := []int{1, 8, 4096, 64}
	b.ReportAllocs()
	for b.Loop() {
		_ = contiguousStrides(shape)
	}
}

func BenchmarkPinnedArray_ContiguousStrides_3D(b *testing.B) {
	shape := []int{1, 4096, 2048}
	b.ReportAllocs()
	for b.Loop() {
		_ = contiguousStrides(shape)
	}
}

// --- PinSlice/Release per-call cost ---

// Isolated runtime.Pinner cost — independent of the mlx_array wrapper.
// This is the floor cost of the zero-copy strategy: if PinSlice itself
// were expensive, the pinned path would lose at small sizes regardless
// of the mdspan win at large sizes.
func BenchmarkPinnedArray_PinSlice_Release_4MiB(b *testing.B) {
	raw := makePinnedFloat32Bytes(1024 * 1024)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		var view core.PinnedView
		core.PinSlice(raw, &view)
		view.Release()
	}
}

func BenchmarkPinnedArray_PinSlice_Release_256B(b *testing.B) {
	raw := makePinnedFloat32Bytes(64)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	for b.Loop() {
		var view core.PinnedView
		core.PinSlice(raw, &view)
		view.Release()
	}
}

// --- shapeElementCount — tiny but called on every pinned-array build ---

func BenchmarkPinnedArray_ShapeElementCount_4D(b *testing.B) {
	shape := []int{1, 8, 16384, 64}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = shapeElementCount(shape)
	}
}
