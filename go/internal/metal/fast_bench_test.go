// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Benchmarks for fast.go decode-time hot paths that did not previously
// have direct bench coverage.  W11-Y adds them to make the
// nativePagedSingleTokenAttention pool win and the singleTokenCacheUpdate
// shape-scratch win observable in benchmem.  Existing fused-op surfaces
// (RMSNorm, LayerNorm, RoPE, SDPA, SDPAPaged) already have their own
// dedicated bench files; this one only covers the gaps.

import (
	"math"
	"testing"
)

// --- nativePagedSingleTokenAttention ---
//
// Decode-step native paged attention. Each invocation crosses cgo with a
// run of K/V page handles. The W11-Y pool of *[]C.mlx_array converts the
// two per-call C.calloc/C.free trips into a sync.Pool round-trip.

func benchNativePagedSingleToken(b *testing.B, pageCount int, pageSize int32) {
	const B, H, D int32 = 1, 8, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(pageCount, B, H, pageSize, D)
	defer Free(q)
	defer Free(keys...)
	defer Free(values...)
	all := append([]*Array{q}, keys...)
	all = append(all, values...)
	Materialize(all...)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.ReportAllocs()
	for b.Loop() {
		y, ok, err := nativePagedSingleTokenAttention(q, keys, values, scale)
		if err != nil {
			b.Fatalf("nativePagedSingleTokenAttention: %v", err)
		}
		if !ok {
			b.Fatal("nativePagedSingleTokenAttention: ok = false")
		}
		Materialize(y)
		Free(y)
	}
}

func BenchmarkNativePagedSingleToken_2Pages_Page256(b *testing.B) {
	benchNativePagedSingleToken(b, 2, 256)
}

func BenchmarkNativePagedSingleToken_4Pages_Page256(b *testing.B) {
	benchNativePagedSingleToken(b, 4, 256)
}

func BenchmarkNativePagedSingleToken_8Pages_Page256(b *testing.B) {
	benchNativePagedSingleToken(b, 8, 256)
}

func BenchmarkNativePagedSingleToken_16Pages_Page256(b *testing.B) {
	benchNativePagedSingleToken(b, 16, 256)
}

// --- singleTokenCacheUpdate ---
//
// Per-layer, per-decode-step cache write. The W11-Y change drops the
// per-call `make([]int32, ndim)` allocation that token.Shape() pays by
// switching to a stack-allocated ShapeInto scratch.

func BenchmarkSingleTokenCacheUpdate_Heads8_Cap512_D128(b *testing.B) {
	const B, H, Cap, D int32 = 1, 8, 512, 128
	cache := RandomUniform(0, 1, []int32{B, H, Cap, D}, DTypeFloat32)
	token := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	offset := FromValue(3)
	defer Free(cache, token, offset)
	Materialize(cache, token, offset)
	b.ReportAllocs()
	for b.Loop() {
		updated := singleTokenCacheUpdate(cache, token, offset)
		Materialize(updated)
		Free(updated)
	}
}

func BenchmarkSingleTokenCacheUpdate_Heads32_Cap4096_D128(b *testing.B) {
	const B, H, Cap, D int32 = 1, 32, 4096, 128
	cache := RandomUniform(0, 1, []int32{B, H, Cap, D}, DTypeFloat32)
	token := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	offset := FromValue(17)
	defer Free(cache, token, offset)
	Materialize(cache, token, offset)
	b.ReportAllocs()
	for b.Loop() {
		updated := singleTokenCacheUpdate(cache, token, offset)
		Materialize(updated)
		Free(updated)
	}
}
