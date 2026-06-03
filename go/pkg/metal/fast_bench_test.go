// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Benchmarks for fast.go decode-time hot paths that did not previously
// have direct bench coverage.  W11-Y adds them to make the
// NativePagedSingleTokenAttention pool win and the SingleTokenCacheUpdate
// shape-scratch win observable in benchmem.  Existing fused-op surfaces
// (RMSNorm, LayerNorm, RoPE, SDPA, SDPAPaged) already have their own
// dedicated bench files; this one only covers the gaps.

import (
	"math"
	"testing"
)

func resetMLXBenchMemoryCounters() {
	ClearCache()
	ResetPeakMemory()
}

func reportMLXBenchMemory(b *testing.B) {
	active := GetActiveMemory()
	cache := GetCacheMemory()
	peak := GetPeakMemory()
	b.ReportMetric(float64(active), "mlx_active_B")
	b.ReportMetric(float64(cache), "mlx_cache_B")
	b.ReportMetric(float64(active+cache), "mlx_active_cache_B")
	b.ReportMetric(float64(peak), "mlx_peak_B")
}

// --- NativePagedSingleTokenAttention ---
//
// Decode-step native paged attention. Each invocation crosses cgo with a
// run of K/V page handles. The native scratch pool keeps the key/value handle
// slices reusable without C allocations on the decode path.

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
	resetMLXBenchMemoryCounters()
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.ReportAllocs()
	for b.Loop() {
		y, ok, err := NativePagedSingleTokenAttention(q, keys, values, scale)
		if err != nil {
			b.Fatalf("NativePagedSingleTokenAttention: %v", err)
		}
		if !ok {
			b.Fatal("NativePagedSingleTokenAttention: ok = false")
		}
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
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

// --- SingleTokenCacheUpdate ---
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
		updated := SingleTokenCacheUpdate(cache, token, offset)
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
		updated := SingleTokenCacheUpdate(cache, token, offset)
		Materialize(updated)
		Free(updated)
	}
}

// --- SingleTokenCausalMask ---
//
// Per-layer causal mask build during decode. W11-Y measured this
// surface to investigate caching the 0 / -1e9 scalars at package
// scope (saving the per-call FromValue + Free pair), but the cached
// variant regressed wall-clock by ~55 percent at both 512 and 4096
// capacity — MLX's Where op pays measurable refcount-management
// overhead when the same scalar arrays are aliased across many
// invocations. Benches kept so the next visitor sees the surface
// without needing to re-add coverage.

func BenchmarkSingleTokenCausalMask_Cap512(b *testing.B) {
	offset := FromValue(7)
	defer Free(offset)
	Materialize(offset)
	b.ReportAllocs()
	for b.Loop() {
		mask := SingleTokenCausalMask(512, offset)
		Materialize(mask)
		Free(mask)
	}
}

func BenchmarkSingleTokenCausalMask_Cap4096(b *testing.B) {
	offset := FromValue(123)
	defer Free(offset)
	Materialize(offset)
	b.ReportAllocs()
	for b.Loop() {
		mask := SingleTokenCausalMask(4096, offset)
		Materialize(mask)
		Free(mask)
	}
}
