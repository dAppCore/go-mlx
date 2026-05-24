// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// SDPA paged bench coverage map (W7-E, Wave 7).
//
// ScaledDotProductAttentionPaged is the decode-time attention path
// that consumes K/V pages directly without concatenating them first.
// It's what the PagedKVCache feeds during a generation step.
//
// Coverage:
//   - Single-page (fast path that degenerates to plain SDPA).
//   - Multi-page at varying page counts (2, 4, 8, 16) to surface the
//     per-page cost.
//   - Page-size sweep: 256 vs 512 vs 1024 (the hyper-long boundary).
//   - 4D K/V shape consistent with PagedKVCache emissions.

import (
	"math"
	"testing"
)

// --- Helpers ---

// buildPagedKV constructs n pages of shape [B, H, pageSize, D].
func buildPagedKV(n int, B, H, pageSize, D int32) (keys, values []*Array) {
	keys = make([]*Array, n)
	values = make([]*Array, n)
	for i := 0; i < n; i++ {
		keys[i] = RandomUniform(0, 1, []int32{B, H, pageSize, D}, DTypeFloat32)
		values[i] = RandomUniform(0, 1, []int32{B, H, pageSize, D}, DTypeFloat32)
	}
	return
}

// --- Single-page degeneration (compare against plain SDPA) ---

func BenchmarkSDPAPaged_SinglePage_Page512_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 512, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(1, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

// --- Multi-page paged decode ---

func BenchmarkSDPAPaged_2Pages_Page256_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 256, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(2, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

func BenchmarkSDPAPaged_4Pages_Page256_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 256, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(4, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

func BenchmarkSDPAPaged_8Pages_Page256_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 256, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(8, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

func BenchmarkSDPAPaged_16Pages_Page256_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 256, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(16, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

// --- Page-size sweep ---

func BenchmarkSDPAPaged_8Pages_Page512_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 512, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(8, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

func BenchmarkSDPAPaged_8Pages_Page1024_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 1024, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKV(8, B, H, P, D)
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
		y := ScaledDotProductAttentionPaged(q, keys, values, scale)
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}
