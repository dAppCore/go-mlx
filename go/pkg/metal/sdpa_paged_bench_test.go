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
	return buildPagedKVWithDType(n, B, H, pageSize, D, DTypeFloat32)
}

func buildPagedKVWithDType(n int, B, H, pageSize, D int32, dtype DType) (keys, values []*Array) {
	keys = make([]*Array, n)
	values = make([]*Array, n)
	for i := range n {
		keys[i] = RandomUniform(0, 1, []int32{B, H, pageSize, D}, dtype)
		values[i] = RandomUniform(0, 1, []int32{B, H, pageSize, D}, dtype)
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

func BenchmarkSDPAPaged_16Pages_Page1024_Q1_D128(b *testing.B) {
	const B, H, P, D int32 = 1, 8, 1024, 128
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

func BenchmarkSDPAPagedFastConcat_8Pages_Page1024_Q1_D128(b *testing.B) {
	benchmarkSDPAPagedFastConcat(b, 8, 1024, DTypeFloat32)
}

func BenchmarkSDPAPagedFastConcat_16Pages_Page1024_Q1_D128(b *testing.B) {
	benchmarkSDPAPagedFastConcat(b, 16, 1024, DTypeFloat32)
}

func BenchmarkSDPAPagedNative_8Pages_Page1024_Q1_D128(b *testing.B) {
	benchmarkSDPAPagedNative(b, 8, 1024, DTypeFloat32)
}

func BenchmarkSDPAPagedNative_16Pages_Page1024_Q1_D128(b *testing.B) {
	benchmarkSDPAPagedNative(b, 16, 1024, DTypeFloat32)
}

func BenchmarkSDPAPaged_8Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedDType(b, 8, 1024, DTypeFloat16)
}

func BenchmarkSDPAPaged_16Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedDType(b, 16, 1024, DTypeFloat16)
}

func BenchmarkSDPAPagedFastConcat_8Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedFastConcat(b, 8, 1024, DTypeFloat16)
}

func BenchmarkSDPAPagedFastConcat_16Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedFastConcat(b, 16, 1024, DTypeFloat16)
}

func BenchmarkSDPAPagedFastConcat_8Pages_Page1024_QF32KVF16_CastQ(b *testing.B) {
	benchmarkSDPAPagedFastConcatMixedQuery(b, 8, 1024, true)
}

func BenchmarkSDPAPagedFastConcat_8Pages_Page1024_QF32KVF16_MixedQ(b *testing.B) {
	benchmarkSDPAPagedFastConcatMixedQuery(b, 8, 1024, false)
}

func BenchmarkSDPAPagedFastConcat_16Pages_Page1024_QF32KVF16_CastQ(b *testing.B) {
	benchmarkSDPAPagedFastConcatMixedQuery(b, 16, 1024, true)
}

func BenchmarkSDPAPagedFastConcat_16Pages_Page1024_QF32KVF16_MixedQ(b *testing.B) {
	benchmarkSDPAPagedFastConcatMixedQuery(b, 16, 1024, false)
}

func BenchmarkSDPAPagedNative_8Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedNative(b, 8, 1024, DTypeFloat16)
}

func BenchmarkSDPAPagedNative_16Pages_Page1024_Q1_D128_F16(b *testing.B) {
	benchmarkSDPAPagedNative(b, 16, 1024, DTypeFloat16)
}

func benchmarkSDPAPagedDType(b *testing.B, pageCount int, pageSize int32, dtype DType) {
	const B, H, D int32 = 1, 8, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, dtype)
	keys, values := buildPagedKVWithDType(pageCount, B, H, pageSize, D, dtype)
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

func benchmarkSDPAPagedNative(b *testing.B, pageCount int, pageSize int32, dtype DType) {
	const B, H, D int32 = 1, 8, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, dtype)
	keys, values := buildPagedKVWithDType(pageCount, B, H, pageSize, D, dtype)
	defer Free(q)
	defer Free(keys...)
	defer Free(values...)
	all := append([]*Array{q}, keys...)
	all = append(all, values...)
	Materialize(all...)

	scale := float32(1.0 / math.Sqrt(float64(D)))
	warm, ok, err := NativePagedSingleTokenAttention(q, keys, values, scale)
	if err != nil {
		b.Fatalf("NativePagedSingleTokenAttention warmup: %v", err)
	}
	if !ok {
		b.Fatal("NativePagedSingleTokenAttention warmup did not accept input")
	}
	Materialize(warm)
	Free(warm)

	resetMLXBenchMemoryCounters()
	b.ReportAllocs()
	for b.Loop() {
		y, ok, err := NativePagedSingleTokenAttention(q, keys, values, scale)
		if err != nil {
			b.Fatalf("NativePagedSingleTokenAttention: %v", err)
		}
		if !ok {
			b.Fatal("NativePagedSingleTokenAttention did not accept input")
		}
		Materialize(y)
		Free(y)
	}
	reportMLXBenchMemory(b)
}

func benchmarkSDPAPagedFastConcat(b *testing.B, pageCount int, pageSize int32, dtype DType) {
	const B, H, D int32 = 1, 8, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, dtype)
	keys, values := buildPagedKVWithDType(pageCount, B, H, pageSize, D, dtype)
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
		kBase, vBase := ConcatenatePagedState(keys, values)
		y := ScaledDotProductAttention(q, kBase, vBase, scale, false)
		Materialize(y)
		Free(y, kBase, vBase)
	}
	reportMLXBenchMemory(b)
}

func benchmarkSDPAPagedFastConcatMixedQuery(b *testing.B, pageCount int, pageSize int32, castQuery bool) {
	const B, H, D int32 = 1, 8, 128
	q := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	keys, values := buildPagedKVWithDType(pageCount, B, H, pageSize, D, DTypeFloat16)
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
		kBase, vBase := ConcatenatePagedState(keys, values)
		attentionQ := q
		var ownedQ *Array
		// Cast the query to the KV dtype when they differ — the same trivial
		// pre-attention cast gemma4.attentionQueryForKV performs (it moved to
		// package gemma4 with the architecture; reconstructed here on public ops
		// so this metal SDPA bench stays in package metal).
		if castQuery {
			if kd := kBase.Dtype(); q.Dtype() != kd && (kd == DTypeFloat16 || kd == DTypeBFloat16) {
				ownedQ = AsType(q, kd)
				attentionQ = ownedQ
			}
		}
		y := ScaledDotProductAttention(attentionQ, kBase, vBase, scale, false)
		Materialize(y)
		Free(ownedQ, y, kBase, vBase)
	}
	reportMLXBenchMemory(b)
}
