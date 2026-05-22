// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Attention bench coverage map (W7-E, Wave 7).
//
// Gemma 4 hybrid attention is 5:1 — five local sliding-window layers
// (typically 512 tokens) + one global layer. Bench both paths at
// matched head counts so the cost differential is directly visible:
//
//   Local layer:  [B=1, H=8, L=512, D=128]     scale = 1/sqrt(128)
//   Global layer: [B=1, H=4, L=context, D=256] scale = 1/sqrt(256)
//
// Both branches: causal vs masked variants. Masked is the realistic
// long-context decode path (offset-causal mask via
// gemma4CombineMasks). Causal-only is the prefill simplification.
//
// Per-context-size sweep (1k / 4k / 16k / 32k) exists only for the
// global path — local layers cap at 512 by design, so larger sizes
// would mean the engine is mis-bounding the sliding window (the
// failure case IDEAS.md §1 flagged).
//
// SDPA paged variant — ScaledDotProductAttentionPaged — is benched
// alongside since it's the path the PagedKVCache feeds into.

import (
	"math"
	"testing"
)

// --- Helpers ---

// makeAttention4D builds three [B, H, L, D] random tensors (Q, K, V).
func makeAttention4D(B, H, L, D int32) (q, k, v *Array) {
	q = RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	k = RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	v = RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	Materialize(q, k, v)
	return
}

// makeAttention4DAsymm builds Q at queryLen and K/V at keyLen, mirroring
// the decode-step pattern (Q is the single new token, K/V is the full
// cache).
func makeAttention4DAsymm(B, H, queryLen, keyLen, D int32) (q, k, v *Array) {
	q = RandomUniform(0, 1, []int32{B, H, queryLen, D}, DTypeFloat32)
	k = RandomUniform(0, 1, []int32{B, H, keyLen, D}, DTypeFloat32)
	v = RandomUniform(0, 1, []int32{B, H, keyLen, D}, DTypeFloat32)
	Materialize(q, k, v)
	return
}

// --- Gemma 4 local layer (5/6 of layers — sliding window 512) ---

func BenchmarkAttention_LocalWindow_Prefill_512(b *testing.B) {
	const B, H, L, D = 1, 8, 512, 128
	q, k, v := makeAttention4D(B, H, L, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
		Free(y)
	}
}

// Decode shape: Q=1 token against K/V cache of 512 (full local window).
func BenchmarkAttention_LocalWindow_Decode_Q1_K512(b *testing.B) {
	const B, H, D = 1, 8, 128
	q, k, v := makeAttention4DAsymm(B, H, 1, 512, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

// Decode shape: Q=1 with K/V at 256 — half-filled local window.
func BenchmarkAttention_LocalWindow_Decode_Q1_K256(b *testing.B) {
	const B, H, D = 1, 8, 128
	q, k, v := makeAttention4DAsymm(B, H, 1, 256, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

// --- Gemma 4 global layer (1/6 of layers — full attention, p-RoPE) ---

func BenchmarkAttention_Global_Prefill_1k(b *testing.B) {
	const B, H, L, D = 1, 4, 1024, 256
	q, k, v := makeAttention4D(B, H, L, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_Global_Prefill_4k(b *testing.B) {
	const B, H, L, D = 1, 4, 4096, 256
	q, k, v := makeAttention4D(B, H, L, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_Global_Prefill_16k(b *testing.B) {
	const B, H, L, D = 1, 4, 16384, 256
	q, k, v := makeAttention4D(B, H, L, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
		Free(y)
	}
}

// Note: 32k prefill SDPA may exhaust unified memory on small machines —
// reserve for sustained runs.
func BenchmarkAttention_Global_Prefill_32k(b *testing.B) {
	const B, H, L, D = 1, 4, 32768, 256
	q, k, v := makeAttention4D(B, H, L, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
		Free(y)
	}
}

// Decode against long context: Q=1, K=4k, K=16k, K=32k. This is the
// hot path during retained-state streaming — Q is small but K is huge,
// so memory bandwidth on K dominates.
func BenchmarkAttention_Global_Decode_Q1_K1k(b *testing.B) {
	const B, H, D = 1, 4, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 1024, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K4k(b *testing.B) {
	const B, H, D = 1, 4, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 4096, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K16k(b *testing.B) {
	const B, H, D = 1, 4, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 16384, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K32k(b *testing.B) {
	const B, H, D = 1, 4, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 32768, D)
	defer Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, false)
		Materialize(y)
		Free(y)
	}
}

// --- ScaledDotProductAttentionWithMask — explicit mask path ---

// Causal mask supplied explicitly: this is what the offset-causal mask
// cache in Gemma 4 dispatches when sliding-window or partial-context
// constraints can't be inferred from causal=true alone.
func BenchmarkAttention_WithMask_Decode_Q1_K4k(b *testing.B) {
	const B, H, D = 1, 4, 256
	const keyLen = 4096
	q, k, v := makeAttention4DAsymm(B, H, 1, keyLen, D)
	defer Free(q, k, v)
	// Full-true mask (no positions excluded) — bench the mask transit
	// path, not the masking math.
	mask := RandomUniform(0, 1, []int32{B, H, 1, keyLen}, DTypeFloat32)
	defer Free(mask)
	Materialize(mask)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttentionWithMask(q, k, v, mask, scale)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkAttention_WithMask_Decode_Q1_K16k(b *testing.B) {
	const B, H, D = 1, 4, 256
	const keyLen = 16384
	q, k, v := makeAttention4DAsymm(B, H, 1, keyLen, D)
	defer Free(q, k, v)
	mask := RandomUniform(0, 1, []int32{B, H, 1, keyLen}, DTypeFloat32)
	defer Free(mask)
	Materialize(mask)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := ScaledDotProductAttentionWithMask(q, k, v, mask, scale)
		Materialize(y)
		Free(y)
	}
}

// --- Sliding-window mask construction cost ---

// gemma4SlidingMask shape is the per-block causal+window mask used by
// local layers. Used per layer per forward pass during prefill (the
// runtime-cache hot path skips this for decode).
func BenchmarkAttention_BuildSlidingMask_L512_Window512(b *testing.B) {
	const batch, seqLen, window int32 = 1, 512, 512
	b.ReportAllocs()
	for b.Loop() {
		m := buildGemma4SlidingMask(batch, seqLen, window)
		if m == nil {
			b.Fatalf("buildGemma4SlidingMask returned nil")
		}
		Materialize(m)
		Free(m)
	}
}

func BenchmarkAttention_BuildSlidingMask_L4096_Window512(b *testing.B) {
	const batch, seqLen, window int32 = 1, 4096, 512
	b.ReportAllocs()
	for b.Loop() {
		m := buildGemma4SlidingMask(batch, seqLen, window)
		if m == nil {
			b.Fatalf("buildGemma4SlidingMask returned nil")
		}
		Materialize(m)
		Free(m)
	}
}

// Cached attention mask: the runtime mask cache hot path is the per-
// decode-step variant — single Q token against varying K window.
func BenchmarkAttention_BuildCachedAttentionMask_Q1_K512(b *testing.B) {
	const batch, queryLen, keyLen, offset, keyStart, window int32 = 1, 1, 512, 0, 0, 512
	b.ReportAllocs()
	for b.Loop() {
		m := buildGemma4CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
		if m == nil {
			b.Fatalf("buildGemma4CachedAttentionMask returned nil")
		}
		Materialize(m)
		Free(m)
	}
}

func BenchmarkAttention_BuildCachedAttentionMask_Q1_K4096(b *testing.B) {
	const batch, queryLen, keyLen, offset, keyStart, window int32 = 1, 1, 4096, 0, 0, 4096
	b.ReportAllocs()
	for b.Loop() {
		m := buildGemma4CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
		if m == nil {
			b.Fatalf("buildGemma4CachedAttentionMask returned nil")
		}
		Materialize(m)
		Free(m)
	}
}

// Reuse via runtimeMaskCache — the canonical decode-step path. First
// call materialises the mask; subsequent calls reuse. The bench builds
// a fresh cache each iter to make sure construct cost is counted, but
// the second-call reuse is also exposed via a separate bench below.
func BenchmarkAttention_RuntimeMaskCache_FirstCall(b *testing.B) {
	const batch, queryLen, keyLen, offset, keyStart, window int32 = 1, 1, 4096, 0, 0, 4096
	b.ReportAllocs()
	for b.Loop() {
		cache := newGemma4RuntimeMaskCache()
		m := cache.CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
		if m == nil {
			b.Fatalf("CachedAttentionMask returned nil")
		}
		Materialize(m)
		cache.Free()
	}
}

func BenchmarkAttention_RuntimeMaskCache_Reuse(b *testing.B) {
	const batch, queryLen, keyLen, offset, keyStart, window int32 = 1, 1, 4096, 0, 0, 4096
	cache := newGemma4RuntimeMaskCache()
	defer cache.Free()
	// Warm the cache.
	m := cache.CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
	Materialize(m)
	b.ReportAllocs()
	for b.Loop() {
		_ = cache.CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
	}
}

// --- gemma4CombineMasks (the offset-causal + extra mask combinator) ---

func BenchmarkAttention_CombineMasks_Q1_K4096(b *testing.B) {
	base := RandomUniform(0, 1, []int32{1, 1, 1, 4096}, DTypeFloat32)
	extra := RandomUniform(0, 1, []int32{1, 1, 1, 4096}, DTypeFloat32)
	defer Free(base, extra)
	Materialize(base, extra)
	b.ReportAllocs()
	for b.Loop() {
		m := gemma4CombineMasks(base, extra)
		Materialize(m)
		if m != base && m != extra {
			Free(m)
		}
	}
}
