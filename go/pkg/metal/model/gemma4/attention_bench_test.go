// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

// Attention bench coverage map (W7-E, Wave 7).
//
// Gemma 4 hybrid attention is 5:1 — five local sliding-window layers
// (512 tokens for E2B/E4B-style packs, 1024 for 12B Unified) + one global
// layer. Bench both paths at
// matched head counts so the cost differential is directly visible:
//
//   Local layer:  [B=1, H=8, L=512, D=256]     scale = 1/sqrt(256)
//   Global layer: [B=1, H=4, L=context, D=512] scale = 1/sqrt(512)
//
// D values corrected 2026-07: every real gemma-4 pack (E2B/E4B/12B/26B/31B)
// declares head_dim=256 for sliding_attention layers and global_head_dim=512
// for full_attention layers (load.go: HeadDim defaults from the declared
// config, then "if !isSliding && cfg.GlobalHeadDim > 0 { headDim =
// cfg.GlobalHeadDim }" overrides it on full-attention layers only). The prior
// D=128/D=256 pairing measured exactly half the real width on both branches.
//
// Both branches: causal vs masked variants. Masked is the realistic
// long-context decode path (offset-causal mask via
// gemma4CombineMasks). Causal-only is the prefill simplification.
//
// Per-context-size sweep (1k / 4k / 16k / 32k) exists only for the
// global path — local layers cap at the model-native window, so larger retained
// local contexts would mean the engine is mis-bounding the sliding window (the
// failure case IDEAS.md §1 flagged).
//
// SDPA paged variant — ScaledDotProductAttentionPaged — is benched
// alongside since it's the path the PagedKVCache feeds into.

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// --- Helpers ---

// makeAttention4D builds three [B, H, L, D] random tensors (Q, K, V).
func makeAttention4D(B, H, L, D int32) (q, k, v *metal.Array) {
	q = metal.RandomUniform(0, 1, []int32{B, H, L, D}, metal.DTypeFloat32)
	k = metal.RandomUniform(0, 1, []int32{B, H, L, D}, metal.DTypeFloat32)
	v = metal.RandomUniform(0, 1, []int32{B, H, L, D}, metal.DTypeFloat32)
	metal.Materialize(q, k, v)
	return
}

// makeAttention4DAsymm builds Q at queryLen and K/V at keyLen, mirroring
// the decode-step pattern (Q is the single new token, K/V is the full
// cache).
func makeAttention4DAsymm(B, H, queryLen, keyLen, D int32) (q, k, v *metal.Array) {
	q = metal.RandomUniform(0, 1, []int32{B, H, queryLen, D}, metal.DTypeFloat32)
	k = metal.RandomUniform(0, 1, []int32{B, H, keyLen, D}, metal.DTypeFloat32)
	v = metal.RandomUniform(0, 1, []int32{B, H, keyLen, D}, metal.DTypeFloat32)
	metal.Materialize(q, k, v)
	return
}

// --- Gemma 4 local layer (5/6 of layers — sliding window, head_dim 256) ---

func BenchmarkAttention_LocalWindow_Prefill_512(b *testing.B) {
	const B, H, L, D = 1, 8, 512, 256
	q, k, v := makeAttention4D(B, H, L, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, true)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// Decode shape: Q=1 token against K/V cache of 512 (full local window).
func BenchmarkAttention_LocalWindow_Decode_Q1_K512(b *testing.B) {
	const B, H, D = 1, 8, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 512, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// Decode shape: Q=1 with K/V at 256 — half-filled local window.
func BenchmarkAttention_LocalWindow_Decode_Q1_K256(b *testing.B) {
	const B, H, D = 1, 8, 256
	q, k, v := makeAttention4DAsymm(B, H, 1, 256, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// --- Gemma 4 global layer (1/6 of layers — full attention, p-RoPE, global_head_dim 512) ---

func BenchmarkAttention_Global_Prefill_1k(b *testing.B) {
	const B, H, L, D = 1, 4, 1024, 512
	q, k, v := makeAttention4D(B, H, L, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, true)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkAttention_Global_Prefill_4k(b *testing.B) {
	const B, H, L, D = 1, 4, 4096, 512
	q, k, v := makeAttention4D(B, H, L, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, true)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkAttention_Global_Prefill_16k(b *testing.B) {
	const B, H, L, D = 1, 4, 16384, 512
	q, k, v := makeAttention4D(B, H, L, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, true)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// Note: 32k prefill SDPA may exhaust unified memory on small machines —
// reserve for sustained runs.
func BenchmarkAttention_Global_Prefill_32k(b *testing.B) {
	const B, H, L, D = 1, 4, 32768, 512
	q, k, v := makeAttention4D(B, H, L, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * L * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, true)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// Decode against long context: Q=1, K=4k, K=16k, K=32k. This is the
// hot path during retained-state streaming — Q is small but K is huge,
// so memory bandwidth on K dominates.
func BenchmarkAttention_Global_Decode_Q1_K1k(b *testing.B) {
	const B, H, D = 1, 4, 512
	q, k, v := makeAttention4DAsymm(B, H, 1, 1024, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// N-batched decode harness — defeats the ~200us per-eval sync floor that makes
// the single-call decode benches above unable to see real per-kernel GPU time.
// Chains N attention calls (output [1,H,1,D] feeds the next query, same shape, a
// genuine serial dependency so MLX cannot dedup the subgraph) and evals ONCE.
// ns/op covers N calls + one sync, so per-call = ns/op / N reveals the real
// kernel cost below the floor. This is the instrument for "is the native decode
// kernel optimisable, or already at its bandwidth/compute floor?".
func BenchmarkAttention_Global_Decode_Q1_K1k_Batched256(b *testing.B) {
	const B, H, D, N = 1, 4, 512, 256
	q0, k, v := makeAttention4DAsymm(B, H, 1, 1024, D)
	defer metal.Free(q0, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*metal.Array, 0, N)
		q := q0
		for range N {
			y := metal.ScaledDotProductAttention(q, k, v, scale, false)
			outs = append(outs, y)
			q = y
		}
		if err := metal.Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(outs...)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K4k(b *testing.B) {
	const B, H, D = 1, 4, 512
	q, k, v := makeAttention4DAsymm(B, H, 1, 4096, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K16k(b *testing.B) {
	const B, H, D = 1, 4, 512
	q, k, v := makeAttention4DAsymm(B, H, 1, 16384, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkAttention_Global_Decode_Q1_K32k(b *testing.B) {
	const B, H, D = 1, 4, 512
	q, k, v := makeAttention4DAsymm(B, H, 1, 32768, D)
	defer metal.Free(q, k, v)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttention(q, k, v, scale, false)
		metal.Materialize(y)
		metal.Free(y)
	}
}

// --- ScaledDotProductAttentionWithMask — explicit mask path ---

// Causal mask supplied explicitly: this is what the offset-causal mask
// cache in Gemma 4 dispatches when sliding-window or partial-context
// constraints can't be inferred from causal=true alone.
func BenchmarkAttention_WithMask_Decode_Q1_K4k(b *testing.B) {
	const B, H, D = 1, 4, 512
	const keyLen = 4096
	q, k, v := makeAttention4DAsymm(B, H, 1, keyLen, D)
	defer metal.Free(q, k, v)
	// Full-true mask (no positions excluded) — bench the mask transit
	// path, not the masking math.
	mask := metal.RandomUniform(0, 1, []int32{B, H, 1, keyLen}, metal.DTypeFloat32)
	defer metal.Free(mask)
	metal.Materialize(mask)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttentionWithMask(q, k, v, mask, scale)
		metal.Materialize(y)
		metal.Free(y)
	}
}

func BenchmarkAttention_WithMask_Decode_Q1_K16k(b *testing.B) {
	const B, H, D = 1, 4, 512
	const keyLen = 16384
	q, k, v := makeAttention4DAsymm(B, H, 1, keyLen, D)
	defer metal.Free(q, k, v)
	mask := metal.RandomUniform(0, 1, []int32{B, H, 1, keyLen}, metal.DTypeFloat32)
	defer metal.Free(mask)
	metal.Materialize(mask)
	scale := float32(1.0 / math.Sqrt(float64(D)))
	b.SetBytes(int64(B * H * D * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := metal.ScaledDotProductAttentionWithMask(q, k, v, mask, scale)
		metal.Materialize(y)
		metal.Free(y)
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
		metal.Materialize(m)
		metal.Free(m)
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
		metal.Materialize(m)
		metal.Free(m)
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
		metal.Materialize(m)
		metal.Free(m)
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
		metal.Materialize(m)
		metal.Free(m)
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
		metal.Materialize(m)
		cache.Free()
	}
}

func BenchmarkAttention_RuntimeMaskCache_Reuse(b *testing.B) {
	const batch, queryLen, keyLen, offset, keyStart, window int32 = 1, 1, 4096, 0, 0, 4096
	cache := newGemma4RuntimeMaskCache()
	defer cache.Free()
	// Warm the cache.
	m := cache.CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
	metal.Materialize(m)
	b.ReportAllocs()
	for b.Loop() {
		_ = cache.CachedAttentionMask(batch, queryLen, keyLen, offset, keyStart, window)
	}
}

// --- gemma4CombineMasks (the offset-causal + extra mask combinator) ---

func BenchmarkAttention_CombineMasks_Q1_K4096(b *testing.B) {
	base := metal.RandomUniform(0, 1, []int32{1, 1, 1, 4096}, metal.DTypeFloat32)
	extra := metal.RandomUniform(0, 1, []int32{1, 1, 1, 4096}, metal.DTypeFloat32)
	defer metal.Free(base, extra)
	metal.Materialize(base, extra)
	b.ReportAllocs()
	for b.Loop() {
		m := gemma4CombineMasks(base, extra)
		metal.Materialize(m)
		if m != base && m != extra {
			metal.Free(m)
		}
	}
}
