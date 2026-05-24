// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// KV cache bench coverage map (W7-E, Wave 7).
//
// Five cache variants live in cache.go + prompt_cache.go:
//
//   KVCache          — unbounded, grows by step chunks (256). Owner-layer
//                      pattern for Gemma 4 global attention (1/6 of layers).
//   RotatingKVCache  — bounded, slides at maxSize. Should map onto local
//                      sliding-window layers (5/6 of layers, capped at 512).
//   FixedKVCache     — fixed-capacity ring with explicit overflow. Used by
//                      the native fixed-owner attention path.
//   QuantizedKVCache — int8 quantised K/V with optional q4 (key/value
//                      bits configurable). Memory floor.
//   PagedKVCache     — page-based growing cache with prealloc gate
//                      (GO_MLX_ENABLE_PAGED_KV_PREALLOC). Targets the
//                      paged-attention dispatch path.
//
// Coverage shape:
//   - Single-token Append at typical context sizes (1, 32, 512, 4096).
//     Sliding-window-cap (RotatingKVCache @ 512) is the cap that
//     enforces Gemma 4 local layer behaviour — bench the steady-state
//     append cost AFTER cap.
//   - Reset cost (free + zero state).
//   - Stretched-context Append (16k+) for KVCache + PagedKVCache to
//     surface the O(N) concat tax noted in IDEAS.md §1.
//
// Each Append loop pre-builds the K/V input and re-creates the cache
// per iteration to keep the measurement on the Update path rather than
// allocation amortisation. State is Evaled per iter to flush the
// Metal graph — without this, we'd just be measuring graph
// construction.

import "testing"

// --- Helpers ---

// makeSingleTokenKVShape returns a [B, H, 1, D] K/V pair for a single
// token append. Reused across cache variants — keeps payload size
// constant so the variant overhead is isolated.
func makeSingleTokenKVShape(B, H, D int32) (*Array, *Array) {
	k := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	v := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	Materialize(k, v)
	return k, v
}

// makeMultiTokenKVShape returns [B, H, L, D] for prefill-style append.
func makeMultiTokenKVShape(B, H, L, D int32) (*Array, *Array) {
	k := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	v := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	Materialize(k, v)
	return k, v
}

// --- KVCache (unbounded) ---

func BenchmarkKVCache_Append_SingleToken_FromEmpty(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewKVCache()
		_, _ = cache.Update(k, v, 1)
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Repeated single-token append — first 32 tokens. Below the 256 step
// boundary, so no buffer regrow happens.
func BenchmarkKVCache_Append_SingleToken_To32(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewKVCache()
		for i := 0; i < 32; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// 512 tokens — crosses the 256 step boundary twice, triggering buffer
// regrow. This is where the concat tax shows up.
func BenchmarkKVCache_Append_SingleToken_To512(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewKVCache()
		for i := 0; i < 512; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Multi-token prefill: one fat Update of 512 tokens.
func BenchmarkKVCache_Append_512TokenPrefill(b *testing.B) {
	k, v := makeMultiTokenKVShape(1, 8, 512, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewKVCache()
		_, _ = cache.Update(k, v, 512)
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// 4k prefill — typical agentic-turn shape.
func BenchmarkKVCache_Append_4096TokenPrefill(b *testing.B) {
	k, v := makeMultiTokenKVShape(1, 8, 4096, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewKVCache()
		_, _ = cache.Update(k, v, 4096)
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Reset cost is folded into the per-iteration KVCache_Append loops
// (each iter ends with cache.Reset). A dedicated Reset bench needs
// StopTimer/StartTimer pairing that b.Loop() does not support; for
// pure Reset cost see the allocs delta in KVCache_Append benches.

// --- RotatingKVCache (bounded sliding window — Gemma 4 local layer cap) ---

// 512-token cap matches Gemma 4 local sliding-window layers.
func BenchmarkRotatingKVCache_Append_SingleToken_BelowCap(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewRotatingKVCache(512)
		for i := 0; i < 128; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Append past the cap — this is the steady-state local layer cost.
// If the ring buffer rolls correctly, ns/op should stabilise here
// instead of growing linearly.
func BenchmarkRotatingKVCache_Append_SingleToken_PastCap(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewRotatingKVCache(512)
		// Fill past cap so we measure the steady-state path.
		for i := 0; i < 1024; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Larger cap — non-Gemma local-window scenarios.
func BenchmarkRotatingKVCache_Append_SingleToken_Cap4096_Below(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewRotatingKVCache(4096)
		for i := 0; i < 512; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// 4k cap, append past cap — long-context local-layer steady state.
func BenchmarkRotatingKVCache_Append_SingleToken_Cap4096_PastCap(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewRotatingKVCache(4096)
		for i := 0; i < 8192; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Multi-token rotating prefill — exercises updateConcat path.
func BenchmarkRotatingKVCache_Append_512Prefill_Cap512(b *testing.B) {
	k, v := makeMultiTokenKVShape(1, 8, 512, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewRotatingKVCache(512)
		_, _ = cache.Update(k, v, 512)
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// --- FixedKVCache (fixed-capacity ring) ---

func BenchmarkFixedKVCache_Append_SingleToken_Cap512_Below(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewFixedKVCache(512)
		for i := 0; i < 256; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Past cap — overflow path inside FixedKVCache.updateOverflow.
func BenchmarkFixedKVCache_Append_SingleToken_Cap512_PastCap(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewFixedKVCache(512)
		for i := 0; i < 1024; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// FP16 storage path — relevant for memory-bound long context.
func BenchmarkFixedKVCache_Append_SingleToken_FP16(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewFixedKVCacheWithDType(512, DTypeFloat16)
		for i := 0; i < 256; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// --- QuantizedKVCache (int8 / q4) ---

func BenchmarkQuantizedKVCache_Append_SingleToken_Q8Q8(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewQuantizedKVCache(512, 8, 8)
		for i := 0; i < 128; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

func BenchmarkQuantizedKVCache_Append_SingleToken_Q8Q4(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewQuantizedKVCache(512, 8, 4)
		for i := 0; i < 128; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// 4k prefill quantised — memory-bound path. Eval cost includes the
// quantize step on the just-written tail.
func BenchmarkQuantizedKVCache_Append_4096Prefill_Q8Q8(b *testing.B) {
	k, v := makeMultiTokenKVShape(1, 8, 4096, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewQuantizedKVCache(4096, 8, 8)
		_, _ = cache.Update(k, v, 4096)
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// --- PagedKVCache: page-based append ---

func BenchmarkPagedKVCache_Append_SingleToken_PageSize256_To128(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < 128; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Cross-page boundary repeatedly — exercises the page concat /
// prealloc decision in appendPages.
func BenchmarkPagedKVCache_Append_SingleToken_PageSize64_To512(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 64)
		for i := 0; i < 512; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Prealloc on — should reduce per-page allocations.
func BenchmarkPagedKVCache_Append_SingleToken_PreallocOn(b *testing.B) {
	restore := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "1")
	defer restore()
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < 256; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Prealloc off — baseline append-concat path.
func BenchmarkPagedKVCache_Append_SingleToken_PreallocOff(b *testing.B) {
	restore := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "0")
	defer restore()
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < 256; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// Prealloc + larger page count — 4k tokens with 256-token pages
// means 16 pages, exercising the page-list traversal cost.
func BenchmarkPagedKVCache_Append_4096Tokens_PageSize256_Prealloc(b *testing.B) {
	restore := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "1")
	defer restore()
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < 4096; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// MaxSize trim — bounded paged cache behaviour.
func BenchmarkPagedKVCache_Append_BoundedTo1024_PastCap(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(1024, 256)
		for i := 0; i < 2048; i++ {
			_, _ = cache.Update(k, v, 1)
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

// UpdateBorrowedPages — the borrowed-state hot path used by the
// fixed-owner attention dispatcher to avoid full-page clones.
func BenchmarkPagedKVCache_UpdateBorrowedPages_To128(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < 128; i++ {
			state := cache.UpdateBorrowedPages(k, v, 1)
			state.Free()
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		cache.Reset()
	}
}

func BenchmarkSharedKV_CloneFixedBorrowed_Gemma4LocalWindow_L512(b *testing.B) {
	keys := RandomUniform(-1, 1, []int32{1, 8, 512, 64}, DTypeFloat16)
	values := RandomUniform(-1, 1, []int32{1, 8, 512, 64}, DTypeFloat16)
	defer Free(keys, values)
	Materialize(keys, values)

	kv := sharedKV{Keys: keys, Values: values, Fixed: true, Borrowed: true}
	b.ReportAllocs()
	for b.Loop() {
		retained := kv.clone()
		retained.free()
	}
}

func BenchmarkSharedKV_ClonePagedBorrowed_8Pages(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	cache := NewPagedKVCache(0, 256)
	for i := 0; i < 2048; i++ {
		state := cache.UpdateBorrowedPages(k, v, 1)
		state.Free()
	}
	if err := Eval(cache.State()...); err != nil {
		b.Fatalf("Eval: %v", err)
	}
	pages := cache.BorrowedPageState()
	kv := sharedKV{Pages: pages, Offset: cache.Offset()}
	b.ReportAllocs()
	for b.Loop() {
		retained := kv.clone()
		retained.free()
	}
	cache.Reset()
}

// --- KV cache state access (no Update — pure reads) ---

func BenchmarkKVCache_StateAccess_After128(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	cache := NewKVCache()
	for i := 0; i < 128; i++ {
		_, _ = cache.Update(k, v, 1)
	}
	if err := Eval(cache.State()...); err != nil {
		b.Fatalf("Eval: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = cache.State()
	}
	cache.Reset()
}

func BenchmarkPagedKVCache_StateAccess_After128_PageSize256(b *testing.B) {
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	cache := NewPagedKVCache(0, 256)
	for i := 0; i < 128; i++ {
		_, _ = cache.Update(k, v, 1)
	}
	if err := Eval(cache.State()...); err != nil {
		b.Fatalf("Eval: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = cache.State()
	}
	cache.Reset()
}

// --- Detach cost (post-Eval break-graph-references step) ---

// Folded into KVCache_Append loops via the per-iter Reset path — a
// dedicated Detach bench needs StopTimer/StartTimer pairing that
// b.Loop() does not support cleanly. The detach call is part of every
// cache.Reset cycle in the Append benches above.
