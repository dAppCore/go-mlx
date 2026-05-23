// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Prompt cache bench coverage map (W7-E, Wave 7).
//
// The prompt-cache subsystem (prompt_cache.go) feeds the retained-
// state "warm restore" path that IDEAS.md Q1 makes load-bearing for
// the .mp4-as-portable-knowledge thesis. The full prompt-cache hot
// path requires a loaded model — too heavy for these synthetic
// benches — but the lower-level building blocks ARE benchable:
//
//   - longestTokenPrefix — called once per match attempt; cost scales
//     with prompt size.
//   - cacheStateArraysForDetach + evalCachesBeforeDetach — bench the
//     detach setup on synthetic caches.
//   - Slice/Concatenate of the KV cache tensors — the underlying ops
//     that the prompt-cache restore path leans on.
//
// Anything that requires a real *Model fixture is deferred to a
// separate fixture-loading harness (covered by smaller surface
// benches in this file's "prefill helpers" section).

import "testing"

// --- longestTokenPrefix (token-prefix scan) ---

// 1k tokens, full match.
func BenchmarkPromptCache_LongestTokenPrefix_1k_FullMatch(b *testing.B) {
	a := make([]int32, 1024)
	for i := range a {
		a[i] = int32(i)
	}
	c := make([]int32, 1024)
	copy(c, a)
	b.ReportAllocs()
	for b.Loop() {
		_ = longestTokenPrefix(a, c)
	}
}

// 4k tokens, full match — typical warm-restore prefix size.
func BenchmarkPromptCache_LongestTokenPrefix_4k_FullMatch(b *testing.B) {
	a := make([]int32, 4096)
	for i := range a {
		a[i] = int32(i)
	}
	c := make([]int32, 4096)
	copy(c, a)
	b.ReportAllocs()
	for b.Loop() {
		_ = longestTokenPrefix(a, c)
	}
}

// 32k tokens, full match — long-context retained-state warm restore.
func BenchmarkPromptCache_LongestTokenPrefix_32k_FullMatch(b *testing.B) {
	a := make([]int32, 32768)
	for i := range a {
		a[i] = int32(i)
	}
	c := make([]int32, 32768)
	copy(c, a)
	b.ReportAllocs()
	for b.Loop() {
		_ = longestTokenPrefix(a, c)
	}
}

// 32k tokens, divergence at 16k — typical "agent turn" pattern where
// the new prompt extends a previously cached prefix.
func BenchmarkPromptCache_LongestTokenPrefix_32k_DivergeAt16k(b *testing.B) {
	a := make([]int32, 32768)
	for i := range a {
		a[i] = int32(i)
	}
	c := make([]int32, 32768)
	copy(c[:16384], a[:16384])
	for i := 16384; i < len(c); i++ {
		c[i] = int32(i + 1000000)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = longestTokenPrefix(a, c)
	}
}

// 32k tokens, divergence at position 0 — worst case (every position
// scanned for nothing).
func BenchmarkPromptCache_LongestTokenPrefix_32k_DivergeAt0(b *testing.B) {
	a := make([]int32, 32768)
	for i := range a {
		a[i] = int32(i)
	}
	c := make([]int32, 32768)
	for i := range c {
		c[i] = int32(i + 1)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = longestTokenPrefix(a, c)
	}
}

// --- Slice cost — KV cache retained-state slice extraction ---

// Per IDEAS.md, the .mp4 retained-state path treats KV tensors as a
// continuous tape. Reading a slice at offset N is a Slice op.
func BenchmarkPromptCache_KVSlice_From32k_To4kSlice(b *testing.B) {
	const B, H, L, D = 1, 8, 32768, 64
	tape := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	defer Free(tape)
	Materialize(tape)
	b.ReportAllocs()
	for b.Loop() {
		out := Slice(tape, []int32{0, 0, 0, 0}, []int32{B, H, 4096, D})
		Materialize(out)
		Free(out)
	}
}

// Read a 4k slice from the middle of a 32k tape (offset 14336).
func BenchmarkPromptCache_KVSlice_From32k_MiddleSlice(b *testing.B) {
	const B, H, L, D = 1, 8, 32768, 64
	tape := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	defer Free(tape)
	Materialize(tape)
	b.ReportAllocs()
	for b.Loop() {
		out := Slice(tape, []int32{0, 0, 14336, 0}, []int32{B, H, 18432, D})
		Materialize(out)
		Free(out)
	}
}

// Slice [B,H,L,D] to a single token's [B,H,1,D]. This is the per-
// token write target — every Update on FixedKVCache effectively
// requires this kind of slice setup.
func BenchmarkPromptCache_KVSlice_OneTokenWindow(b *testing.B) {
	const B, H, L, D = 1, 8, 16384, 64
	tape := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	defer Free(tape)
	Materialize(tape)
	b.ReportAllocs()
	for b.Loop() {
		out := Slice(tape, []int32{0, 0, 8192, 0}, []int32{B, H, 8193, D})
		Materialize(out)
		Free(out)
	}
}

// --- Concatenate cost — append new token's K/V to existing cache ---

// IDEAS.md §1: "If you are dynamically concatenating new tokens to the
// KV arrays instead of writing into a pre-allocated buffer with offset
// indexing, you are triggering massive background memory copies (O(N²)
// data movement)."
//
// Bench the Concatenate cost at varying base sizes to confirm the
// O(N) scaling. If it scales worse than O(N), the engine is hitting
// the copy trap.
func BenchmarkPromptCache_KVConcat_4k_PlusToken(b *testing.B) {
	const B, H, D = 1, 8, 64
	base := RandomUniform(0, 1, []int32{B, H, 4096, D}, DTypeFloat32)
	one := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	defer Free(base, one)
	Materialize(base, one)
	b.ReportAllocs()
	for b.Loop() {
		out := Concatenate([]*Array{base, one}, 2)
		Materialize(out)
		Free(out)
	}
}

func BenchmarkPromptCache_KVConcat_16k_PlusToken(b *testing.B) {
	const B, H, D = 1, 8, 64
	base := RandomUniform(0, 1, []int32{B, H, 16384, D}, DTypeFloat32)
	one := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	defer Free(base, one)
	Materialize(base, one)
	b.ReportAllocs()
	for b.Loop() {
		out := Concatenate([]*Array{base, one}, 2)
		Materialize(out)
		Free(out)
	}
}

func BenchmarkPromptCache_KVConcat_32k_PlusToken(b *testing.B) {
	const B, H, D = 1, 8, 64
	base := RandomUniform(0, 1, []int32{B, H, 32768, D}, DTypeFloat32)
	one := RandomUniform(0, 1, []int32{B, H, 1, D}, DTypeFloat32)
	defer Free(base, one)
	Materialize(base, one)
	b.ReportAllocs()
	for b.Loop() {
		out := Concatenate([]*Array{base, one}, 2)
		Materialize(out)
		Free(out)
	}
}

// Multi-page Concatenate — what PagedKVCache appendPagesConcat does.
func BenchmarkPromptCache_KVConcat_4Pages_512Each(b *testing.B) {
	const B, H, D = 1, 8, 64
	pages := make([]*Array, 4)
	for i := range pages {
		pages[i] = RandomUniform(0, 1, []int32{B, H, 512, D}, DTypeFloat32)
	}
	defer Free(pages...)
	Materialize(pages...)
	b.ReportAllocs()
	for b.Loop() {
		out := Concatenate(pages, 2)
		Materialize(out)
		Free(out)
	}
}

func BenchmarkPromptCache_KVConcat_16Pages_256Each(b *testing.B) {
	const B, H, D = 1, 8, 64
	pages := make([]*Array, 16)
	for i := range pages {
		pages[i] = RandomUniform(0, 1, []int32{B, H, 256, D}, DTypeFloat32)
	}
	defer Free(pages...)
	Materialize(pages...)
	b.ReportAllocs()
	for b.Loop() {
		out := Concatenate(pages, 2)
		Materialize(out)
		Free(out)
	}
}

// --- prefillCacheStateArrays — bench against a few synthetic caches ---

func BenchmarkPromptCache_PrefillCacheStateArrays_8Caches(b *testing.B) {
	caches := make([]Cache, 8)
	for i := range caches {
		caches[i] = NewKVCache()
	}
	// Push one append into each so State() returns non-nil.
	k, v := makeSingleTokenKVShape(1, 8, 64)
	defer Free(k, v)
	for _, c := range caches {
		_, _ = c.Update(k, v, 1)
	}
	if err := Eval(caches[0].State()...); err != nil {
		b.Fatalf("Eval: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = prefillCacheStateArrays(caches)
	}
	for _, c := range caches {
		c.Reset()
	}
}

func BenchmarkPromptCache_PrefillCacheStateArrays_26Caches_Gemma4(b *testing.B) {
	caches := make([]Cache, 26)
	for i := range caches {
		caches[i] = NewKVCache()
	}
	k, v := makeSingleTokenKVShape(1, 4, 64)
	defer Free(k, v)
	for _, c := range caches {
		_, _ = c.Update(k, v, 1)
	}
	if err := Eval(caches[0].State()...); err != nil {
		b.Fatalf("Eval: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = prefillCacheStateArrays(caches)
	}
	for _, c := range caches {
		c.Reset()
	}
}

// --- copyCachePrefix — golden-path warm-restore per-K and per-V hit ---
//
// Wave 11 (W11-W): copyCachePrefix is the hot inner of
// restorePromptCachesWithRequestFixedSize — called twice per restored
// cache (K and V). The W11-W swap dropped Shape() heap alloc + two
// `[]int32{...}` literals fed into Slice() to stack scratch + Slice4
// scalar-pass. Bench at the cache-tape sizes that the warm-restore
// substrate sees in production.

// 4k prefix copy — covers same-length fast path (no Slice op).
func BenchmarkPromptCache_CopyCachePrefix_4k_FullLen(b *testing.B) {
	const B, H, L, D = 1, 8, 4096, 64
	tape := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	defer Free(tape)
	Materialize(tape)
	b.ReportAllocs()
	for b.Loop() {
		out, err := copyCachePrefix(tape, L)
		if err != nil {
			b.Fatalf("copyCachePrefix: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}

// 32k tape, 4k prefix — covers the slice-then-copy path that warm
// restore actually walks when re-installing a saved prefix into a
// larger pre-allocated buffer.
func BenchmarkPromptCache_CopyCachePrefix_32kTape_4kPrefix(b *testing.B) {
	const B, H, L, D = 1, 8, 32768, 64
	tape := RandomUniform(0, 1, []int32{B, H, L, D}, DTypeFloat32)
	defer Free(tape)
	Materialize(tape)
	b.ReportAllocs()
	for b.Loop() {
		out, err := copyCachePrefix(tape, 4096)
		if err != nil {
			b.Fatalf("copyCachePrefix: %v", err)
		}
		Materialize(out)
		Free(out)
	}
}

// --- snapshotFixedCache → restoreFixedCacheSnapshot — golden-path round trip ---
//
// Fixed-cache restore is the W11-W primary target (Gemma 4 warm-load).
// snapshotFixedCache copies the prefix out of the on-device buffer;
// restoreFixedCacheSnapshot allocates a maxSize buffer and writes the
// prefix back in. Both touch SliceUpdateInplace4 / Slice4 after W11-W.

func BenchmarkPromptCache_FixedCacheSnapshotRestore_RoundTrip(b *testing.B) {
	const maxSize = 512
	const prefixLen = 256
	cache := NewFixedKVCache(maxSize)
	defer cache.Reset()
	k, v := makeKV(prefixLen)
	defer Free(k, v)
	stateK, stateV := cache.Update(k, v, prefixLen)
	Free(stateK, stateV)

	b.ReportAllocs()
	for b.Loop() {
		snap, ok, err := snapshotFixedCache(cache, prefixLen)
		if err != nil || !ok {
			b.Fatalf("snapshotFixedCache: ok=%v err=%v", ok, err)
		}
		restored, arrays, err := restoreFixedCacheSnapshot(snap, prefixLen, prefixLen, maxSize)
		if err != nil {
			freeCacheSnapshot(snap)
			b.Fatalf("restoreFixedCacheSnapshot: %v", err)
		}
		if err := Eval(arrays...); err != nil {
			freeCaches([]Cache{restored})
			freeCacheSnapshot(snap)
			b.Fatalf("Eval: %v", err)
		}
		freeCaches([]Cache{restored})
		freeCacheSnapshot(snap)
	}
}
