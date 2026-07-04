// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkPagedKVCache_AppendSingleTokenPageConcat_128(b *testing.B) {
	benchmarkPagedKVCacheAppendSingleTokenPage(b, false, 128)
}

func BenchmarkPagedKVCache_AppendSingleTokenPagePrealloc_128(b *testing.B) {
	benchmarkPagedKVCacheAppendSingleTokenPage(b, true, 128)
}

func benchmarkPagedKVCacheAppendSingleTokenPage(b *testing.B, prealloc bool, tokens int) {
	k, v := makeSingleTokenKV(1)
	defer Free(k, v)
	Materialize(k, v)

	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCacheWithPrealloc(0, 256, prealloc)
		for range tokens {
			state := cache.UpdateBorrowedPages(k, v, 1)
			state.Free()
		}
		if err := Eval(cache.State()...); err != nil {
			b.Fatalf("Eval cache state: %v", err)
		}
		cache.Reset()
		clearMetalCacheAfterBenchIteration(b)
	}
}
