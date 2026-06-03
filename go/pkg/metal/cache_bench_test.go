// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkPagedKVCache_AppendSingleTokenPageConcat_128(b *testing.B) {
	benchmarkPagedKVCacheAppendSingleTokenPage(b, "0", 128)
}

func BenchmarkPagedKVCache_AppendSingleTokenPagePrealloc_128(b *testing.B) {
	benchmarkPagedKVCacheAppendSingleTokenPage(b, "1", 128)
}

func benchmarkPagedKVCacheAppendSingleTokenPage(b *testing.B, prealloc string, tokens int) {
	restore := SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", prealloc)
	defer restore()

	k, v := makeSingleTokenKV(1)
	defer Free(k, v)
	Materialize(k, v)

	b.ReportAllocs()
	for b.Loop() {
		cache := NewPagedKVCache(0, 256)
		for i := 0; i < tokens; i++ {
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
