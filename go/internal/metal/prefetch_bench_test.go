// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func benchmarkAsyncDecodePrefetchTrace(b *testing.B, split bool) {
	old := enableAsyncDecodePrefetch
	enableAsyncDecodePrefetch = true
	b.Cleanup(func() { enableAsyncDecodePrefetch = old })

	cache := NewPagedKVCache(0, 256)
	defer cache.Reset()
	k, v := makeSingleTokenKVShape(1, 2, 16)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 1)
	state.Free()

	base := Zeros([]int32{1, 1, 8}, DTypeFloat32)
	defer Free(base)
	Materialize(base)

	var stack [64]*Array
	b.ReportAllocs()
	for b.Loop() {
		out := Add(base, base)
		var err error
		if split {
			_, err = asyncDecodePrefetchWithCachesTraceSplit("Benchmark", 0, "trace split", out, []Cache{cache})
		} else {
			_, err = asyncDecodePrefetchWithCachesTrace("Benchmark", 0, "trace combined", out, []Cache{cache})
		}
		if err != nil {
			Free(out)
			b.Fatal(err)
		}
		outputs := stack[:0]
		outputs = append(outputs, out)
		outputs = appendCacheDirtyState(outputs, cache)
		if err := Eval(outputs...); err != nil {
			Free(out)
			b.Fatal(err)
		}
		Free(out)
	}
}

func BenchmarkAsyncDecodePrefetchTrace_CombinedDirtyKV(b *testing.B) {
	benchmarkAsyncDecodePrefetchTrace(b, false)
}

func BenchmarkAsyncDecodePrefetchTrace_SplitDirtyKV(b *testing.B) {
	benchmarkAsyncDecodePrefetchTrace(b, true)
}
