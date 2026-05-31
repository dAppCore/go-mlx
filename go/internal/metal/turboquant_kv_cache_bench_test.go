// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkTurboQuantKVCache_Update_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	keyArray := FromValues(keys, int(layout.Shape.Batch), int(layout.Shape.Heads), layout.PageTokens, int(layout.Shape.HeadDim))
	valueArray := FromValues(values, int(layout.Shape.Batch), int(layout.Shape.Heads), layout.PageTokens, int(layout.Shape.HeadDim))
	defer Free(keyArray, valueArray)

	b.ReportAllocs()
	for b.Loop() {
		cache := NewTurboQuantKVCache(0, layout.PageTokens)
		outK, outV := cache.Update(keyArray, valueArray, layout.PageTokens)
		if err := cache.Err(); err != nil {
			b.Fatalf("Update() error = %v", err)
		}
		if outK.Dim(2) != int(layout.PageTokens) || outV.Dim(2) != int(layout.PageTokens) {
			b.Fatalf("restored length = %d/%d, want %d", outK.Dim(2), outV.Dim(2), layout.PageTokens)
		}
		cache.Reset()
	}
}

func BenchmarkTurboQuantKVCache_SnapshotRestore_D128_T8(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	keyArray := FromValues(keys, int(layout.Shape.Batch), int(layout.Shape.Heads), layout.PageTokens, int(layout.Shape.HeadDim))
	valueArray := FromValues(values, int(layout.Shape.Batch), int(layout.Shape.Heads), layout.PageTokens, int(layout.Shape.HeadDim))
	cache := NewTurboQuantKVCache(0, layout.PageTokens)
	outK, outV := cache.Update(keyArray, valueArray, layout.PageTokens)
	if err := cache.Err(); err != nil {
		b.Fatalf("Update() error = %v", err)
	}
	snapshot, ok, err := snapshotTurboQuantCache(cache, layout.PageTokens)
	if err != nil {
		b.Fatalf("snapshotTurboQuantCache() error = %v", err)
	}
	if !ok {
		b.Fatal("snapshotTurboQuantCache() ok = false, want true")
	}
	defer func() {
		cache.Reset()
		Free(keyArray, valueArray, outK, outV)
	}()

	b.ReportAllocs()
	for b.Loop() {
		restored, arrays, err := appendRestoreTurboQuantCacheSnapshot(nil, snapshot, layout.PageTokens, layout.PageTokens)
		if err != nil {
			b.Fatalf("appendRestoreTurboQuantCacheSnapshot() error = %v", err)
		}
		if len(arrays) != 2 || arrays[0].Dim(2) != int(layout.PageTokens) {
			b.Fatalf("restored arrays = %d len %d, want K/V length %d", len(arrays), arrays[0].Dim(2), layout.PageTokens)
		}
		freeCaches([]Cache{restored})
	}
}
