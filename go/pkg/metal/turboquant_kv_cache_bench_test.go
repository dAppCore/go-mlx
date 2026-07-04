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

func BenchmarkTurboQuantKVCache_Update_D128_T16_P4(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	layout.Shape.SeqLen = 16
	layout.PageTokens = 16
	layout.PageSize = 4
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	keyArray := FromValues(keys, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	valueArray := FromValues(values, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	defer Free(keyArray, valueArray)

	b.ReportAllocs()
	for b.Loop() {
		cache := NewTurboQuantKVCache(0, layout.PageSize)
		outK, outV := cache.Update(keyArray, valueArray, int(layout.Shape.SeqLen))
		if err := cache.Err(); err != nil {
			b.Fatalf("Update() error = %v", err)
		}
		if outK.Dim(2) != int(layout.Shape.SeqLen) || outV.Dim(2) != int(layout.Shape.SeqLen) {
			b.Fatalf("restored length = %d/%d, want %d", outK.Dim(2), outV.Dim(2), layout.Shape.SeqLen)
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
		FreeCaches([]Cache{restored})
	}
}

func BenchmarkTurboQuantKVCache_SnapshotRestore_D128_T16_P4(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	layout.Shape.SeqLen = 16
	layout.PageTokens = 16
	layout.PageSize = 4
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	keyArray := FromValues(keys, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	valueArray := FromValues(values, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	cache := NewTurboQuantKVCache(0, layout.PageSize)
	outK, outV := cache.Update(keyArray, valueArray, int(layout.Shape.SeqLen))
	if err := cache.Err(); err != nil {
		b.Fatalf("Update() error = %v", err)
	}
	snapshot, ok, err := snapshotTurboQuantCache(cache, int(layout.Shape.SeqLen))
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
		restored, arrays, err := appendRestoreTurboQuantCacheSnapshot(nil, snapshot, int(layout.Shape.SeqLen), int(layout.Shape.SeqLen))
		if err != nil {
			b.Fatalf("appendRestoreTurboQuantCacheSnapshot() error = %v", err)
		}
		if len(arrays) != 2 || arrays[0].Dim(2) != int(layout.Shape.SeqLen) {
			b.Fatalf("restored arrays = %d len %d, want K/V length %d", len(arrays), arrays[0].Dim(2), layout.Shape.SeqLen)
		}
		FreeCaches([]Cache{restored})
	}
}

func BenchmarkTurboQuantKVCache_AppendState_D128_T8(b *testing.B) {
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
	defer func() {
		cache.Reset()
		Free(keyArray, valueArray, outK, outV)
	}()

	dst := make([]*Array, 0, 2)
	b.ReportAllocs()
	for b.Loop() {
		dst = dst[:0]
		dst = cache.AppendState(dst)
		if len(dst) != 2 {
			b.Fatalf("AppendState len = %d, want K/V", len(dst))
		}
	}
}

func BenchmarkTurboQuantKVCache_PayloadEstimate_D128_T16_P4(b *testing.B) {
	layout := turboQuantKVReferenceBenchPageLayout()
	layout.Shape.SeqLen = 16
	layout.PageTokens = 16
	layout.PageSize = 4
	keys := turboQuantKVReferenceBenchVector(int(layout.PageElementCount()))
	values := turboQuantKVReferenceBenchQuery(int(layout.PageElementCount()))
	keyArray := FromValues(keys, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	valueArray := FromValues(values, int(layout.Shape.Batch), int(layout.Shape.Heads), int(layout.Shape.SeqLen), int(layout.Shape.HeadDim))
	defer Free(keyArray, valueArray)

	cache := NewTurboQuantKVCache(0, layout.PageSize)
	outK, outV := cache.Update(keyArray, valueArray, int(layout.Shape.SeqLen))
	if err := cache.Err(); err != nil {
		b.Fatalf("Update() error = %v", err)
	}
	defer func() {
		cache.Reset()
		Free(outK, outV)
	}()
	if len(cache.payloads) != 4 {
		b.Fatalf("payload pages = %d, want 4", len(cache.payloads))
	}
	if _, err := cache.PayloadEstimate(); err != nil {
		b.Fatalf("warm PayloadEstimate() error = %v", err)
	}

	b.ReportAllocs()
	for b.Loop() {
		estimate, err := cache.PayloadEstimate()
		if err != nil {
			b.Fatalf("PayloadEstimate() error = %v", err)
		}
		if estimate.Pages != 4 || estimate.PayloadBytes == 0 || estimate.FP16BaselineBytes == 0 {
			b.Fatalf("estimate = %+v, want four-page payload accounting", estimate)
		}
	}
}
