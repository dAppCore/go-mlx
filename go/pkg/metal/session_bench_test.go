// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkSession_RestorePagedCaches_Copy_8x512(b *testing.B) {
	benchmarkSessionRestorePagedCaches(b, false)
}

func BenchmarkSession_RestorePagedCaches_Transfer_8x512(b *testing.B) {
	benchmarkSessionRestorePagedCaches(b, true)
}

func benchmarkSessionRestorePagedCaches(b *testing.B, transfer bool) {
	requireMetalRuntime(b)
	const (
		pageCount     = 8
		tokensPerPage = 512
		pageSize      = 1024
	)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		snapshots := []cacheSnapshot{benchmarkSessionPagedCacheSnapshot(pageCount, tokensPerPage, pageSize)}
		b.StartTimer()
		var (
			restored []Cache
			err      error
		)
		if transfer {
			restored, err = restoreSessionCachesTransferringPaged(snapshots)
		} else {
			restored, err = restoreSessionCaches(snapshots)
		}
		b.StopTimer()
		if err != nil {
			freeCacheSnapshots(snapshots)
			b.Fatalf("restoreSessionCaches: %v", err)
		}
		freeCaches(restored)
		freeCacheSnapshots(snapshots)
		b.StartTimer()
	}
}

func benchmarkSessionPagedCacheSnapshot(pageCount, tokensPerPage, pageSize int) cacheSnapshot {
	kPages := make([]*Array, pageCount)
	vPages := make([]*Array, pageCount)
	values := make([]float32, tokensPerPage)
	for page := range pageCount {
		for i := range values {
			values[i] = float32(page*tokensPerPage + i + 1)
		}
		kPages[page] = FromValues(values, 1, 1, tokensPerPage, 1)
		vPages[page] = FromValues(values, 1, 1, tokensPerPage, 1)
	}
	return cacheSnapshot{
		mode:   KVCacheModePaged,
		kPages: kPages,
		vPages: vPages,
		offset: pageCount * tokensPerPage,
		length: pageCount * tokensPerPage,
		step:   pageSize,
	}
}
