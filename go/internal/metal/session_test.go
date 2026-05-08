// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestSessionCacheSnapshot_RestoresWrappedRotatingOffset_Good(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot RestoresWrappedRotatingOffset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewRotatingKVCache(2)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval rotating cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer freeCaches([]Cache{cache})

	snapshot, ok, err := snapshotSessionCache(cache)
	if err != nil {
		t.Fatalf("snapshotSessionCache: %v", err)
	}
	if !ok {
		t.Fatal("snapshotSessionCache() ok = false, want true")
	}
	if snapshot.offset != 4 || snapshot.length != 2 {
		t.Fatalf("snapshot offset/length = %d/%d, want 4/2", snapshot.offset, snapshot.length)
	}
	defer Free(snapshot.keys, snapshot.values)

	restored, err := restoreSessionCaches([]cacheSnapshot{snapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches: %v", err)
	}
	defer freeCaches(restored)
	if len(restored) != 1 {
		t.Fatalf("restored len = %d, want 1", len(restored))
	}
	if restored[0].Offset() != 4 || restored[0].Len() != 2 {
		t.Fatalf("restored offset/len = %d/%d, want 4/2", restored[0].Offset(), restored[0].Len())
	}
}

func TestSessionCacheSnapshot_Bad(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	_, ok, err := snapshotSessionCache(nil)
	if err != nil {
		t.Fatalf("snapshotSessionCache(nil) error = %v", err)
	}
	if ok {
		t.Fatal("snapshotSessionCache(nil) ok = true, want false")
	}
}

func TestSessionCacheSnapshot_Ugly(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewKVCache()

	_, ok, err := snapshotSessionCache(cache)

	if err != nil {
		t.Fatalf("snapshotSessionCache(empty) error = %v", err)
	}
	if ok {
		t.Fatal("snapshotSessionCache(empty) ok = true, want false")
	}
}

func TestSessionKVSnapshot_RestoreLayerAndLogits_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreLayerAndLogits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := &KVSnapshot{
		Version:      KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		TokenOffset:  4,
		SeqLen:       2,
		HeadDim:      2,
		LogitShape:   []int32{1, 1, 3},
		Logits:       []float32{0.1, 0.2, 0.7},
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewRotatingKVCache(8))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	defer Free(layerSnapshot.keys, layerSnapshot.values)
	restored, err := restoreSessionCaches([]cacheSnapshot{layerSnapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches() error = %v", err)
	}
	defer freeCaches(restored)
	logits, err := restoreSnapshotLogits(snapshot)
	if err != nil {
		t.Fatalf("restoreSnapshotLogits() error = %v", err)
	}
	defer Free(logits)

	if restored[0].Offset() != 4 || restored[0].Len() != 2 {
		t.Fatalf("restored offset/len = %d/%d, want 4/2", restored[0].Offset(), restored[0].Len())
	}
	if shape := logits.Shape(); len(shape) != 3 || shape[2] != 3 {
		t.Fatalf("logit shape = %v, want [1 1 3]", shape)
	}
}
