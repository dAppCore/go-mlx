// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The restore-path differ (#66's prescribed instrument, built for #73/#74):
// one live FixedKVCache, snapshotted and restored through BOTH constructors —
// the engine prompt-cache path (snapshotFixedCache -> restoreFixedCacheSnapshot)
// and the conversation WAKE path (snapshotKVCaches -> restoreKVCachesFromSnapshot)
// — then the live structs are diffed field by field and given one real Update
// as the first-forward proxy. The serve's per-turn pipelined degrade fires at
// step 0 on woken conversations; whatever field differs here is the lead.

const (
	diffHeads   = 2
	diffHeadDim = 32
	diffPrefix  = 100
	diffBound   = 24576
)

// diffFill writes n deterministic tokens into the cache via real Updates.
func diffFill(t *testing.T, cache *FixedKVCache, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		k := FromValue(float32(i + 1))
		kb := BroadcastTo(k, []int32{1, diffHeads, 1, diffHeadDim})
		v := FromValue(float32(-(i + 1)))
		vb := BroadcastTo(v, []int32{1, diffHeads, 1, diffHeadDim})
		uk, uv := cache.Update(kb, vb, 1)
		if uk == nil || uv == nil {
			t.Fatalf("update %d returned nil", i)
		}
		if err := Eval(uk, uv); err != nil {
			t.Fatalf("update %d eval: %v", i, err)
		}
		DetachCaches([]Cache{cache})
		Free(k, kb, v, vb)
	}
}

type fixedCacheFacts struct {
	offset, length, maxSize, bandCap int
	shapeCached                      bool
	batch, heads, keyDim, valueDim   int32
	storageDim2                      int
	dtype                            DType
	pendingArmed, pendingViolated    bool
	retired                          int
	hasStorageDType                  bool
}

func factsOf(c *FixedKVCache) fixedCacheFacts {
	f := fixedCacheFacts{
		offset: c.offset, length: c.length, maxSize: c.maxSize, bandCap: c.bandCap,
		shapeCached: c.shapeCached,
		batch:       c.batch, heads: c.heads, keyDim: c.keyDim, valueDim: c.valueDim,
		pendingArmed: c.pendingArmed, pendingViolated: c.pendingViolated,
		retired: len(c.retired), hasStorageDType: c.hasStorageDType,
	}
	if c.keys != nil && c.keys.Valid() {
		f.storageDim2 = c.keys.Dim(2)
		f.dtype = c.keys.Dtype()
	}
	return f
}

func TestFixedCacheRestorePathsAgree(t *testing.T) {
	restore := EngineFeatures{FixedSlidingCache: true, FixedSlidingCacheBound: true}.Apply()
	t.Cleanup(restore)

	// The truth: a live cache grown by real updates.
	live := NewFixedKVCache(diffBound)
	defer FreeCaches([]Cache{live})
	diffFill(t, live, diffPrefix)
	if live.Offset() != diffPrefix {
		t.Fatalf("live offset = %d, want %d", live.Offset(), diffPrefix)
	}
	liveFacts := factsOf(live)
	t.Logf("live    : %+v", liveFacts)

	// Path A — the engine prompt-cache restore.
	snapA, ok, err := snapshotFixedCache(live, diffPrefix)
	if err != nil || !ok {
		t.Fatalf("snapshotFixedCache: ok=%v err=%v", ok, err)
	}
	cacheAny, arrays, err := restoreFixedCacheSnapshot(snapA, diffPrefix, diffPrefix, 0)
	if err != nil {
		t.Fatalf("restoreFixedCacheSnapshot: %v", err)
	}
	if err := Eval(arrays...); err != nil {
		t.Fatalf("path A eval: %v", err)
	}
	Detach(arrays...)
	cacheA := cacheAny.(*FixedKVCache)
	defer FreeCaches([]Cache{cacheA})

	// Path B — the conversation WAKE restore, through the same Model entry the
	// serve uses: capture a KVSnapshot from the live cache, restore it into
	// fresh caches built from the model's templates.
	m := &Model{model: &fakeModel{numLayers: 1, usesFixedCache: true}}
	tokens := make([]int32, diffPrefix)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	kvSnap, err := m.snapshotKVCaches(tokens, []Cache{live})
	if err != nil {
		t.Fatalf("snapshotKVCaches: %v", err)
	}
	wakeCaches, err := m.restoreKVCachesFromSnapshot(kvSnap)
	if err != nil {
		t.Fatalf("restoreKVCachesFromSnapshot: %v", err)
	}
	defer FreeCaches(wakeCaches)
	if len(wakeCaches) != 1 {
		t.Fatalf("wake restored %d caches, want 1", len(wakeCaches))
	}
	cacheB, ok := wakeCaches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("wake cache is %T, want *FixedKVCache", wakeCaches[0])
	}

	// The diff — every divergent field is a finding.
	a, b := factsOf(cacheA), factsOf(cacheB)
	t.Logf("prompt-cache restore: %+v", a)
	t.Logf("wake restore        : %+v", b)
	if a != b {
		t.Errorf("restore paths DISAGREE:\n  prompt-cache: %+v\n  wake:         %+v", a, b)
	}

	// Both must hold the live content.
	wantK, err := CopyCachePrefix(mustState(t, live, 0), diffPrefix)
	if err != nil {
		t.Fatalf("copy live prefix: %v", err)
	}
	gotAK, err := CopyCachePrefix(mustState(t, cacheA, 0), diffPrefix)
	if err != nil {
		t.Fatalf("copy path A prefix: %v", err)
	}
	gotBK, err := CopyCachePrefix(mustState(t, cacheB, 0), diffPrefix)
	if err != nil {
		t.Fatalf("copy path B prefix: %v", err)
	}
	defer freeAll(wantK, gotAK, gotBK)
	assertSameFloats(t, "path A keys", wantK, gotAK)
	assertSameFloats(t, "path B keys", wantK, gotBK)

	// First-forward proxy: one more real Update on each restored cache.
	for name, c := range map[string]*FixedKVCache{"prompt-cache": cacheA, "wake": cacheB} {
		k := FromValue(float32(999))
		kb := BroadcastTo(k, []int32{1, diffHeads, 1, diffHeadDim})
		v := FromValue(float32(-999))
		vb := BroadcastTo(v, []int32{1, diffHeads, 1, diffHeadDim})
		uk, uv := c.Update(kb, vb, 1)
		if uk == nil || uv == nil {
			t.Errorf("%s: post-restore Update returned nil", name)
		} else if err := Eval(uk, uv); err != nil {
			t.Errorf("%s: post-restore Update eval: %v", name, err)
		}
		DetachCaches([]Cache{c})
		Free(k, kb, v, vb)
		if c.Offset() != diffPrefix+1 {
			t.Errorf("%s: post-update offset = %d, want %d", name, c.Offset(), diffPrefix+1)
		}
		if c.pendingViolated {
			t.Errorf("%s: post-update pendingViolated set on an unarmed cache", name)
		}
	}
}

func mustState(t *testing.T, c Cache, idx int) *Array {
	t.Helper()
	state, owned := CacheReadState(c)
	t.Cleanup(func() { Free(owned...) })
	if len(state) <= idx || state[idx] == nil || !state[idx].Valid() {
		t.Fatalf("cache read state %d invalid", idx)
	}
	return state[idx]
}

func freeAll(arrays ...*Array) { Free(arrays...) }

func assertSameFloats(t *testing.T, label string, want, got *Array) {
	t.Helper()
	if err := Eval(want, got); err != nil {
		t.Fatalf("%s eval: %v", label, err)
	}
	w, g := want.Floats(), got.Floats()
	if len(w) != len(g) {
		t.Errorf("%s: length %d vs %d", label, len(w), len(g))
		return
	}
	for i := range w {
		if w[i] != g[i] {
			t.Errorf("%s: first divergence at %d: %v vs %v", label, i, w[i], g[i])
			return
		}
	}
}

// TestFixedCacheRestorePathsAgree_SlidingFullWindow mirrors the serve's
// turn-3+ shape: a window-clamped sliding cache (maxSize = window), restored
// FULL at a logical offset far past the window — the postCap regime's first
// decode step on a woken conversation. The wake lane degrades at step 0 on
// exactly this shape.
func TestFixedCacheRestorePathsAgree_SlidingFullWindow(t *testing.T) {
	restoreGates := EngineFeatures{FixedSlidingCache: true, FixedSlidingCacheBound: true}.Apply()
	t.Cleanup(restoreGates)
	const window = 64
	const logicalOffset = 300 // tokens seen; window keeps the last 64

	live := NewFixedKVCache(window)
	defer FreeCaches([]Cache{live})
	diffFill(t, live, logicalOffset)
	t.Logf("live    : %+v", factsOf(live))
	if live.Offset() != logicalOffset || live.Len() != window {
		t.Fatalf("live offset/len = %d/%d, want %d/%d", live.Offset(), live.Len(), logicalOffset, window)
	}

	// Path A — prompt-cache restore at the full logical offset.
	snapA, ok, err := snapshotFixedCache(live, logicalOffset)
	if err != nil || !ok {
		t.Fatalf("snapshotFixedCache: ok=%v err=%v", ok, err)
	}
	restoreLen := min(snapshotCacheLength(snapA), logicalOffset)
	cacheAny, arrays, err := restoreFixedCacheSnapshot(snapA, restoreLen, logicalOffset, 0)
	if err != nil {
		t.Fatalf("restoreFixedCacheSnapshot: %v", err)
	}
	if err := Eval(arrays...); err != nil {
		t.Fatalf("path A eval: %v", err)
	}
	Detach(arrays...)
	cacheA := cacheAny.(*FixedKVCache)
	defer FreeCaches([]Cache{cacheA})

	// Path B — the wake restore through the Model entry.
	m := &Model{model: &fakeModel{numLayers: 1, usesFixedCache: true}}
	tokens := make([]int32, logicalOffset)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	kvSnap, err := m.snapshotKVCaches(tokens, []Cache{live})
	if err != nil {
		t.Fatalf("snapshotKVCaches: %v", err)
	}
	wakeCaches, err := m.restoreKVCachesFromSnapshot(kvSnap)
	if err != nil {
		t.Fatalf("restoreKVCachesFromSnapshot: %v", err)
	}
	defer FreeCaches(wakeCaches)
	cacheB, ok := wakeCaches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("wake cache is %T, want *FixedKVCache", wakeCaches[0])
	}

	a, b := factsOf(cacheA), factsOf(cacheB)
	t.Logf("prompt-cache restore: %+v", a)
	t.Logf("wake restore        : %+v", b)
	if a != b {
		// Fixed 2026-06-12 (#75): KVLayerSnapshot records the source cache's
		// MaxSize at capture (snapshot v6) and the wake restore prefers it
		// over the wake-era template geometry. A divergence here means the
		// recorded-maxSize plumbing regressed.
		t.Errorf("restore paths DISAGREE:\n  prompt-cache: %+v\n  wake:         %+v", a, b)
	}
	// Compare each against the LIVE truth too — agreeing with each other is
	// not enough if both diverge from the cache they snapshotted.
	liveF := factsOf(live)
	for name, f := range map[string]fixedCacheFacts{"prompt-cache": a, "wake": b} {
		if f.offset != liveF.offset || f.length != liveF.length || f.maxSize != liveF.maxSize {
			t.Errorf("%s diverges from live: offset/len/max = %d/%d/%d, live %d/%d/%d",
				name, f.offset, f.length, f.maxSize, liveF.offset, liveF.length, liveF.maxSize)
		}
	}

	// The postCap first-step inputs the compiled layer needs on a woken cache.
	for name, c := range map[string]*FixedKVCache{"live": live, "prompt-cache": cacheA, "wake": cacheB} {
		if c.Len() < c.MaxSize() {
			t.Errorf("%s: Len %d < MaxSize %d — postCap regime ineligible after restore", name, c.Len(), c.MaxSize())
		}
		shift, last := c.SlidingUpdateInputs()
		if shift == nil || last == nil || !shift.Valid() || !last.Valid() {
			t.Errorf("%s: SlidingUpdateInputs unavailable — compiled postCap declines on this cache", name)
		}
	}
}
