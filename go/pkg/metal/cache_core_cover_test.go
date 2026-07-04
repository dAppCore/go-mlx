// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// cache_core_cover_test.go covers the core KV/Fixed/Paged/Rotating cache state
// accessors and the package-level state-append helpers the decode-driven tests
// leave partly dark: AppendState across the concrete types, the empty-cache
// early-outs, the State() fallback in appendCacheState, the rotating ordered
// view, and RepeatPagedState's repeat loop. Synthetic K/V via makeKV.

func TestCache_FixedKVCacheStateAccessors_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewFixedKVCache(512)
	// Empty cache: every accessor reports the empty shape without crashing.
	if c.State() != nil {
		t.Error("empty fixed State() should be nil")
	}
	if got := c.AppendState(nil); got != nil {
		t.Errorf("empty fixed AppendState = %v, want nil", got)
	}
	k0, v0 := c.ReadState()
	if k0 != nil || v0 != nil {
		t.Error("empty fixed ReadState should be nil/nil")
	}

	k, v := makeKV(3)
	c.Update(k, v, 3)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	if st := c.State(); len(st) != 2 {
		t.Fatalf("populated fixed State() len = %d, want 2", len(st))
	}
	if got := c.AppendState(nil); len(got) != 2 {
		t.Fatalf("populated fixed AppendState len = %d, want 2", len(got))
	}
	// An unarmed cache with no staged pending write appends nothing.
	if got := c.AppendPendingState(nil); got != nil {
		t.Errorf("unarmed AppendPendingState = %v, want nil (no pending)", got)
	}
	rk, rv := c.ReadState()
	if rk == nil || rv == nil {
		t.Error("populated fixed ReadState returned nil")
	}

	// TruncateTo within the linear pre-cap fill keeps the cache valid.
	if !c.TruncateTo(2) {
		t.Error("TruncateTo(2) within fill returned false")
	}
	if !c.TruncateTo(100) {
		t.Error("TruncateTo beyond Len should be a no-op success")
	}
	if c.TruncateTo(-1) {
		t.Error("TruncateTo(-1) should return false")
	}
}

func TestCache_PagedAppendState_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewPagedKVCache(0, 4)
	// Empty paged cache: AppendState early-outs to dst unchanged.
	if got := c.AppendState(nil); got != nil {
		t.Errorf("empty paged AppendState = %v, want nil", got)
	}

	k, v := makeKV(2)
	c.UpdatePages(k, v, 2).Free()
	Free(k, v)
	defer FreeCaches([]Cache{c})

	if got := c.AppendState(nil); len(got) == 0 {
		t.Error("populated paged AppendState returned no pages")
	}
	if got := c.AppendDirtyState(nil); len(got) == 0 {
		t.Error("populated paged AppendDirtyState returned no pages")
	}
}

// appendCacheState prefers AppendState where implemented and falls back to
// State() for caches that don't (recurrentCache has no AppendState). A nil cache
// returns dst unchanged.
func TestCache_AppendCacheState_FallbackAndNil_Good(t *testing.T) {
	requireMetalRuntime(t)

	if got := appendCacheState(nil, nil); got != nil {
		t.Errorf("appendCacheState(nil, nil) = %v, want nil", got)
	}

	// recurrentCache implements State() but not stateAppender → the fallback
	// loop runs over its State() slots.
	rc := NewRecurrentCache()
	s0 := Zeros([]int32{1, 1, 2, 2}, DTypeFloat32)
	rc.SetRecurrentState([]*Array{s0})
	got := appendCacheState(nil, rc)
	if len(got) != 1 || got[0] != s0 {
		t.Fatalf("recurrent fallback append = %v, want [s0]", got)
	}
	rc.Reset()

	// A populated KVCache takes the AppendState fast path.
	kv := NewKVCache()
	k, v := makeKV(2)
	kv.Update(k, v, 2)
	Free(k, v)
	defer FreeCaches([]Cache{kv})
	if appended := appendCacheState(nil, kv); len(appended) != 2 {
		t.Fatalf("KVCache append len = %d, want 2", len(appended))
	}
}

// RotatingKVCache.orderedState returns nil when empty and the windowed leading
// slice once populated.
func TestCache_RotatingOrderedState_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRotatingKVCache(512)
	if c.orderedState() != nil {
		t.Error("empty rotating orderedState should be nil")
	}

	k, v := makeKV(4)
	c.Update(k, v, 4)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	ordered := c.orderedState()
	if len(ordered) != 2 {
		t.Fatalf("rotating orderedState len = %d, want 2", len(ordered))
	}
	Materialize(ordered...)
	// The ordered key view's seq dim must equal the visible window length.
	if ordered[0].Dim(2) != c.Len() {
		t.Errorf("ordered key seq-dim = %d, want %d", ordered[0].Dim(2), c.Len())
	}
	Free(ordered...)
}

// RepeatPagedState returns the originals untouched for factor<=1 and materialised
// GQA-repeated pages for factor>1.
func TestCache_RepeatPagedState_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Build a one-page paged state [B=1, H=2, L=2, D=4].
	c := NewPagedKVCache(0, 4)
	k, v := makeKV(2)
	c.UpdatePages(k, v, 2).Free()
	Free(k, v)
	defer FreeCaches([]Cache{c})
	state := c.PageState()
	defer state.Free()

	// factor 1 → originals, no owned arrays to free.
	k1, v1, owned1 := RepeatPagedState(state, 1)
	if owned1 != nil {
		t.Error("factor 1 should own nothing")
	}
	if len(k1) != len(state.Keys) || len(v1) != len(state.Values) {
		t.Error("factor 1 should pass the originals through")
	}

	// factor 2 → each page repeated along the head axis, all owned.
	k2, v2, owned2 := RepeatPagedState(state, 2)
	defer Free(owned2...)
	if len(owned2) != len(state.Keys)+len(state.Values) {
		t.Fatalf("factor 2 owned %d arrays, want %d", len(owned2), len(state.Keys)+len(state.Values))
	}
	Materialize(k2...)
	Materialize(v2...)
	// Repeated key page doubles the head dimension (axis 1).
	if k2[0].Dim(1) != 4 {
		t.Errorf("repeated key head-dim = %d, want 4 (2 heads x factor 2)", k2[0].Dim(1))
	}
}
