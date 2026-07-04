// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// kvFactoryFakeMixer is a minimal scheme.Mixer for the factory's mode/build
// decisions — construction + registry resolution only, no Metal runtime.
type kvFactoryFakeMixer struct{ state scheme.StateKind }

func (kvFactoryFakeMixer) Kind() string              { return "kvfactory-fake" }
func (f kvFactoryFakeMixer) State() scheme.StateKind { return f.state }

// kvFactoryFakeModer additionally names a specific cache scheme — the MLA shape.
type kvFactoryFakeModer struct {
	kvFactoryFakeMixer
	mode string
}

func (f kvFactoryFakeModer) CacheMode() string { return f.mode }

// TestCacheModeForMixer_Good: a mixer with no CacheModer gets the default scheme
// for its StateKind; a mixer that names one gets that name.
func TestCacheModeForMixer_Good(t *testing.T) {
	cases := []struct {
		name string
		m    scheme.Mixer
		want string
	}{
		{"kv-default", kvFactoryFakeMixer{scheme.StateKVCache}, cacheModeDefault},
		{"recurrent", kvFactoryFakeMixer{scheme.StateRecurrent}, cacheModeRecurrent},
		{"named-mla", kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, "mla-latent"}, "mla-latent"},
	}
	for _, c := range cases {
		if got := CacheModeForMixer(c.m); got != c.want {
			t.Errorf("%s: CacheModeForMixer = %q, want %q", c.name, got, c.want)
		}
	}
}

// TestCacheModeForMixer_Bad: an empty CacheMode() falls through to the StateKind
// default rather than naming the empty mode (which would never resolve).
func TestCacheModeForMixer_Bad(t *testing.T) {
	m := kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, ""}
	if got := CacheModeForMixer(m); got != cacheModeDefault {
		t.Errorf("empty CacheMode() = %q, want fallthrough to %q", got, cacheModeDefault)
	}
}

// TestNewCacheForMode_Good: registered compute-bearing schemes build a cache.
func TestNewCacheForMode_Good(t *testing.T) {
	for _, mode := range []string{"mla-latent", "recurrent"} {
		c, ok := NewCacheForMode(mode, CacheParams{})
		if !ok || c == nil {
			t.Fatalf("NewCacheForMode(%q) = (%v, %v), want a cache", mode, c, ok)
		}
	}
}

// TestNewCacheForMode_Bad: an unregistered mode resolves to (nil, false).
func TestNewCacheForMode_Bad(t *testing.T) {
	if c, ok := NewCacheForMode("no-such-cache-mode", CacheParams{}); ok || c != nil {
		t.Fatalf("NewCacheForMode(unknown) = (%v, %v), want (nil, false)", c, ok)
	}
}

// TestNewCacheForMixer_Good: the factory builds a non-nil cache for each shape —
// the MLA-moder its latent store, a plain KV mixer the default, a recurrent
// mixer its holder.
func TestNewCacheForMixer_Good(t *testing.T) {
	mixers := []scheme.Mixer{
		kvFactoryFakeModer{kvFactoryFakeMixer{scheme.StateKVCache}, "mla-latent"},
		kvFactoryFakeMixer{scheme.StateKVCache},
		kvFactoryFakeMixer{scheme.StateRecurrent},
	}
	for _, m := range mixers {
		if c := NewCacheForMixer(m, CacheParams{}); c == nil {
			t.Fatalf("NewCacheForMixer(%q) = nil, want a cache", m.Kind())
		}
	}
}

// TestCacheFixed_NewFixedKVCacheAtOffset_Good: the restore-position constructor
// seeds offset+length without touching any tensor storage, so the counters read
// back exactly as supplied (the deserialised-session restore path).
func TestCacheFixed_NewFixedKVCacheAtOffset_Good(t *testing.T) {
	c := NewFixedKVCacheAtOffset(512, 37, 33)
	if got := c.MaxSize(); got != 512 {
		t.Errorf("MaxSize() = %d, want 512", got)
	}
	if got := c.Offset(); got != 37 {
		t.Errorf("Offset() = %d, want 37", got)
	}
	if got := c.Len(); got != 33 {
		t.Errorf("Len() = %d, want 33", got)
	}
	// No storage was allocated, so State() reports the empty cache.
	if s := c.State(); s != nil {
		t.Errorf("State() = %v on storage-less restore, want nil", s)
	}
}

// TestCache_CachesTruncateTo_Good: a nil-storage cache always truncates in place
// (Len()<=n is trivially satisfiable), so a slice of fresh caches all succeed.
func TestCache_CachesTruncateTo_Good(t *testing.T) {
	caches := []Cache{NewKVCache(), NewFixedKVCache(256), NewKVCache()}
	if !CachesTruncateTo(caches, 4) {
		t.Fatal("CachesTruncateTo over empty caches = false, want true")
	}
	// An empty slice is vacuously all-succeed.
	if !CachesTruncateTo(nil, 0) {
		t.Fatal("CachesTruncateTo(nil) = false, want true")
	}
}

// TestCache_CachesTruncateTo_Bad: a cache that cannot truncate in place fails the
// batch, and one failure must poison the whole result so the caller rebuilds.
func TestCache_CachesTruncateTo_Bad(t *testing.T) {
	// A nil Cache entry can never truncate (CacheTruncateTo(nil)=false), so a
	// slice containing one reports overall failure even alongside good caches.
	caches := []Cache{NewKVCache(), nil}
	if CachesTruncateTo(caches, 0) {
		t.Fatal("CachesTruncateTo with a nil entry = true, want false")
	}
}

// TestCache_CacheTruncateTo_Bad: the single-cache guard rejects a nil cache and a
// negative target outright (no TruncateTo dispatch on either).
func TestCache_CacheTruncateTo_Bad(t *testing.T) {
	if CacheTruncateTo(nil, 4) {
		t.Error("CacheTruncateTo(nil, 4) = true, want false")
	}
	if CacheTruncateTo(NewKVCache(), -1) {
		t.Error("CacheTruncateTo(cache, -1) = true, want false")
	}
}

// TestCachePaged_RepeatPagedState_Bad: a repeat factor of 1 (or less) is a no-op —
// the original page slices pass straight through and no owned arrays are made, so
// the GQA repeat is skipped without allocating.
func TestCachePaged_RepeatPagedState_Bad(t *testing.T) {
	state := PagedKVState{
		Keys:   []*Array{{}, {}},
		Values: []*Array{{}},
		Length: 3,
	}
	keys, values, owned := RepeatPagedState(state, 1)
	if len(keys) != 2 || len(values) != 1 {
		t.Fatalf("RepeatPagedState(factor=1) keys=%d values=%d, want 2/1 (pass-through)", len(keys), len(values))
	}
	if owned != nil {
		t.Fatalf("RepeatPagedState(factor=1) owned = %v, want nil (nothing materialised)", owned)
	}
	// factor 0 takes the same no-op branch.
	if _, _, owned := RepeatPagedState(state, 0); owned != nil {
		t.Fatalf("RepeatPagedState(factor=0) owned = %v, want nil", owned)
	}
}

// TestCachePaged_PagedStateNeedsMaterializedRepeat_Good: the pure-Go predicate that
// decides whether a GQA expand needs a real RepeatKV. It walks the factor, the
// K/V length agreement, and each page's validity/rank/head-dim, covering every
// branch with synthetic page handles (no tensor data, no Metal).
func TestCachePaged_PagedStateNeedsMaterializedRepeat_Good(t *testing.T) {
	// factor<=1 → never needs materialising.
	if PagedStateNeedsMaterializedRepeat(PagedKVState{}, 1) {
		t.Error("factor=1 needs materialise = true, want false")
	}
	// No pages → nothing to repeat.
	if PagedStateNeedsMaterializedRepeat(PagedKVState{}, 2) {
		t.Error("empty pages needs materialise = true, want false")
	}
	// Mismatched K/V page counts → declines (cannot pair them).
	mismatch := PagedKVState{Keys: []*Array{{}}, Values: []*Array{{}, {}}}
	if PagedStateNeedsMaterializedRepeat(mismatch, 2) {
		t.Error("mismatched K/V counts needs materialise = true, want false")
	}
	// A nil/invalid page forces the materialised path (cannot broadcast it).
	withNil := PagedKVState{Keys: []*Array{nil}, Values: []*Array{nil}}
	if !PagedStateNeedsMaterializedRepeat(withNil, 2) {
		t.Error("nil page needs materialise = false, want true")
	}
}

// TestCacheFixed_storageKV_Bad: without a configured storage dtype the helper is a
// pure pass-through — k, v hand straight back and no owned conversions are made,
// so it is safe to invoke with nil tensors (the no-conversion decode path).
func TestCacheFixed_storageKV_Bad(t *testing.T) {
	c := NewFixedKVCache(128) // hasStorageDType defaults false
	k, v, owned := c.storageKV(nil, nil)
	if k != nil || v != nil || owned != nil {
		t.Fatalf("storageKV pass-through = (%v,%v,%v), want all nil", k, v, owned)
	}
	// A nil receiver is also a defined pass-through (defensive guard).
	var nc *FixedKVCache
	if _, _, owned := nc.storageKV(nil, nil); owned != nil {
		t.Fatalf("nil-receiver storageKV owned = %v, want nil", owned)
	}
}

// TestCachePaged_storageKV_Bad: the paged cache shares the same dtype-passthrough
// guard — no storage dtype means the inputs return untouched.
func TestCachePaged_storageKV_Bad(t *testing.T) {
	c := NewPagedKVCache(1024, 128)
	k, v, owned := c.storageKV(nil, nil)
	if k != nil || v != nil || owned != nil {
		t.Fatalf("paged storageKV pass-through = (%v,%v,%v), want all nil", k, v, owned)
	}
}

// TestCache_cacheStorageKV_Bad: a dtype with no byte width (an unmapped sentinel)
// short-circuits before any AsType — inputs return unchanged with no owned arrays.
func TestCache_cacheStorageKV_Bad(t *testing.T) {
	if DTypeByteSize(DType(99)) != 0 {
		t.Skip("sentinel DType(99) unexpectedly has a byte width")
	}
	k, v, owned := cacheStorageKV(nil, nil, DType(99))
	if k != nil || v != nil || owned != nil {
		t.Fatalf("cacheStorageKV(zero-width dtype) = (%v,%v,%v), want unchanged + nil owned", k, v, owned)
	}
	// A valid dtype but nil/invalid inputs skips conversion (the .Valid() guard),
	// returning an empty owned set rather than touching Metal.
	if _, _, owned := cacheStorageKV(nil, nil, DTypeFloat16); len(owned) != 0 {
		t.Fatalf("cacheStorageKV(nil inputs) owned = %v, want empty", owned)
	}
}

// TestCacheFixed_AppendStateEmpty_Good: on a storage-less fixed cache the three
// state appenders are all defined no-ops — they return dst untouched (no pending
// arrays, no committed storage to surface).
func TestCacheFixed_AppendStateEmpty_Good(t *testing.T) {
	c := NewFixedKVCache(256)
	seed := []*Array{{}}
	if got := c.AppendState(seed); len(got) != 1 {
		t.Errorf("AppendState on empty cache grew dst to %d, want 1 (unchanged)", len(got))
	}
	if got := c.AppendPendingState(seed); len(got) != 1 {
		t.Errorf("AppendPendingState on empty cache grew dst to %d, want 1", len(got))
	}
	// RetireAfterNextEval with no arrays is a no-op and must not panic.
	c.RetireAfterNextEval()
}

// TestCachePaged_AppendStateEmpty_Good: a paged cache with no pages appends nothing.
func TestCachePaged_AppendStateEmpty_Good(t *testing.T) {
	c := NewPagedKVCache(1024, 128)
	seed := []*Array{{}}
	if got := c.AppendState(seed); len(got) != 1 {
		t.Errorf("paged AppendState on empty cache grew dst to %d, want 1", len(got))
	}
	// Detach on a paged cache is a documented no-op (page views are reused across
	// decode steps); it must run without panicking.
	c.Detach()
}
