// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestKVCache_Accessors_Good(t *testing.T) {
	c := &KVCache{offset: 7, step: 256}
	if got := c.Offset(); got != 7 {
		t.Fatalf("Offset() = %d, want 7", got)
	}
	if got := c.Step(); got != 256 {
		t.Fatalf("Step() = %d, want 256", got)
	}
}

func TestRotatingKVCache_Accessors_Good(t *testing.T) {
	c := &RotatingKVCache{maxSize: 1024}
	if got := c.MaxSize(); got != 1024 {
		t.Fatalf("MaxSize() = %d, want 1024", got)
	}
}

func TestFixedKVCache_Accessors_Good(t *testing.T) {
	c := &FixedKVCache{maxSize: 512}
	if got := c.MaxSize(); got != 512 {
		t.Fatalf("MaxSize() = %d, want 512", got)
	}
}

func TestPagedKVCache_Accessors_Good(t *testing.T) {
	c := &PagedKVCache{maxSize: 4096, pageSize: 256}
	if got := c.MaxSize(); got != 4096 {
		t.Fatalf("MaxSize() = %d, want 4096", got)
	}
	if got := c.PageSize(); got != 256 {
		t.Fatalf("PageSize() = %d, want 256", got)
	}
}

func TestQuantizedKVCache_Accessors_Good(t *testing.T) {
	c := &QuantizedKVCache{maxSize: 2048, step: 256, keyBits: 8, valueBits: 4}
	if got := c.MaxSize(); got != 2048 {
		t.Fatalf("MaxSize() = %d, want 2048", got)
	}
	if got := c.Step(); got != 256 {
		t.Fatalf("Step() = %d, want 256", got)
	}
	k, v := c.Bits()
	if k != 8 {
		t.Fatalf("Bits() key = %d, want 8", k)
	}
	if v != 4 {
		t.Fatalf("Bits() value = %d, want 4", v)
	}
}

// TestKVCache_StepDefault_Good: NewKVCache seeds the 256-token chunk size, and a
// freshly built cache exposes no key/value tensors yet (pure-Go field reads).
func TestKVCache_StepDefault_Good(t *testing.T) {
	c := NewKVCache()
	if got := c.Step(); got != 256 {
		t.Errorf("NewKVCache().Step() = %d, want 256", got)
	}
	if c.Keys() != nil {
		t.Error("Keys() before first Update = non-nil, want nil")
	}
	if c.Values() != nil {
		t.Error("Values() before first Update = non-nil, want nil")
	}
}

// TestRotatingKVCache_StepDefault_Good: the rotating cache's growth-chunk default
// and its pre-Update accessors (raw key/value tensors are nil until the first
// Update fills the ring buffer).
func TestRotatingKVCache_StepDefault_Good(t *testing.T) {
	c := NewRotatingKVCache(1024)
	if got := c.Step(); got <= 0 {
		t.Errorf("NewRotatingKVCache().Step() = %d, want > 0", got)
	}
	if c.Keys() != nil || c.Values() != nil {
		t.Error("rotating Keys()/Values() before first Update = non-nil, want nil")
	}
}

// TestKVCache_KeysValues_Good: after a real Update the raw-tensor accessors expose
// the backing storage (not a copy) — the [B,H,L,D] buffer the cache grew, with L
// at least the tokens written. Needs a Metal device for the slice-update ops.
func TestKVCache_KeysValues_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewKVCache()
	k, v := makeKV(3)
	outK, outV := c.Update(k, v, 3)
	Materialize(outK, outV)

	rawK, rawV := c.Keys(), c.Values()
	if rawK == nil || rawV == nil {
		t.Fatalf("Keys()/Values() after Update = (%v,%v), want backing tensors", rawK, rawV)
	}
	// Buffer is [1,2,>=3,4]: the pre-allocated step may exceed the 3 written.
	ks := rawK.Shape()
	if len(ks) != 4 || ks[0] != 1 || ks[1] != 2 || ks[3] != 4 || ks[2] < 3 {
		t.Errorf("Keys() shape = %v, want [1 2 >=3 4]", ks)
	}
	if !rawK.Valid() || !rawV.Valid() {
		t.Error("backing tensors should be valid after Update")
	}
}

// TestRotatingKVCache_KeysValues_Good: the rotating ring also surfaces its raw
// key/value buffers once an Update has populated them.
func TestRotatingKVCache_KeysValues_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRotatingKVCache(64)
	k, v := makeKV(4)
	outK, outV := c.Update(k, v, 4)
	Materialize(outK, outV)

	if c.Keys() == nil || c.Values() == nil {
		t.Fatal("rotating Keys()/Values() after Update = nil, want backing tensors")
	}
	if !c.Keys().Valid() || !c.Values().Valid() {
		t.Error("rotating backing tensors should be valid after Update")
	}
}

// TestCachePaged_KPagesVPages_Good: a paged cache exposes its per-block key/value
// page tensors after an Update writes the first page. Before any Update the page
// slices are nil (pure-Go), and the post-Update slices are equal-length and valid.
func TestCachePaged_KPagesVPages_Good(t *testing.T) {
	c := NewPagedKVCache(1024, 16)
	// Pre-Update: no pages allocated yet.
	if c.KPages() != nil || c.VPages() != nil {
		t.Error("KPages()/VPages() before first Update = non-nil, want nil")
	}

	requireMetalRuntime(t)
	k, v := makeKV(4)
	outK, outV := c.Update(k, v, 4)
	Materialize(outK, outV)

	kp, vp := c.KPages(), c.VPages()
	if len(kp) == 0 || len(kp) != len(vp) {
		t.Fatalf("KPages()=%d VPages()=%d after Update, want equal and non-empty", len(kp), len(vp))
	}
	for i := range kp {
		if kp[i] == nil || !kp[i].Valid() || vp[i] == nil || !vp[i].Valid() {
			t.Errorf("page %d not a valid backing tensor", i)
		}
	}
}

// TestCacheFixed_EnsureDecodeCapacity_Good: on a storage-less fixed cache the band
// grower is a defined no-op (nothing to grow until the first Update allocates the
// committed buffer), and the invalid seqLen guard rejects a sub-1 request. Both
// are pure-Go early returns — no Metal device required.
func TestCacheFixed_EnsureDecodeCapacity_Good(t *testing.T) {
	c := NewFixedKVCache(256)
	// keys==nil → EnsureDecodeCapacityFor returns before reading any dims.
	c.EnsureDecodeCapacity()
	c.EnsureDecodeCapacityFor(4)
	// seqLen<1 is rejected even once storage exists; on an empty cache it is the
	// same nil-guard return. Neither call may mutate the counters.
	c.EnsureDecodeCapacityFor(0)
	if c.Offset() != 0 || c.Len() != 0 {
		t.Fatalf("EnsureDecodeCapacity mutated empty cache to off=%d len=%d, want 0/0", c.Offset(), c.Len())
	}
}
