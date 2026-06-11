// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func fixedPendingTestArrays(t *testing.T, seed float32) (*Array, *Array) {
	t.Helper()
	k := FromValues([]float32{seed, seed + 1, seed + 2, seed + 3}, 1, 1, 2, 2)
	v := FromValues([]float32{seed + 4, seed + 5, seed + 6, seed + 7}, 1, 1, 2, 2)
	return k, v
}

// TestFixedKVCache_PendingDiscard proves the speculation contract: an armed
// adoption stages instead of swapping, and a discard leaves the cache —
// storage handles, offset, length — exactly as before the speculated forward.
func TestFixedKVCache_PendingDiscard(t *testing.T) {
	c := NewFixedKVCache(8)
	baseK, baseV := fixedPendingTestArrays(t, 1)
	c.ReplaceFixedFromNativeBorrowed(baseK, baseV, 2) // committed state: offset 2

	if c.Offset() != 2 || c.Keys() != baseK || c.Values() != baseV {
		t.Fatalf("committed state not established: offset=%d", c.Offset())
	}

	stagedK, stagedV := fixedPendingTestArrays(t, 10)
	c.ArmPending()
	state := c.ReplaceFixedFromNativeBorrowed(stagedK, stagedV, 1)
	if state.Keys != stagedK || state.Values != stagedV {
		t.Fatalf("armed adoption must hand consumers the staged arrays")
	}
	if c.Offset() != 2 || c.Keys() != baseK || c.Values() != baseV {
		t.Fatalf("armed adoption mutated committed state: offset=%d", c.Offset())
	}

	c.DiscardPending()
	if c.Offset() != 2 || c.Len() != 2 || c.Keys() != baseK || c.Values() != baseV {
		t.Fatalf("discard changed cache state: offset=%d len=%d", c.Offset(), c.Len())
	}
	if c.PendingViolated() {
		t.Fatalf("clean stage/discard must not flag a violation")
	}
	c.Reset()
}

// TestFixedKVCache_PendingCommit proves a committed stage matches what the
// unarmed adoption would have produced: handles swapped, offset advanced.
func TestFixedKVCache_PendingCommit(t *testing.T) {
	c := NewFixedKVCache(8)
	baseK, baseV := fixedPendingTestArrays(t, 1)
	c.ReplaceFixedFromNativeBorrowed(baseK, baseV, 2)

	stagedK, stagedV := fixedPendingTestArrays(t, 20)
	c.ArmPending()
	c.ReplaceFixedFromNativeBorrowed(stagedK, stagedV, 1)
	c.CommitPending()

	if c.Offset() != 3 || c.Len() != 3 {
		t.Fatalf("commit did not advance: offset=%d len=%d", c.Offset(), c.Len())
	}
	if c.Keys() != stagedK || c.Values() != stagedV {
		t.Fatalf("commit did not adopt the staged arrays")
	}
	c.Reset()
}

// TestFixedKVCache_WriteThroughPending proves the masked-write lane: an armed
// write-through adoption swaps the storage immediately (the write landed at a
// masked index) but defers the offset; discard leaves the visible state — the
// offset and length — untouched, and commit advances it.
func TestFixedKVCache_WriteThroughPending(t *testing.T) {
	c := NewFixedKVCache(8)
	baseK, baseV := fixedPendingTestArrays(t, 1)
	c.ReplaceFixedFromNativeBorrowed(baseK, baseV, 2) // committed: offset 2

	writtenK, writtenV := fixedPendingTestArrays(t, 10)
	c.ArmPending()
	state := c.ReplaceFixedWriteThroughBorrowed(writtenK, writtenV, 1)
	if state.Keys != writtenK || state.Values != writtenV {
		t.Fatalf("write-through must hand consumers the swapped storage")
	}
	if c.Keys() != writtenK || c.Values() != writtenV {
		t.Fatalf("write-through must swap the storage handles immediately")
	}
	if c.Offset() != 2 || c.Len() != 2 {
		t.Fatalf("armed write-through advanced visible state: offset=%d len=%d", c.Offset(), c.Len())
	}

	c.DiscardPending()
	if c.Offset() != 2 || c.Len() != 2 || c.Keys() != writtenK {
		t.Fatalf("discard changed visible state: offset=%d len=%d", c.Offset(), c.Len())
	}

	// Next speculation commits: offset advances, storage already in place.
	nextK, nextV := fixedPendingTestArrays(t, 20)
	c.ArmPending()
	c.ReplaceFixedWriteThroughBorrowed(nextK, nextV, 1)
	c.CommitPending()
	if c.Offset() != 3 || c.Len() != 3 || c.Keys() != nextK {
		t.Fatalf("commit did not advance write-through state: offset=%d len=%d", c.Offset(), c.Len())
	}
	c.Reset()
}

// TestFixedKVCache_BandGrowth proves the stepped-band storage: allocation
// starts at the 1024 floor regardless of the hard cap, grows to the covering
// band on crossing, and carries the committed content across unchanged.
func TestFixedKVCache_BandGrowth(t *testing.T) {
	const cap = 4096
	c := NewFixedKVCache(cap)

	write := func(seq int, seed float32) {
		values := make([]float32, seq*2)
		for i := range values {
			values[i] = seed + float32(i)
		}
		k := FromValues(values, 1, 1, seq, 2)
		v := FromValues(values, 1, 1, seq, 2)
		outK, outV := c.Update(k, v, seq)
		Free(outK, outV, k, v)
	}

	write(1000, 1)
	if c.bandCap != 1024 {
		t.Fatalf("initial band = %d, want the 1024 floor", c.bandCap)
	}
	if c.keys.Dim(2) != 1024 {
		t.Fatalf("storage capacity = %d, want 1024", c.keys.Dim(2))
	}

	// Crossing 1024 grows to 2048 and preserves the committed prefix.
	write(100, 5000)
	if c.bandCap != 2048 || c.keys.Dim(2) != 2048 {
		t.Fatalf("post-crossing band = %d storage = %d, want 2048", c.bandCap, c.keys.Dim(2))
	}
	if c.Offset() != 1100 || c.Len() != 1100 {
		t.Fatalf("growth disturbed counters: offset=%d len=%d", c.Offset(), c.Len())
	}
	k, v := c.validState()
	if err := Eval(k, v); err != nil {
		t.Fatalf("Eval grown state: %v", err)
	}
	got := k.Floats()
	if got[0] != 1 || got[1] != 2 {
		t.Fatalf("grown storage lost the committed prefix: got %v", got[:2])
	}
	if got[2000] != 5000 || got[2001] != 5001 {
		t.Fatalf("grown storage lost the post-crossing write: got %v", got[2000:2002])
	}
	Free(k, v)
	c.Reset()
}

// TestFixedKVCache_PendingViolation proves the degrade signal: a generic
// (mutating) Update while armed flags the violation the pipelined loop uses
// to drop back to serial decode.
func TestFixedKVCache_PendingViolation(t *testing.T) {
	c := NewFixedKVCache(8)
	k := FromValues([]float32{1, 2}, 1, 1, 1, 2)
	v := FromValues([]float32{3, 4}, 1, 1, 1, 2)
	c.EnsureShape(1, 1, 2, 2, DTypeFloat32, DTypeFloat32)
	c.ArmPending()
	outK, outV := c.Update(k, v, 1)
	Free(outK, outV, k, v)
	if !c.PendingViolated() {
		t.Fatalf("generic Update while armed must flag a pending violation")
	}
	c.Reset()
	if c.PendingViolated() {
		t.Fatalf("Reset must clear the violation flag")
	}
}

// TestFixedKVCache_TruncateTo covers the MTP verify rollback: pre-cap offset
// rollback succeeds (linear fill, masked columns become dead storage),
// at-capacity declines (possible rotation), and a staged adoption is
// discarded by the rollback.
func TestFixedKVCache_TruncateTo(t *testing.T) {
	cache := NewFixedKVCache(64)
	k := Zeros([]int32{1, 2, 8, 4}, DTypeFloat32)
	v := Zeros([]int32{1, 2, 8, 4}, DTypeFloat32)
	outK, outV := cache.Update(k, v, 8)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Update eval: %v", err)
	}
	Free(k, v)
	if got := cache.Offset(); got != 8 {
		t.Fatalf("offset after update = %d, want 8", got)
	}
	if !CacheTruncateTo(cache, 5) {
		t.Fatalf("pre-cap truncate declined")
	}
	if got := cache.Offset(); got != 5 {
		t.Fatalf("offset after truncate = %d, want 5", got)
	}
	if CacheTruncateTo(cache, -1) {
		t.Fatalf("negative truncate accepted")
	}
	if !CacheTruncateTo(cache, 7) {
		t.Fatalf("no-op truncate (n past fill) should report true")
	}
	if got := cache.Offset(); got != 5 {
		t.Fatalf("no-op truncate moved the offset to %d", got)
	}
	cache.Reset()

	// At capacity the window may have rotated — must decline to the rebuild.
	full := NewFixedKVCache(8)
	fk := Zeros([]int32{1, 2, 8, 4}, DTypeFloat32)
	fv := Zeros([]int32{1, 2, 8, 4}, DTypeFloat32)
	fOutK, fOutV := full.Update(fk, fv, 8)
	if err := Eval(fOutK, fOutV); err != nil {
		t.Fatalf("full Update eval: %v", err)
	}
	Free(fk, fv)
	if CacheTruncateTo(full, 4) {
		t.Fatalf("at-capacity truncate must decline (possible rotation)")
	}
	full.Reset()
}
