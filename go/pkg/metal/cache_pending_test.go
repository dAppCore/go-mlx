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
