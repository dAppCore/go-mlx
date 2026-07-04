// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// decode_fast_cover_test.go covers the decode replay offset plumbing, the
// multi-token mask/cache-update fast ops, and the uncompiled-MLP gemm preference
// setter — all reachable with synthetic arrays / scalars, no model.

func TestDecodeReplay_SharedOffsetRoundTrip_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Probe gate is read-only; just exercise the accessor.
	_ = lthnReplayProbeEnabled()

	// Default: no shared offset installed.
	if LthnReplayOffset() != nil {
		t.Fatal("a shared replay offset was installed before SetLthnReplayOffset")
	}

	off := FromValues([]int32{5}, 1)
	SetLthnReplayOffset(off)
	if LthnReplayOffset() != off {
		t.Fatal("LthnReplayOffset did not return the installed array")
	}

	// Overwrite the offset buffer in place, then read it back.
	lthnWriteOffset(off, 9)
	Materialize(off)
	if got := off.Int(); got != 9 {
		t.Errorf("offset after lthnWriteOffset = %d, want 9", got)
	}

	// Clear the install so the global doesn't leak into other tests.
	SetLthnReplayOffset(nil)
	if LthnReplayOffset() != nil {
		t.Fatal("SetLthnReplayOffset(nil) did not clear the shared offset")
	}
	Free(off)
}

func TestDecodeReplay_ArrayWriteFloats_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValues([]float32{1, 2, 3, 4}, 4)
	defer Free(a)
	// Empty write is a no-op.
	lthnArrayWriteFloats(a, nil)
	// Overwrite all four slots.
	lthnArrayWriteFloats(a, []float32{10, 20, 30, 40})
	Materialize(a)
	got := a.Floats()
	want := []float32{10, 20, 30, 40}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("after write = %v, want %v", got, want)
		}
	}
}

// MultiTokenCausalMask reduces to the single-token mask for seqLen<=1 and builds
// a [1,1,seqLen,capacity] additive mask for seqLen>1.
func TestFast_MultiTokenCausalMask_Good(t *testing.T) {
	requireMetalRuntime(t)

	offset := FromValues([]int32{0}, 1)
	defer Free(offset)

	// seqLen 1 → single-token mask path.
	single := MultiTokenCausalMask(4, offset, 1)
	Materialize(single)
	Free(single)

	// seqLen 3 over capacity 4 → multi-token mask, shape [1,1,3,4].
	multi := MultiTokenCausalMask(4, offset, 3)
	Materialize(multi)
	if multi.NumDims() != 4 || multi.Dim(2) != 3 || multi.Dim(3) != 4 {
		t.Fatalf("multi-token mask shape = %v, want [1 1 3 4]", multi.Shape())
	}
	// Row 0 attends only column 0 (offset 0): col 0 = 0, col 1 = -inf.
	vals := multi.Floats()
	if vals[0] != 0 {
		t.Errorf("mask[0,0] = %v, want 0 (attend)", vals[0])
	}
	if vals[1] >= 0 {
		t.Errorf("mask[0,1] = %v, want very negative (masked)", vals[1])
	}
	Free(multi)
}

// MultiTokenCacheUpdate writes a seqLen-token block into the cache at the offset
// position; seqLen<=1 reduces to the single-token update.
func TestFast_MultiTokenCacheUpdate_Good(t *testing.T) {
	requireMetalRuntime(t)

	offset := FromValues([]int32{1}, 1)
	defer Free(offset)

	// cache [1,1,4,2] zeros; write a 2-token block at columns 1..2.
	cache := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	tokens := FromValues([]float32{7, 8, 9, 10}, 1, 1, 2, 2)
	updated := MultiTokenCacheUpdate(cache, tokens, offset, 2)
	Materialize(updated)
	if updated.Dim(2) != 4 {
		t.Fatalf("updated cache seq-dim = %d, want 4", updated.Dim(2))
	}
	Free(cache, tokens, updated)

	// seqLen 1 reduces to the single-token path.
	cache2 := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	tok1 := FromValues([]float32{1, 2}, 1, 1, 1, 2)
	upd2 := MultiTokenCacheUpdate(cache2, tok1, offset, 1)
	Materialize(upd2)
	Free(cache2, tok1, upd2)
}

func TestDenseMatvec_SetUncompiledMLPPreferGemm_Good(t *testing.T) {
	// Atomic setter — flip both ways. Read-back is via the package's own use;
	// here we just prove the setter runs without panic and restore the default.
	SetUncompiledMLPPreferGemm(true)
	SetUncompiledMLPPreferGemm(false)
}
