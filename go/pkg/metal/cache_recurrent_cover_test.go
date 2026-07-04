// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// cache_recurrent_cover_test.go covers the recurrent holder's lifecycle hooks the
// round-trip test leaves dark: State() (the prompt-cache/snapshot accessor, a
// distinct method from RecurrentState), Detach() across both the empty and live
// arms, and the in-place-reuse branch of SetRecurrentState where the mixer hands
// the same handle straight back. Synthetic state slots only.

func TestRecurrentCache_StateAccessorAndDetach_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRecurrentCache().(*recurrentCache)
	// Empty holder: State() mirrors the nil zero-state, Detach is a no-op.
	if c.State() != nil {
		t.Fatalf("empty holder State() = %v, want nil", c.State())
	}
	c.Detach()

	s0 := Zeros([]int32{1, 2, 4, 4}, DTypeFloat32)
	c.SetRecurrentState([]*Array{s0})
	st := c.State()
	if len(st) != 1 || st[0] != s0 {
		t.Fatalf("State() = %v, want [s0] (same slots as RecurrentState)", st)
	}
	c.Detach() // live-state Detach → graph-parentless copies, must not crash
	// Detach must preserve the stored slots for the next forward.
	if got := c.RecurrentState(); len(got) != 1 {
		t.Fatalf("after Detach, RecurrentState len = %d, want 1", len(got))
	}
	c.Reset()
}

// SetRecurrentState must leave a handle in place (not free it) when the mixer
// re-uses the same array across forwards — an in-place recurrence hands the prior
// state straight back. Here slot s0 is re-used while a sibling slot is replaced,
// so s0 stays valid and only the replaced sibling is freed.
func TestRecurrentCache_SetReusedHandleStaysValid_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRecurrentCache().(*recurrentCache)
	s0 := Zeros([]int32{1, 1, 2, 2}, DTypeFloat32) // re-used in place
	s1 := Zeros([]int32{1, 1, 2, 2}, DTypeFloat32) // replaced next round
	c.SetRecurrentState([]*Array{s0, s1})

	s2 := Zeros([]int32{1, 1, 2, 2}, DTypeFloat32)
	c.SetRecurrentState([]*Array{s0, s2}) // s0 re-used, s1 superseded
	if !s0.Valid() {
		t.Error("re-used state slot s0 was wrongly freed")
	}
	if s1.Valid() {
		t.Error("superseded slot s1 was not freed")
	}
	if got := c.RecurrentState(); len(got) != 2 || got[0] != s0 || got[1] != s2 {
		t.Fatalf("state = %v, want [s0 s2]", got)
	}
	c.Reset()
}
