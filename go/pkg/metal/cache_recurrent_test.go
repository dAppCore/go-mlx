// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// The holder starts empty (zero-state), stores the mixer's opaque state, and
// frees the superseded slot when the mixer hands back fresh state.
func TestRecurrentCache_StateRoundTrip_Good(t *testing.T) {
	c := NewRecurrentCache()
	if got := c.RecurrentState(); got != nil {
		t.Fatalf("fresh holder state = %v, want nil (zero-state start)", got)
	}

	s0 := Zeros([]int32{1, 2, 4, 4}, DTypeFloat32)
	c.SetRecurrentState([]*Array{s0})
	if got := c.RecurrentState(); len(got) != 1 || got[0] != s0 {
		t.Fatalf("after Set, state = %v, want [s0]", got)
	}

	// Replacing with a fresh array adopts the new slot and frees the old.
	s1 := Zeros([]int32{1, 2, 4, 4}, DTypeFloat32)
	c.SetRecurrentState([]*Array{s1})
	if got := c.RecurrentState(); len(got) != 1 || got[0] != s1 {
		t.Fatalf("after replace, state = %v, want [s1]", got)
	}
	if s0.Valid() {
		t.Error("old state slot was not freed on replace")
	}
}

// Update appends no K/V (recurrent layers contribute none), only advances the
// position; Reset clears both state and offset.
func TestRecurrentCache_UpdateNoKV_Ugly(t *testing.T) {
	c := NewRecurrentCache()
	k, v := c.Update(nil, nil, 5)
	if k != nil || v != nil {
		t.Errorf("recurrent Update returned K/V %v,%v, want nil,nil", k, v)
	}
	if c.Offset() != 5 || c.Len() != 5 {
		t.Errorf("offset/len = %d/%d, want 5/5", c.Offset(), c.Len())
	}

	c.SetRecurrentState([]*Array{Zeros([]int32{1, 1, 2, 2}, DTypeFloat32)})
	c.Reset()
	if c.RecurrentState() != nil || c.Offset() != 0 {
		t.Errorf("after Reset state=%v offset=%d, want nil/0", c.RecurrentState(), c.Offset())
	}
}

// The "recurrent" scheme now carries compute (the holder) and Serves
// StateRecurrent, so Compatible pairs it with a recurrent mixer and the engine
// builds a RecurrentCache for it.
func TestRecurrentScheme_Compatible_Good(t *testing.T) {
	cc, ok := CacheComputeFor("recurrent")
	if !ok {
		t.Fatal("recurrent cache compute did not resolve")
	}
	if cc.Serves() != scheme.StateRecurrent {
		t.Fatalf("recurrent serves %v, want recurrent", cc.Serves())
	}
	if _, isRecurrent := cc.NewCache(CacheParams{}).(RecurrentCache); !isRecurrent {
		t.Error("recurrent scheme NewCache did not build a RecurrentCache")
	}
}
