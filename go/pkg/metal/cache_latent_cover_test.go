// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// cache_latent_cover_test.go covers the MLA latent stores' lifecycle hooks the
// decode-correctness test leaves dark: the rotating (bounded) sibling in full,
// plus the nil-latent State / Detach branches of the growing store. Synthetic
// arrays only — no model load.

// TestRotatingLatentKVCache_WindowSlides_Good exercises the bounded MLA latent
// store across the trim boundary: prefill below the cap, then push past it so the
// window slides. Offset stays monotonic while Len saturates at maxSize, and the
// trimmed latent keeps only the most-recent tokens.
func TestRotatingLatentKVCache_WindowSlides_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRotatingLatentKVCache(3)
	if c.Len() != 0 || c.Offset() != 0 {
		t.Fatalf("fresh rotating latent: len=%d offset=%d, want 0/0", c.Len(), c.Offset())
	}
	if c.State() != nil {
		t.Fatalf("fresh rotating latent State() = %v, want nil", c.State())
	}

	// Prefill 2 tokens [B=1, L=2, latentDim=2] — still under the cap of 3.
	a := FromValues([]float32{1, 2, 3, 4}, 1, 2, 2)
	full, _ := c.Update(a, nil, 2)
	if c.Offset() != 2 || c.Len() != 2 {
		t.Fatalf("after prefill: offset=%d len=%d, want 2/2", c.Offset(), c.Len())
	}
	if d := full.Dim(1); d != 2 {
		t.Fatalf("prefill latent seq-dim = %d, want 2", d)
	}

	// Decode 2 more tokens — total 4 > cap 3, the window must slide to the tail.
	b := FromValues([]float32{5, 6, 7, 8}, 1, 2, 2)
	full2, _ := c.Update(b, nil, 2)
	if c.Offset() != 4 {
		t.Fatalf("after decode: offset=%d, want 4 (monotonic)", c.Offset())
	}
	if c.Len() != 3 {
		t.Fatalf("after decode: len=%d, want 3 (saturated at maxSize)", c.Len())
	}
	if d := full2.Dim(1); d != 3 {
		t.Fatalf("trimmed latent seq-dim = %d, want 3 (windowed)", d)
	}
	// Tail = tokens 2,3,4 → rows [3,4],[5,6],[7,8] (token 1's [1,2] dropped).
	got := full2.Floats()
	want := []float32{3, 4, 5, 6, 7, 8}
	if len(got) != len(want) {
		t.Fatalf("windowed latent len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("windowed latent = %v, want %v (oldest token dropped)", got, want)
		}
	}

	if st := c.State(); len(st) != 1 {
		t.Fatalf("State() len = %d, want 1", len(st))
	}
	c.Detach() // graph-parentless copy of the live latent — must not crash
	c.Reset()
	if c.Offset() != 0 || c.State() != nil {
		t.Fatalf("after Reset: offset=%d state=%v, want 0/nil", c.Offset(), c.State())
	}
	// Reset on an already-empty store hits the nil-guard arms.
	c.Reset()
	c.Detach()
}

// TestLatentKVCache_DetachAndNilState_Good covers the growing store's hooks the
// concat test skips: State() and Detach() before any Update (nil latent), then
// Detach() with a live latent.
func TestLatentKVCache_DetachAndNilState_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewLatentKVCache()
	if c.State() != nil {
		t.Fatalf("nil-latent State() = %v, want nil", c.State())
	}
	c.Detach() // nil-latent Detach must be a no-op, not a crash

	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	c.Update(a, nil, 2)
	c.Detach() // live-latent Detach → graph-parentless copy
	if c.Offset() != 2 {
		t.Fatalf("offset after detach = %d, want 2 (detach preserves state)", c.Offset())
	}
	c.Reset()
}
