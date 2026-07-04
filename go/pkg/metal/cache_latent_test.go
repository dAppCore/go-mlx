// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
)

// TestMLALatentScheme_BuildsLatentCache_Good proves the mla-latent scheme now
// builds the dedicated single-tensor store via the KV factory front-door — not
// the two-tensor KVCache it used to alias. No Metal runtime: registry + type.
func TestMLALatentScheme_BuildsLatentCache_Good(t *testing.T) {
	c, ok := NewCacheForMode(string(KVCacheModeMLALatent), CacheParams{})
	if !ok {
		t.Fatal("mla-latent scheme not registered with the cache factory")
	}
	if _, isLatent := c.(*latentKVCache); !isLatent {
		t.Fatalf("mla-latent built %T, want *latentKVCache", c)
	}
}

// TestLatentKVCache_ConcatAcrossChunks_Good is the decode-correctness contract:
// the latent store concatenates a single tensor across chunks along the sequence
// axis and tracks the token offset. Prefill a 2-token latent [1,2,3], then decode
// one more token [1,1,3]; the store must return the full [1,3,3] latent with the
// new token appended, and Offset must advance 2 → 3. State exposes the live
// latent; Reset clears it.
func TestLatentKVCache_ConcatAcrossChunks_Good(t *testing.T) {
	if !metaltest.RunMetalTests || !MetalAvailable() {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}

	c := NewLatentKVCache()

	// Prefill: latent for 2 tokens, [B=1, L=2, latentDim=3].
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	full1, _ := c.Update(a, nil, 2)
	if c.Offset() != 2 {
		t.Fatalf("offset after prefill = %d, want 2", c.Offset())
	}
	if d := full1.Dim(1); d != 2 {
		t.Fatalf("prefill latent seq-dim = %d, want 2", d)
	}

	// Decode: one more token [1,1,3]. The store concatenates along the seq axis.
	b := FromValues([]float32{7, 8, 9}, 1, 1, 3)
	full2, _ := c.Update(b, nil, 1)
	if c.Offset() != 3 {
		t.Fatalf("offset after decode = %d, want 3", c.Offset())
	}
	if d := full2.Dim(1); d != 3 {
		t.Fatalf("decode latent seq-dim = %d, want 3 (prior 2 + new 1)", d)
	}
	got := full2.Floats()
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	if len(got) != len(want) {
		t.Fatalf("concatenated latent len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("concatenated latent = %v, want %v (prefill then decode token)", got, want)
		}
	}

	if st := c.State(); len(st) != 1 {
		t.Fatalf("State() len = %d, want 1 (the live latent)", len(st))
	}
	c.Reset()
	if c.Offset() != 0 || c.State() != nil {
		t.Fatalf("after Reset: offset=%d state=%v, want 0 / nil", c.Offset(), c.State())
	}
}
