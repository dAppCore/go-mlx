// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// mixer_compaction_cover_test.go covers the small mixer-context + compaction-cache
// helpers the heavier suites skip: MixerComputeFor resolution, the MixerCtx
// recurrent type-assert, and the CompactionKVCache state-append / detach lifecycle.

func TestMixerComputeFor_UnknownKind_Bad(t *testing.T) {
	if _, ok := MixerComputeFor("not-a-mixer-kind"); ok {
		t.Error("MixerComputeFor resolved an unregistered mixer kind")
	}
}

// MixerCtx.Recurrent type-asserts the held cache: a RecurrentCache resolves, a
// plain KV cache does not.
func TestMixerCtx_Recurrent_BothArms_Good(t *testing.T) {
	requireMetalRuntime(t)

	rc := NewRecurrentCache()
	ctx := &MixerCtx{Cache: rc}
	if got, ok := ctx.Recurrent(); !ok || got != rc {
		t.Errorf("recurrent-cache ctx Recurrent() = (%v,%v), want (rc,true)", got, ok)
	}

	kv := NewKVCache()
	defer FreeCaches([]Cache{kv})
	ctx2 := &MixerCtx{Cache: kv}
	if _, ok := ctx2.Recurrent(); ok {
		t.Error("KV-cache ctx Recurrent() reported true, want false")
	}
}

// CompactionKVCache state accessors across the empty and populated shapes.
func TestCompactionKVCache_StateAndDetach_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewCompactionKVCache(64, 0)
	// Empty cache: nil receiver-safe and empty-state behaviour.
	if got := c.AppendState(nil); got != nil {
		t.Errorf("empty compaction AppendState = %v, want nil", got)
	}
	c.Detach() // empty detach is a no-op
	if c.State() != nil {
		t.Error("empty compaction State() should be nil")
	}

	// Populate via Update, then the accessors expose the held window.
	k, v := makeKV(3)
	c.Update(k, v, 3)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	if got := c.AppendState(nil); len(got) != 2 {
		t.Fatalf("populated compaction AppendState len = %d, want 2", len(got))
	}
	if st := c.State(); len(st) != 2 {
		t.Fatalf("populated compaction State() len = %d, want 2", len(st))
	}
	c.Detach() // populated detach → graph-parentless copies, must not crash
	if c.Len() != 3 {
		t.Errorf("compaction Len after detach = %d, want 3", c.Len())
	}

	// Nil-receiver guards.
	var nilCache *CompactionKVCache
	if got := nilCache.AppendState(nil); got != nil {
		t.Error("nil compaction AppendState should return nil")
	}
	nilCache.Detach()
}
