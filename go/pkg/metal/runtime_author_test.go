// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"errors"
	"testing"
)

// runtime_author.go is the model-SDK accessor surface. Most symbols are thin
// pass-throughs that delegate to a loaded *Model's private machinery; those
// require a real model and are exercised by the model-package suites. The tests
// below cover the GUARD branches that are reachable WITHOUT loading a model — the
// nil-receiver and nil-argument paths each accessor defends — plus the
// PromptCacheEntry accessors whose nil behaviour is pure-Go.

func TestRuntimeAuthor_NilModelAccessors_Bad(t *testing.T) {
	var m *Model

	if got := m.UnderlyingModel(); got != nil {
		t.Fatalf("(*Model)(nil).UnderlyingModel() = %v, want nil", got)
	}
	if got := m.RuntimeTokenizer(); got != nil {
		t.Fatalf("(*Model)(nil).RuntimeTokenizer() = %v, want nil", got)
	}
	if m.PromptCacheEnabled() {
		t.Fatal("(*Model)(nil).PromptCacheEnabled() = true, want false")
	}
	if got := m.PrefillChunkSize(); got != 0 {
		t.Fatalf("(*Model)(nil).PrefillChunkSize() = %d, want 0", got)
	}

	// Setters on a nil model must be safe no-ops (no panic, nothing to store).
	m.SetLastErr(errors.New("ignored"))
	m.SetLastMetrics(Metrics{GeneratedTokens: 3})
	m.StorePromptCacheEntry(&PromptCacheEntry{})
	m.StorePromptCacheEntry(nil)
}

func TestRuntimeAuthor_PromptCacheEntryAccessors_Bad(t *testing.T) {
	// A nil entry yields nil tensors and an error from RestoreCaches rather than
	// dereferencing through nil.
	var entry *PromptCacheEntry
	if l := entry.Logits(); l != nil {
		t.Fatalf("(*PromptCacheEntry)(nil).Logits() = %v, want nil", l)
	}
	if h := entry.Hidden(); h != nil {
		t.Fatalf("(*PromptCacheEntry)(nil).Hidden() = %v, want nil", h)
	}
	caches, err := entry.RestoreCaches(0, 0)
	if err == nil {
		t.Fatal("(*PromptCacheEntry)(nil).RestoreCaches() err = nil, want a nil-entry error")
	}
	if caches != nil {
		t.Fatalf("RestoreCaches on nil entry caches = %v, want nil", caches)
	}
}

func TestRuntimeAuthor_PromptCacheEntryAccessors_Good(t *testing.T) {
	// An entry that holds neither logits nor hidden state reports nil for both —
	// the accessor distinguishes "no entry" from "entry with no cached state".
	entry := &PromptCacheEntry{}
	if l := entry.Logits(); l != nil {
		t.Fatalf("empty entry Logits() = %v, want nil", l)
	}
	if h := entry.Hidden(); h != nil {
		t.Fatalf("empty entry Hidden() = %v, want nil", h)
	}

	// RestoreCaches over an entry with no snapshots returns an empty (non-error)
	// cache set — there is nothing to restore but the call is well-formed.
	caches, err := entry.RestoreCaches(0, 0)
	if err != nil {
		t.Fatalf("RestoreCaches over empty entry err = %v, want nil", err)
	}
	if len(caches) != 0 {
		t.Fatalf("RestoreCaches over empty entry = %d caches, want 0", len(caches))
	}
}
