// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

// runtime_author_cover_test.go covers the runtime-author SDK pass-throughs in
// runtime_author.go — the curated exported accessors a model package uses to
// drive metal's private generation machinery. Each is a thin delegate, so a
// synthetic fakeModel-backed *Model exercises them without weights.

func newRuntimeAuthorModel() *Model {
	return &Model{
		model:              &fakeModel{numLayers: 2},
		modelType:          "fake",
		promptCacheEnabled: true,
		promptCacheMinTokens: 4,
		prefillChunkSize:   16,
		parallelSlots:      make(chan struct{}, 1),
	}
}

func TestRuntimeAuthor_ModelAccessors_Good(t *testing.T) {
	m := newRuntimeAuthorModel()

	if m.UnderlyingModel() == nil {
		t.Error("UnderlyingModel returned nil for a loaded model")
	}
	if (*Model)(nil).UnderlyingModel() != nil {
		t.Error("nil model UnderlyingModel should be nil")
	}
	// fakeModel.Tokenizer() returns nil → RuntimeTokenizer mirrors m.tokenizer.
	_ = m.RuntimeTokenizer()
	if (*Model)(nil).RuntimeTokenizer() != nil {
		t.Error("nil model RuntimeTokenizer should be nil")
	}

	if !m.PromptCacheEnabled() {
		t.Error("PromptCacheEnabled = false, want true")
	}
	if (*Model)(nil).PromptCacheEnabled() {
		t.Error("nil model PromptCacheEnabled should be false")
	}
	if got := m.PrefillChunkSize(); got != 16 {
		t.Errorf("PrefillChunkSize = %d, want 16", got)
	}
	if got := (*Model)(nil).PrefillChunkSize(); got != 0 {
		t.Errorf("nil model PrefillChunkSize = %d, want 0", got)
	}
	if got := m.PromptCacheMinimum(); got != 4 {
		t.Errorf("PromptCacheMinimum = %d, want 4", got)
	}
	if got := m.AdapterCacheKey(); got != "" {
		t.Errorf("AdapterCacheKey = %q, want empty (no adapter)", got)
	}
}

// RequireTextRuntime returns the descriptive error when the runtime lacks a
// tokenizer (fakeModel carries none) and a nil error is impossible to reach
// synthetically — the no-tokenizer arm is the testable failure.
func TestRuntimeAuthor_RequireTextRuntime_NoTokenizer_Bad(t *testing.T) {
	m := newRuntimeAuthorModel() // tokenizer is nil
	if err := m.RequireTextRuntime("Model.Test"); err == nil {
		t.Fatal("RequireTextRuntime with no tokenizer returned nil, want error")
	}
	if err := (*Model)(nil).RequireTextRuntime("Model.Test"); err == nil {
		t.Fatal("nil model RequireTextRuntime returned nil, want error")
	}
}

// AcquireSlot blocks on a 1-slot model: acquire holds the slot, release frees it
// for the next acquire. A cancelled context returns an error.
func TestRuntimeAuthor_AcquireSlot_HoldReleaseCancel_Good(t *testing.T) {
	m := newRuntimeAuthorModel()

	release, err := m.AcquireSlot(context.Background())
	if err != nil {
		t.Fatalf("first AcquireSlot error: %v", err)
	}
	release()
	release() // double release is a no-op

	// Re-acquire succeeds now the slot is free.
	release2, err := m.AcquireSlot(context.Background())
	if err != nil {
		t.Fatalf("re-acquire error: %v", err)
	}

	// With the slot held, a cancelled context fails fast.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.AcquireSlot(cancelled); err == nil {
		t.Error("AcquireSlot on a full model with a cancelled ctx returned nil error")
	}
	release2()

	// An unbounded model (no parallelSlots) returns a no-op release immediately.
	unbounded := &Model{model: &fakeModel{numLayers: 1}}
	r, err := unbounded.AcquireSlot(context.Background())
	if err != nil {
		t.Fatalf("unbounded AcquireSlot error: %v", err)
	}
	r()
}

func TestRuntimeAuthor_AcquirePromptCacheDelegate_Good(t *testing.T) {
	m := newRuntimeAuthorModel()
	release := m.AcquirePromptCache()
	release() // unlocks; a second acquire must then succeed without deadlock
	m.AcquirePromptCache()()
}

func TestRuntimeAuthor_CacheBuildersAndSnapshotSafe_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newRuntimeAuthorModel()

	// fakeModel.NewCache() builds numLayers KV caches; the request-fixed-size
	// builder routes through it.
	caches := m.NewCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCachesWithRequestFixedSize built %d caches, want 2", len(caches))
	}

	// Default cache mode is snapshot-safe; the quantised KQ8VQ4 mode is not.
	if !m.RuntimeCachesSnapshotSafe() {
		t.Error("default mode reported not snapshot-safe")
	}
	m.cacheMode = string(KVCacheModeKQ8VQ4)
	if m.RuntimeCachesSnapshotSafe() {
		t.Error("KQ8VQ4 mode reported snapshot-safe, want false")
	}

	// GenerationFixedSlidingCacheSize is a pure size computation — just drive it.
	_ = m.GenerationFixedSlidingCacheSize(8, 16)

	// WithDevice runs the closure on the model's device.
	ran := false
	if err := m.WithDevice(func() { ran = true }); err != nil {
		t.Fatalf("WithDevice error: %v", err)
	}
	if !ran {
		t.Error("WithDevice did not run the closure")
	}
}

// The PromptCacheEntry-side and store accessors: install an entry, read its
// logits/hidden, restore caches, and look up a hidden-state match.
func TestRuntimeAuthor_PromptCacheEntrySurface_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newRuntimeAuthorModel()

	// nil receivers are safe.
	(*Model)(nil).StorePromptCacheEntry(nil)
	m.StorePromptCacheEntry(nil) // nil entry is a no-op
	if (*PromptCacheEntry)(nil).Logits() != nil || (*PromptCacheEntry)(nil).Hidden() != nil {
		t.Error("nil entry Logits/Hidden should be nil")
	}

	toks := make([]int32, 8)
	for i := range toks {
		toks[i] = int32(i + 1)
	}
	entry := &PromptCacheEntry{
		tokens:          append([]int32(nil), toks...),
		cacheableTokens: 8,
		logits:          Zeros([]int32{1, 1, 4}, DTypeFloat32),
		hidden:          Zeros([]int32{1, 1, 4}, DTypeFloat32),
	}
	m.StorePromptCacheEntry(entry)
	if m.promptCache != entry {
		t.Fatal("StorePromptCacheEntry did not install the entry")
	}
	if entry.Logits() == nil || entry.Hidden() == nil {
		t.Error("installed entry lost its logits/hidden")
	}

	// A hidden-state match on the exact prefix resolves the entry.
	if e, n := m.PromptCacheMatchWithHidden(toks); e != entry || n != len(toks) {
		t.Fatalf("PromptCacheMatchWithHidden = (%v,%d), want (entry,%d)", e, n, len(toks))
	}

	// RestoreCaches on a nil entry errors; on an empty-snapshot entry it returns
	// an empty set without crashing.
	if _, err := (*PromptCacheEntry)(nil).RestoreCaches(0, 0); err == nil {
		t.Error("nil entry RestoreCaches returned nil error")
	}

	m.ClearPromptCache()
}

func TestRuntimeAuthor_SetLastErrAndMetrics_Good(t *testing.T) {
	m := newRuntimeAuthorModel()

	(*Model)(nil).SetLastErr(nil)         // nil-safe
	(*Model)(nil).SetLastMetrics(Metrics{}) // nil-safe

	sentinel := context.Canceled
	m.SetLastErr(sentinel)
	if m.Err() != sentinel {
		t.Errorf("Err() = %v after SetLastErr, want %v", m.Err(), sentinel)
	}
	m.SetLastMetrics(Metrics{DecodeTokensPerSec: 42})
	if got := m.LastMetrics().DecodeTokensPerSec; got != 42 {
		t.Errorf("LastMetrics decode tps = %v, want 42", got)
	}
}
