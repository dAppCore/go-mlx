// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// TestCandidateToMLXLoadOpts_AllFields — every tuned-profile field
// must produce a matching mlx.LoadOption. The count check is the
// regression guard: if a future TuningCandidate field is added and
// not mapped here, the test still passes but the count flags the
// drift on review. The applied-config test below catches the real
// content via apply.
func TestCandidateToMLXLoadOpts_AllFields(t *testing.T) {
	c := inference.TuningCandidate{
		ContextLength:        8192,
		ParallelSlots:        4,
		PromptCache:          true,
		PromptCacheMinTokens: 64,
		CachePolicy:          "rotating",
		CacheMode:            "fp16",
		BatchSize:            32,
		PrefillChunkSize:     1024,
		ExpectedQuantization: 4,
		MemoryLimitBytes:     64 << 30,
		CacheLimitBytes:      8 << 30,
		WiredLimitBytes:      32 << 30,
		Adapter: inference.AdapterIdentity{
			Path: "/tmp/adapter",
		},
	}
	opts := candidateToMLXLoadOpts(c)
	if len(opts) == 0 {
		t.Fatal("expected options from fully-populated candidate, got 0")
	}
	// Each non-zero field maps to one LoadOption. The three memory caps
	// (Memory/Cache/Wired) fold into a single WithAllocatorLimits call.
	// PromptCache always produces an option (boolean — true and false
	// are both meaningful). Total: 11 expected for this fixture.
	const wantCount = 11
	if len(opts) != wantCount {
		t.Errorf("got %d options, want %d — TuningCandidate field added without mapping?", len(opts), wantCount)
	}
}

// TestCandidateToMLXLoadOpts_EmptyCandidate — zero-value candidate
// still emits the PromptCache(false) option since it's the only
// boolean. All other fields are zero-skip. Count check catches drift.
func TestCandidateToMLXLoadOpts_EmptyCandidate(t *testing.T) {
	opts := candidateToMLXLoadOpts(inference.TuningCandidate{})
	if len(opts) != 1 {
		t.Errorf("got %d options from empty candidate, want 1 (PromptCache only)", len(opts))
	}
}

// TestCandidateToMLXLoadOpts_OnlyContextLength — a sparse candidate
// (only ContextLength set, matching the pre-#79 behaviour where serve
// flowed only this field) produces ContextLength + PromptCache options.
// Documents the floor case.
func TestCandidateToMLXLoadOpts_OnlyContextLength(t *testing.T) {
	c := inference.TuningCandidate{ContextLength: 4096}
	opts := candidateToMLXLoadOpts(c)
	if len(opts) != 2 {
		t.Errorf("got %d options for ContextLength-only candidate, want 2 (ContextLength + PromptCache)", len(opts))
	}
}

// TestHotSwapResolver_ReloadPreservesTunedOpts guards Mantis #1785
// (F-7 N-7): a reload that only carries a per-request option (e.g.
// ContextLength) must keep the auto-tuned boot options rather than
// reloading with bare defaults. reloadLoadOpts overlays the new opts on
// top of initOpts, so the merged slice contains every base option plus
// the overlay (last-wins).
func TestHotSwapResolver_ReloadPreservesTunedOpts(t *testing.T) {
	base := candidateToMLXLoadOpts(inference.TuningCandidate{
		ContextLength: 4096,
		BatchSize:     32,
		CacheMode:     "fp16",
		PromptCache:   true,
	})
	r := newHotSwapResolver("/nonexistent/path", base)

	// Reload carries only a new context length.
	overlay := []mlx.LoadOption{mlx.WithContextLength(8192)}
	merged := r.reloadLoadOpts(overlay)

	if len(merged) != len(base)+len(overlay) {
		t.Fatalf("merged opts dropped the tuned base: got %d, want %d", len(merged), len(base)+len(overlay))
	}
	// The overlay must come last so it wins on apply.
	last := merged[len(merged)-1]
	var cfg mlx.LoadConfig
	last(&cfg)
	if cfg.ContextLength != 8192 {
		t.Errorf("overlay option not applied last: ContextLength=%d, want 8192", cfg.ContextLength)
	}

	// A nil overlay (no per-reload opts) must still preserve the full base.
	if got := len(r.reloadLoadOpts(nil)); got != len(base) {
		t.Errorf("nil overlay dropped base opts: got %d, want %d", got, len(base))
	}
}

// TestHotSwapResolver_NotNil — the resolver factory always returns a
// usable resolver (no panic on construction). The actual load is
// lazy on ResolveModel; this test exercises the factory only.
func TestHotSwapResolver_NotNil(t *testing.T) {
	r := newHotSwapResolver("/nonexistent/path", nil)
	if r == nil {
		t.Fatal("newHotSwapResolver returned nil")
	}
	if r.CurrentPath() != "/nonexistent/path" {
		t.Errorf("CurrentPath before load: got %q want %q", r.CurrentPath(), "/nonexistent/path")
	}
}
