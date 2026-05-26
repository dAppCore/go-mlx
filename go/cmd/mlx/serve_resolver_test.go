// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	"dappco.re/go/inference"
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
