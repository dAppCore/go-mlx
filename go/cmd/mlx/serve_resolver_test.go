// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"
)

// TestCandidateToMLXLoadOpts_AllFields — every tuned-profile field
// must produce a matching mlx.LoadOption. The count check is the
// regression guard: if a future TuningCandidate field is added and
// not mapped here, the test still passes but the count flags the
// drift on review. The applied-config test below catches the real
// content via apply.
// TestCandidateToMLXLoadOpts_EmptyCandidate — zero-value candidate
// still emits the PromptCache(false) option since it's the only
// boolean. All other fields are zero-skip. Count check catches drift.
// TestCandidateToMLXLoadOpts_OnlyContextLength — a sparse candidate
// (only ContextLength set, matching the pre-#79 behaviour where serve
// flowed only this field) produces ContextLength + PromptCache options.
// Documents the floor case.
// TestHotSwapResolver_ReloadPreservesTunedOpts guards Mantis #1785
// (F-7 N-7): a reload that only carries a per-request option (e.g.
// ContextLength) must keep the auto-tuned boot options rather than
// reloading with bare defaults. reloadLoadOpts overlays the new opts on
// top of initOpts, so the merged slice contains every base option plus
// the overlay (last-wins).
// TestHotSwapResolver_NotNil — the resolver factory always returns a
// usable resolver (no panic on construction). The actual load is
// lazy on ResolveModel; this test exercises the factory only.
func TestHotSwapResolver_NotNil(t *testing.T) {
	r := newHotSwapResolver("/nonexistent/path", "", 0, nil)
	if r == nil {
		t.Fatal("newHotSwapResolver returned nil")
	}
	if r.CurrentPath() != "/nonexistent/path" {
		t.Errorf("CurrentPath before load: got %q want %q", r.CurrentPath(), "/nonexistent/path")
	}
}
