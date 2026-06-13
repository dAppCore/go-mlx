// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

const (
	tbOpen  = int32(100)
	tbClose = int32(101)
)

// The state machine on a realistic stream: open, three thought tokens within
// a budget of 3, then a fourth — which must be forced to the close. After the
// forced close, answer tokens pass through untouched.
func TestThinkingBudget_ForcesCloseAtCap_Good(t *testing.T) {
	tr := &thinkingBudgetTracker{budget: 3, openID: tbOpen, closeID: tbClose}

	// open + 3 thought tokens: all pass through unchanged.
	if got := tr.observe(tbOpen); got != tbOpen {
		t.Fatalf("open = %d, want %d", got, tbOpen)
	}
	for n, tok := range []int32{10, 11, 12} {
		if got := tr.observe(tok); got != tok {
			t.Fatalf("thought %d = %d, want %d (within budget)", n, got, tok)
		}
	}
	// 4th thought token: the model's choice (99) is replaced by the close.
	if got := tr.observe(99); got != tbClose {
		t.Fatalf("over-budget token = %d, want forced close %d", got, tbClose)
	}
	if !tr.forcedClose() {
		t.Fatal("forcedClose() = false after a forced close")
	}
	// Answer tokens after the forced close pass through, no further forcing.
	for _, tok := range []int32{20, 21, 22} {
		if got := tr.observe(tok); got != tok {
			t.Fatalf("answer token = %d, want %d (channel closed, no force)", got, tok)
		}
	}
}

// A model that closes on its own before the budget is never forced.
func TestThinkingBudget_NaturalCloseNoForce_Good(t *testing.T) {
	tr := &thinkingBudgetTracker{budget: 100, openID: tbOpen, closeID: tbClose}
	tr.observe(tbOpen)
	tr.observe(10)
	tr.observe(11)
	if got := tr.observe(tbClose); got != tbClose {
		t.Fatalf("natural close = %d, want %d", got, tbClose)
	}
	if got := tr.observe(20); got != 20 {
		t.Fatalf("answer = %d, want 20", got)
	}
	if tr.forcedClose() {
		t.Fatal("forcedClose() = true on a natural close")
	}
}

// A fresh channel re-arms the budget: force in the first, allow + force again
// in the second.
func TestThinkingBudget_ReArmsPerChannel_Good(t *testing.T) {
	tr := &thinkingBudgetTracker{budget: 2, openID: tbOpen, closeID: tbClose}
	tr.observe(tbOpen)
	tr.observe(10)
	tr.observe(11)
	if got := tr.observe(12); got != tbClose {
		t.Fatalf("first channel force = %d, want %d", got, tbClose)
	}
	// New channel opens — count and forced reset.
	tr.observe(tbOpen)
	tr.observe(30)
	tr.observe(31)
	if got := tr.observe(32); got != tbClose {
		t.Fatalf("second channel force = %d, want %d", got, tbClose)
	}
}

// Budget 0, unset channel tokens, equal open==close, and nil receiver are all
// inert — every token passes through.
func TestThinkingBudget_Inert_Ugly(t *testing.T) {
	cases := []*thinkingBudgetTracker{
		nil,
		{budget: 0, openID: tbOpen, closeID: tbClose},
		{budget: 3, openID: 100, closeID: 100},
	}
	for i, tr := range cases {
		for _, tok := range []int32{tbOpen, 10, 11, 12, 13, 14} {
			if got := tr.observe(tok); got != tok {
				t.Fatalf("case %d: token %d altered to %d by an inert tracker", i, tok, got)
			}
		}
		if tr.forcedClose() {
			t.Fatalf("case %d: inert tracker reports a forced close", i)
		}
	}
}

// Budget never opens (thinking off — no open token ever seen): a long stream
// of plain tokens is never touched.
func TestThinkingBudget_NeverOpensNeverForces_Good(t *testing.T) {
	tr := &thinkingBudgetTracker{budget: 1, openID: tbOpen, closeID: tbClose}
	for _, tok := range []int32{5, 6, 7, 8, 9, 10, 11, 12} {
		if got := tr.observe(tok); got != tok {
			t.Fatalf("no channel ever opened, yet token %d altered to %d", tok, got)
		}
	}
	if tr.forcedClose() {
		t.Fatal("forced without ever entering the thought channel")
	}
}
