// SPDX-Licence-Identifier: EUPL-1.2

// Coverage tests for the gemma4 chat formatter — the edge branches the
// behavioural suite in gemma4chat_test.go does not exercise: a real
// system-first message consumed into the <|think|> opening, an unmapped
// role skipped mid-history, an empty message list reaching
// initialSystemRole, and the full roleFromNormalised contract (the
// "developer" arm is only reachable by calling the internal directly,
// since chat.NormaliseRole collapses developer→system before
// roleFromNormalised ever sees it).
//
// Same-package (white-box) like the bench test, so the unexported helpers
// roleFromNormalised / initialSystemRole are callable directly.

package gemma4chat

import (
	"testing"

	"dappco.re/go/mlx/chat"
)

// TestFormat_SystemFirstMessageConsumedIntoThinkOpening_Good drives the
// system branch where messages[0] is an actual system turn: its content is
// trimmed into the <|think|> system block and start advances past it, so the
// turn loop renders only the following turns. Covers the role=="system"
// consume path in Format (the existing thinking tests pass only a user turn,
// so messages[0] is never a system message there).
func TestFormat_SystemFirstMessageConsumedIntoThinkOpening_Good(t *testing.T) {
	got := Format([]chat.Message{
		{Role: "system", Content: "  be terse  "},
		{Role: "user", Content: "hi"},
	}, chat.Config{EnableThinking: true})
	want := "<bos><|turn>system\n<|think|>\nbe terse<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("system-first thinking render = %q, want %q", got, want)
	}
}

// TestFormat_SystemFirstMessageThinkingOff_Good reaches the same system
// consume path with thinking off — the system block opens because
// initialSystemRole(messages) is true, not because EnableThinking is set, so
// no <|think|> line is emitted but messages[0] is still consumed as system.
func TestFormat_SystemFirstMessageThinkingOff_Good(t *testing.T) {
	got := Format([]chat.Message{
		{Role: "system", Content: "system rules"},
		{Role: "user", Content: "hi"},
	}, chat.Config{})
	want := "<bos><|turn>system\nsystem rules<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("system-first thinking-off render = %q, want %q", got, want)
	}
}

// TestFormat_UnmappedRoleSkipped_Good drives the role=="" continue inside the
// turn loop: a message whose normalised role maps to nothing (a tool turn
// here) is dropped, leaving only the user and assistant turns rendered.
func TestFormat_UnmappedRoleSkipped_Good(t *testing.T) {
	got := Format([]chat.Message{
		{Role: "user", Content: "hi"},
		{Role: "tool", Content: "should be dropped"},
		{Role: "assistant", Content: "answer"},
	}, chat.Config{NoGenerationPrompt: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\nanswer<turn|>\n"
	if got != want {
		t.Fatalf("unmapped-role render = %q, want %q", got, want)
	}
}

// TestFormat_EmptyMessagesThinkingOff_Good reaches initialSystemRole with an
// empty slice (its len==0 → false branch): with thinking off the system block
// is gated solely on initialSystemRole, so an empty list renders just BOS plus
// the generation prompt.
func TestFormat_EmptyMessagesThinkingOff_Good(t *testing.T) {
	got := Format(nil, chat.Config{})
	want := "<bos><|turn>model\n"
	if got != want {
		t.Fatalf("empty-messages thinking-off render = %q, want %q", got, want)
	}
}

// TestFormat_EmptyMessagesThinkingOn_Good is the companion empty-list case
// with thinking on: initialSystemRole is short-circuited by EnableThinking, the
// system/think block opens, and the len(messages)>0 guard keeps the consume
// path from indexing the empty slice.
func TestFormat_EmptyMessagesThinkingOn_Good(t *testing.T) {
	got := Format(nil, chat.Config{EnableThinking: true})
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("empty-messages thinking-on render = %q, want %q", got, want)
	}
}

// TestRoleFromNormalised_Contract pins the full normalised-role → template-role
// mapping. It is called directly (white-box) because chat.NormaliseRole
// collapses developer→system upstream, so the developer arm of
// roleFromNormalised is unreachable through Format and only this direct call
// covers it; the unknown→"" default and the assistant/system/user arms are
// pinned in the same table.
func TestRoleFromNormalised_Contract(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"assistant", "model"},
		{"system", "system"},
		{"developer", "system"},
		{"user", "user"},
		{"tool", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := roleFromNormalised(c.in); got != c.want {
			t.Errorf("roleFromNormalised(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestInitialSystemRole_Contract covers initialSystemRole directly across its
// three outcomes: empty (false), system-first (true), and non-system-first
// (false). The empty case is the same branch the empty-list Format test hits;
// pinning it here documents the helper's contract independent of Format.
func TestInitialSystemRole_Contract(t *testing.T) {
	if initialSystemRole(nil) {
		t.Errorf("initialSystemRole(nil) = true, want false")
	}
	if !initialSystemRole([]chat.Message{{Role: "system", Content: "x"}}) {
		t.Errorf("initialSystemRole(system-first) = false, want true")
	}
	if initialSystemRole([]chat.Message{{Role: "user", Content: "x"}}) {
		t.Errorf("initialSystemRole(user-first) = true, want false")
	}
}
