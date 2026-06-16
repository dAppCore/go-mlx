// SPDX-Licence-Identifier: EUPL-1.2

package chat

import "testing"

// The registry is how a model family's chat package contributes its formatter
// without the neutral chat package naming the family. Format dispatches on the
// resolved template name; an unregistered name falls back to the plain renderer.

func TestRegistry_RegisterFormatter_Good(t *testing.T) {
	// Happy path (moved from TestRegisterFormatter_DispatchesByTemplateName_Good):
	// a formatter registered under a template name is dispatched by Format
	// when that template is selected.
	RegisterFormatter("testfmt-good", func(messages []Message, _ Config) string {
		return "FMT:" + messages[0].Content
	})
	got := Format([]Message{{Role: "user", Content: "x"}}, Config{Template: "testfmt-good"})
	if got != "FMT:x" {
		t.Fatalf("RegisterFormatter dispatch = %q, want %q", got, "FMT:x")
	}
}

func TestRegistry_RegisterFormatter_Bad(t *testing.T) {
	// Error-shaped input (moved from
	// TestRegisterFormatter_UnregisteredFallsBackToPlain_Good): a template
	// name that was never passed to RegisterFormatter has no entry, so Format
	// falls back to the plain renderer rather than panicking on a nil map
	// value.
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Template: "nope-unregistered", NoGenerationPrompt: true})
	if got != "hi\n" {
		t.Fatalf("unregistered template after RegisterFormatter = %q, want plain %q", got, "hi\n")
	}
}

func TestRegistry_RegisterFormatter_Ugly(t *testing.T) {
	// Edge case: re-registering the same name overwrites the prior formatter
	// (last writer wins), so a family package re-running its init() — or a
	// test rebinding a name — sees the most recent function, not the first.
	name := "testfmt-ugly"
	RegisterFormatter(name, func(_ []Message, _ Config) string { return "FIRST" })
	RegisterFormatter(name, func(_ []Message, _ Config) string { return "SECOND" })
	got := Format([]Message{{Role: "user", Content: "x"}}, Config{Template: name})
	if got != "SECOND" {
		t.Fatalf("RegisterFormatter re-register = %q, want last-writer SECOND", got)
	}
}
